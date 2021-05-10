import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from config import opt
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from utls import *
from torchsummary import summary
from resnet_4ch import *


# ROI Align
def roi_align(img, boxes, w, h, obj=False, outsize=opt.aggregated_reference, insize=opt.global_feature_size):
    '''  img:b,c,256,256  boxes:b,5,4/b,4  w:b  h:b  '''
    boxes_ = boxes.clone()
    if obj:  # copy foreground box five times
        boxes_ = boxes_.repeat(1, 5)
        boxes_ = boxes_.reshape(boxes.shape[0], 5, 4)
    # Resize the box
    scaled_boxes = torch.zeros(boxes_.shape)
    for i in range(boxes.shape[0]):
        scaled_boxes[i, :, 0] = boxes_[i, :, 0] * (insize / int(w[i]))
        scaled_boxes[i, :, 2] = boxes_[i, :, 2] * (insize / int(w[i]))
        scaled_boxes[i, :, 1] = boxes_[i, :, 1] * (insize / int(h[i]))
        scaled_boxes[i, :, 3] = boxes_[i, :, 3] * (insize / int(h[i]))

    box_list = list(torch.split(scaled_boxes, 1, dim=0))
    for i in range(len(box_list)):
        box_list[i] = torch.squeeze(box_list[i], dim=0).cuda()

    pooled_regions = torchvision.ops.roi_align(img, box_list,
                                               output_size=(outsize, outsize))
    return pooled_regions


# ADD:partial depth map
def depth_mask(img, boxes, w, h, obj=False, outsize=opt.depth_pool_size, insize=opt.img_size):
    '''  img:b,1,256,256  boxes:b,5,4/b,4  w:b  h:b  '''
    boxes_ = boxes.clone()
    if obj:  # copy foreground box five times
        boxes_ = boxes_.repeat(1, 5)
        boxes_ = boxes_.reshape(boxes.shape[0], 5, 4)
    # Resize the box
    scaled_boxes = torch.zeros(boxes_.shape)
    for i in range(boxes.shape[0]):
        scaled_boxes[i, :, 0] = boxes_[i, :, 0] * (insize / int(w[i]))
        scaled_boxes[i, :, 2] = boxes_[i, :, 2] * (insize / int(w[i]))
        scaled_boxes[i, :, 1] = boxes_[i, :, 1] * (insize / int(h[i]))
        scaled_boxes[i, :, 3] = boxes_[i, :, 3] * (insize / int(h[i]))
    depth_map = torch.zeros(boxes.shape[0], 5, 1, insize, insize)
    for i in range(boxes.shape[0]):
        for j in range(5):
            left = int(scaled_boxes[i, j, 0])
            top = int(scaled_boxes[i, j, 1])
            right = int(scaled_boxes[i, j, 2])
            bottom = int(scaled_boxes[i, j, 3])
            depth_map[i, j, :, top:bottom, left:right] = img[i, :, top:bottom, left:right]
    depth_map = depth_map.view(boxes.shape[0] * 5, 1, insize, insize)
    avgpool = nn.AdaptiveAvgPool2d(outsize)
    depth_map = avgpool(depth_map)
    return depth_map


class _PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    # Make a layer normalization for the input x, and then put it into the attention module for self attention
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class _Attention(nn.Module):
    def __init__(self, dim, heads=16, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 1024
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x:b,5,1024
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # multi-head
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # Calculate h attention matrices
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # b,16,5,5
        attn = self.attend(dots)  # b,16,5,5
        attn = rearrange(attn, 'b h n j -> b n (j h)')  # b,5,80
        return attn


class _Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class pro_net(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone
        backbone = resnet18(pretrained=True)
        # drop pool layer and fc layer, resnet18 layer4 output shape: b,512,8,8
        features = list(backbone.children())[:-2]
        backbone = nn.Sequential(*features)
        self.backbone = backbone

        # global predict
        self.fc_global = nn.Linear(opt.global_feature_ch, opt.class_num)  # 512*2

        # reference class
        self.fc_refer_class = nn.Linear(opt.global_feature_ch, opt.refer_class_num)  # 512*1601

        # roi
        self.roi_align = roi_align

        # depth map
        self.depth_mask = depth_mask

        # encoder middle layer: res50-layer5 (Changed the output dimension from 2048 to 1024)
        downsample = nn.Sequential(
            nn.Conv2d(opt.res['inplanes'], opt.res['planes'] * opt.res['expansion'],
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(opt.res['planes'] * opt.res['expansion']),
        )
        layers = []
        layers.append(_Bottleneck(opt.res['inplanes'], opt.res['planes'], 1, downsample))
        for i in range(1, opt.res['blocks']):
            layers.append(_Bottleneck(opt.res['planes'] * opt.res['expansion'], opt.res['planes']))
        self.encoder_middle_layer = nn.Sequential(*layers)
        # encoder GAP
        self.avgpool = nn.AdaptiveAvgPool2d(3)

        # local_feature to fi
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.fc_loacal_feature = nn.Linear(opt.concatenated_dim, opt.local_fea_dim)  # 2048*1024

        # PlanB:multi-head attention
        self.refer_attention = _PreNorm(opt.local_fea_dim, _Attention(opt.local_fea_dim, heads=opt.attention_head,
                                                                      dim_head=opt.attention_dim_head,
                                                                      dropout=opt.attention_dropout))
        self.fc_weight_learn = nn.Linear(5 * opt.attention_head + opt.local_fea_dim, 1)  # 1104*1

        # local predict
        self.fc_local_pre = nn.Linear(opt.local_fea_dim, opt.class_num)  # 1024*2

    def forward(self, img_cat, obj_box, refer_box, depth_fea, w, h):
        '''  img_cat:b,4,256,256  obj_box:b,4  refer_box:b,5,6  depth_fea:b,1,256,256  w:b  h:b  '''
        # global feature
        feature_map = self.backbone(img_cat)  # b,512,8,8 (resnet layer4 output shape: b,c,8,8, if resnet18, c=512)
        global_feature = self.avgpool2(feature_map)  # b,512,1,1
        global_feature = global_feature.view(global_feature.size(0), -1)  # b,512
        global_pre = self.fc_global(global_feature)  # b,2

        # region features
        # reference_class = refer_box[:, :, -2]  # b,5,1 (for the loss of reference classification)
        # reference_class = reference_class.view(-1)  # 5b
        reference_boxes = refer_box[:, :, :4]  # b,5,4
        reference_feature = self.roi_align(feature_map, reference_boxes, w, h)  # 5b,512,3,3
        object_feature = self.roi_align(feature_map, obj_box, w, h, obj=True)  # 5b,512,3,3
        region_feature = torch.cat((reference_feature, object_feature), dim=1)  # 5b,1024,3,3
        # refer_region_pre = self.avgpool2(reference_feature)  # 5b,512,1,1
        # refer_region_pre = refer_region_pre.view(refer_region_pre.size(0), -1)  # 5b,512
        # refer_region_pre = self.fc_refer_class(refer_region_pre)  # 5b,1601

        # depth features
        depth_reference_feature = self.depth_mask(depth_fea, reference_boxes, w, h, obj=False,
                                                  outsize=opt.depth_pool_size,
                                                  insize=opt.img_size)  # 5b,1,7,7
        depth_object_feature = self.depth_mask(depth_fea, obj_box, w, h, obj=True, outsize=opt.depth_pool_size,
                                               insize=opt.img_size)  # 5b,1,7,7
        # depth_reference_feature = self.roi_align(depth_fea, reference_boxes, w, h, obj=False,
        #                                          outsize=opt.depth_pool_size,
        #                                          insize=opt.img_size)  # 5b,1,7,7
        # depth_object_feature = self.roi_align(depth_fea, obj_box, w, h, obj=True, outsize=opt.depth_pool_size,
        #                                       insize=opt.img_size)  # 5b,1,7,7
        depth_feature = torch.cat((depth_reference_feature, depth_object_feature), dim=1)  # 5b,2,7,7
        depth_feature = self.encoder_middle_layer(depth_feature.cuda())  # 5b,1024,7,7
        depth_feature = self.avgpool(depth_feature)  # 5b,1024,3,3

        # local feature
        local_feature = torch.cat((region_feature, depth_feature), dim=1)  # 5b,2048,3,3
        local_feature = self.avgpool2(local_feature)  # 5b,2048,1,1
        local_feature = local_feature.view(img_cat.size(0), 5, -1)  # b,5,2048
        local_feature = self.fc_loacal_feature(local_feature)  # b,5,1024

        # PlanA：average pooling
        local_feature = torch.mean(local_feature, dim=1)  # b,1024
        local_pre = self.fc_local_pre(local_feature)  # b,2

        # # PlanB: attention
        # similarity_vector = self.refer_attention(local_feature)  # b,5,80
        # similarity_weight = torch.cat((local_feature, similarity_vector), dim=2)  # b,5,1104
        # similarity_weight = self.fc_weight_learn(similarity_weight)  # b,5,1
        # # ADD: weight normalize
        # similarity_weight = nn.functional.normalize(similarity_weight, p=2, dim=1)
        # similarity_weight = similarity_weight.repeat(1, 1, opt.local_fea_dim)  # b,5,1024
        # aggregated_local_feature = torch.zeros(img_cat.size(0), opt.local_fea_dim)  # b,1024
        # for i in range(5):
        #     aggregated_local_feature += similarity_weight[:, i, :] * local_feature[:, i, :]
        # local_pre = self.fc_local_pre(aggregated_local_feature)  # b,2

        return global_pre, local_pre  # , refer_region_pre, reference_class


if __name__ == '__main__':
    net = pro_net().encoder_middle_layer
    print(list(net.children()))
