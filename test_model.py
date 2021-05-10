from utls import *


def F1(preds, gts):
    tp = sum(list(map(lambda a, b: a == 1 and b == 1, preds, gts)))
    fp = sum(list(map(lambda a, b: a == 1 and b == 0, preds, gts)))
    fn = sum(list(map(lambda a, b: a == 0 and b == 1, preds, gts)))
    tn = sum(list(map(lambda a, b: a == 0 and b == 0, preds, gts)))
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    f1 = (2 * tp) / (2 * tp + fp + fn)
    bal_acc = (tpr + tnr) / 2
    return f1, bal_acc


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = pro_net()
    ckpts_path = opt.model_path + opt.ckpt_name + '.pth.tar'
    net_dict = net.state_dict()
    state_dict = torch.load(ckpts_path)['state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net_dict.update(new_state_dict)
    net.load_state_dict(net_dict)
    net.to(device)
    net.eval()

    train_loader, test_loader = get_data_loader()
    print("load")

    global_preds = []
    local_preds = []
    gts = []
    with torch.no_grad():
        for batch_index, (img_cat, label, obj_box, refer_box, depth_fea, w, h) in enumerate(test_loader):
            img_cat, label, obj_box, refer_box, depth_fea, w, h = img_cat.to(device), label.to(device), obj_box.to(
                device), refer_box.to(device), depth_fea.to(device), w.to(device), h.to(device)
            global_pre, local_pre = net(img_cat, obj_box, refer_box, depth_fea, w, h)

            global_preds.extend(global_pre.max(1)[1].cpu().numpy())
            local_preds.extend(local_pre.max(1)[1].cpu().numpy())
            gts.extend(label.cpu().numpy())

    glocal_f1, global_bal_acc = F1(global_preds, gts)
    local_f1, local_bal_acc = F1(local_preds, gts)
    print("global : f1={:.3f},bal_acc={:.3f}".format(glocal_f1, global_bal_acc))
    print("local : f1={:.3f},bal_acc={:.3f}".format(local_f1, local_bal_acc))
