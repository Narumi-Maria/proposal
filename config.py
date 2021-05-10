class Config(object):
    # path
    train_data_path = r'/tmp/pycharm_project_60/data/train_data.csv'
    test_data_path = r'/tmp/pycharm_project_60/data/test_data.csv'
    img_path = r'/tmp/pycharm_project_60/data/images/'
    mask_path = r'/tmp/pycharm_project_60/data/'
    box_dic_path = r'/tmp/pycharm_project_60/data/train_box.npy'
    test_box_dic_path = r'/tmp/pycharm_project_60/data/test_box.npy'
    depth_feats_path = r'/tmp/pycharm_project_60/data/train_depth_feats.npy'
    test_depth_feats_path = r'/tmp/pycharm_project_60/data/test_depth_feats.npy'
    model_path = r'/tmp/pycharm_project_60/ckpt/'

    # parameters
    class_num = 2
    refer_class_num = 1601
    img_size = 256
    batch_size = 128
    num_workers = 4
    global_feature_size = 8
    global_feature_ch = 512
    aggregated_reference = 3  # reference dim
    depth_pool_size = 7
    res = {'inplanes': 2, 'planes': 512, 'expansion': 2, 'blocks': 3}  # res50-layer5
    concatenated_dim = 2048
    local_fea_dim = 1024
    attention_head = 16
    attention_dim_head = 64
    attention_dropout = 0.1

    # train
    base_lr = 0.0001
    lr_scheduler_type = 'STEP'
    lr_scheduler_lr_epochs = [5, 8]
    lr_scheduler_lr_mults = 0.1
    lr_scheduler_min_lr = 0.0
    lr_scheduler_lower_bound = -6.0
    lr_scheduler_upper_bound = 3.0
    ifcontinue = False
    ckpt_name = 'Net'
    epochs = 10
    eval_freq = 2
    loss_coefficient = 1




opt = Config()
