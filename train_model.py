from utls import *
import time

global last_epoch, best_prec


def train(train_loader, net, criterion, optimizer, epoch, device):
    start = time.time()
    net.train()
    train_loss = 0
    global_correct = 0
    local_correct = 0
    total = 0

    print("===  Epoch: [{}/{}]  === ".format(epoch + 1, opt.epochs))
    for batch_index, (img_cat, label, obj_box, refer_box, depth_fea, w, h) in enumerate(train_loader):
        img_cat, label, obj_box, refer_box, depth_fea, w, h = img_cat.to(device), label.to(device), obj_box.to(
            device), refer_box.to(device), depth_fea.to(device), w.to(device), h.to(device)
        global_pre, local_pre = net(img_cat, obj_box, refer_box, depth_fea, w, h)

        loss1 = criterion(global_pre, label)
        loss2 = criterion(local_pre, label)
        loss = loss1 + opt.loss_coefficient * loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, global_predicted = global_pre.max(1)
        _, local_predicted = local_pre.max(1)

        total += label.size(0)
        global_correct += global_predicted.eq(label).sum().item()
        local_correct += local_predicted.eq(label).sum().item()
        if (batch_index + 1) % 100 == 0:  # TODO
            print(
                "===  step: [{:3}/{}], train loss: {:.3f} | global acc: {:6.3f}% | local acc: {:6.3f}% | lr: {:.6f}  ===".format(
                    batch_index + 1, len(train_loader), train_loss / (batch_index + 1), 100.0 * global_correct / total,
                    100.0 * local_correct / total, get_current_lr(optimizer)))
    print(
        "===  step: [{:3}/{}], train loss: {:.3f} | global acc: {:6.3f}% | local acc: {:6.3f}% | lr: {:.6f}  ===".format(
            len(train_loader), len(train_loader), train_loss / len(train_loader), 100.0 * global_correct / total,
                                                  00.0 * local_correct / total, get_current_lr(optimizer)))

    end = time.time()
    print("===  cost time: {:.4f}s  ===".format(end - start))


def test(test_loader, net, criterion, optimizer, epoch, device):
    global best_prec
    net.eval()
    test_loss = 0
    global_correct = 0
    local_correct = 0
    total = 0

    print("===  Validate [{}/{}] ===".format(epoch + 1, opt.epochs))
    with torch.no_grad():
        for batch_index, (img_cat, label, obj_box, refer_box, depth_fea, w, h) in enumerate(test_loader):
            img_cat, label, obj_box, refer_box, depth_fea, w, h = img_cat.to(device), label.to(device), obj_box.to(
                device), refer_box.to(device), depth_fea.to(device), w.to(device), h.to(device)
            global_pre, local_pre = net(img_cat, obj_box, refer_box, depth_fea, w, h)

            loss1 = criterion(global_pre, label)
            loss2 = criterion(local_pre, label)
            loss = loss1 + opt.loss_coefficient * loss2
            test_loss += loss.item()

            _, global_predicted = global_pre.max(1)
            _, local_predicted = local_pre.max(1)

            total += label.size(0)
            global_correct += global_predicted.eq(label).sum().item()
            local_correct += local_predicted.eq(label).sum().item()

    print("===  test loss: {:.3f} |  global acc: {:6.3f}% | local acc: {:6.3f}%  ===".format(
        test_loss / (batch_index + 1), 100.0 * global_correct / total, 100.0 * local_correct / total))

    acc = 100. * local_correct / total
    state = {
        'state_dict': net.state_dict(),
        'best_prec': best_prec,
        'last_epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    is_best = acc > best_prec
    save_checkpoint(state, is_best, opt.model_path + opt.ckpt_name)
    if is_best:
        best_prec = acc


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = pro_net()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), opt.base_lr)

    if opt.ifcontinue:
        ckpt_file_name = opt.model_path + opt.ckpt_name + '.pth.tar'
        best_prec, last_epoch, optimizer = load_checkpoint(ckpt_file_name, net, optimizer=optimizer)
    else:
        last_epoch = -1
        best_prec = 0

    train_loader, test_loader = get_data_loader()
    print("load")

    print(("=======  Training  ======="))
    for epoch in range(last_epoch + 1, opt.epochs):
        lr = adjust_learning_rate(optimizer, epoch)
        train(train_loader, net, criterion, optimizer, epoch, device)
        if epoch == 0 or (epoch + 1) % opt.eval_freq == 0 or epoch == opt.epochs - 1:
            test(test_loader, net, criterion, optimizer, epoch, device)
    print(("=======  Training Finished.Best_test_acc: {:.3f}% ========".format(best_prec)))
