# coding:utf-8
from __future__ import division
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from nets.centernet_training import focal_loss, reg_l1_loss
from nets.centernet import CenterNet_Resnet50
from torch.utils.data import DataLoader
from utils.early_stopping import EarlyStopping
from utils.dataloader import CenternetDataset, centernet_dataset_collate
from tqdm import tqdm

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, gen_val, Epoch, cuda):
    total_r_loss = 0
    total_c_loss = 0
    total_loss = 0
    val_loss = 0
    start_time = time.time()

    net.train()
    with tqdm(total=epoch_size, desc='Epoch {}/{}'.format((epoch + 1), Epoch), postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break

            with torch.no_grad():
                if cuda:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in batch]

            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

            optimizer.zero_grad()

            hm, wh, offset = net(batch_images)
            c_loss = focal_loss(hm, batch_hms)
            wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

            loss = c_loss + wh_loss + off_loss

            total_loss += loss.item()
            total_c_loss += c_loss.item()
            total_r_loss += wh_loss.item() + off_loss.item()

            loss.backward()
            optimizer.step()

            waste_time = time.time() - start_time

            pbar.set_postfix(**{'Total_Loss'   : total_loss / (iteration + 1),
                                'lr'           : get_lr(optimizer),
                                'step/s'       : waste_time})
            pbar.update(1)
            start_time = time.time()

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc='Epoch {}/{}'.format((epoch + 1), Epoch), postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_size_val:
                break

            with torch.no_grad():
                if cuda:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in batch]

                batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

                optimizer.zero_grad()
                hm, wh, offset = net(batch_images)
                c_loss = focal_loss(hm, batch_hms)
                wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                loss = c_loss + wh_loss + off_loss

                val_loss += loss.item()

            pbar.set_postfix(**{'Val_Loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % ((epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    return val_loss / (epoch_size_val + 1)

if __name__ == "__main__":

    input_shape = (512, 512, 3)
    train_path = '2007_train.txt'
    val_path = '2007_val.txt'

    classes_path = 'model_data/classesvoc.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    Cuda = True
    Resume = False
    pretrain = False

    model = CenterNet_Resnet50(num_classes, pretrain=pretrain)

    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load("model_data/centernet_resnet50_voc.pth", map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    with open(train_path) as f1:
        lines1 = f1.readlines()
    with open(val_path) as f2:
        lines2 = f2.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines1)
    np.random.seed(None)
    np.random.seed(10101)
    np.random.shuffle(lines2)
    np.random.seed(None)

    num_val = int(len(lines2))
    num_train = int(len(lines1))

    early_stopping = EarlyStopping(patience=8, verbose=True)

    if True:
        # 最开始使用1e-3的学习率可以收敛的更快
        lr = 1e-3
        Batch_size = 16
        Init_Epoch = 0
        Freeze_Epoch = 50

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        train_dataset = CenternetDataset(lines1, input_shape, num_classes)
        val_dataset = CenternetDataset(lines2, input_shape, num_classes)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=1, pin_memory=True,
                         drop_last=True, collate_fn=centernet_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=1, pin_memory=True,
                             drop_last=True, collate_fn=centernet_dataset_collate)

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size

        # 冻结一定部分训练
        model.unfreeze_backbone()

        for epoch in range(Init_Epoch, Freeze_Epoch):
            valloss = fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
            lr_scheduler.step(valloss)

    if True:
        lr = 1e-4
        Batch_size = 16
        Freeze_Epoch = 50
        Unfreeze_Epoch = 300

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        if Resume:
            print('Resume from checkpoint...')
            checkpoint = torch.load('./checkpoint/checkpoint.pkl')
            Init_Epoch = checkpoint['epoch'] + 1
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.last_epoch = Init_Epoch - Freeze_Epoch

        train_dataset = CenternetDataset(lines1, input_shape, num_classes)
        val_dataset = CenternetDataset(lines2, input_shape, num_classes)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=1, pin_memory=True,
                         drop_last=True, collate_fn=centernet_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=1, pin_memory=True,
                             drop_last=True, collate_fn=centernet_dataset_collate)

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size

        # 解冻后训练
        model.unfreeze_backbone()

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            valloss = fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
            lr_scheduler.step(valloss)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            path_checkpoint = './checkpoint/checkpoint.pkl'
            torch.save(checkpoint, path_checkpoint)
            early_stopping(valloss)
            if early_stopping.early_stop:
                print('Early Stopping!')
                break