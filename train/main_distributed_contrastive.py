import argparse
import glob
import os
import shutil
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
#from monai.networks.nets.resnet import resnet50
import torch.nn.functional as F
import numpy as np
# from monai.networks.nets import SEresnet50,ViT
from model import Seresnet50_Contrastive
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import transform as custom_transform
from dataset import Vertebrae_Dataset, ContrastiveBatchSampler
from utils import CustomLogger, calculate_confusion_matrix
import time
# from sklearn import metrics as skm
from losses import SupConLoss
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', '-t', type=str, required=False, default="debug") # name of a experiment
    parser.add_argument('--ckpt_root', '-cp', type=str, required=False, default=f"/ckpt/ckpt2")# path of checkpoint
    parser.add_argument('--tensorboard', '-tb', type=str, required=False, default=f"/ckpt/tensorboard2")# path of tensorboard
    parser.add_argument('--dataset', '-d', type=str, required=False, default=f"/dataset")# path of dataset,refer to pre-process folder
    parser.add_argument('--dataset_tag', '-dt', type=str, required=False, default="delx")# tag of dataset file list
    parser.add_argument("--load_ckpt", action="store_true")
    args = parser.parse_args()
    ####Handcraft Tag####
    # args.tag = "SEnet50_Contrastive_Alter_Enhanced_Aug_SGD_linear"
    args.ckpt_path = os.path.join(args.ckpt_root, args.tag)

    if args.tag == "debug":
        print("Debug mode, won't save any checkpoint")
        args.ckpt_path = os.path.join('/tmp', 'debug_ckpt')
        args.tensorboard = os.path.join('/tmp', 'debug_tensorboard')
        # shutil.rmtree(args.ckpt_path, ignore_errors=True)
        # shutil.rmtree(args.tensorboard, ignore_errors=True)
    #####################
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    torch.cuda.set_device(local_rank)
    writer = None
    test_writer = None

    epoch_init = 0
    g_step = 0

    if local_rank == 0:
        os.makedirs(args.ckpt_path, exist_ok=True)
        train_code_save = os.path.join(args.ckpt_path, "train_code")
        # backup the training code
        shutil.copytree(".", train_code_save)
        
    while not os.path.exists(args.ckpt_path):
        time.sleep(0.5)
    logger = CustomLogger(os.path.join(args.ckpt_path, "test_log.log"), os.path.join(args.ckpt_path, "test_log.csv"))

    # For sdata
    t = transforms.Compose([
        custom_transform.RandomMask3D(20, 2, 0.5),
        transforms.RandomApply([
        custom_transform.RandomColorScale3D(0.1),
        custom_transform.RandomNoise3D(0.05),
        custom_transform.RandomRotation3D(10),
        custom_transform.RandomZoom3D(0.2),
        custom_transform.RandomShift3D(10),
        ], p=0.7),
        custom_transform.RandomAlign3D(128),
        custom_transform.RandomMask3D(20, 2, 0.5)
    ])

    train_data = Vertebrae_Dataset(args.dataset, f"train_file_list.{args.dataset_tag}.yaml", transforms=[t,t])
    #asampler = data.RandomSampler(train_data)
    custom_batch_sample = ContrastiveBatchSampler(train_data)
    trainloader = torch.utils.data.DataLoader(train_data, num_workers=16, batch_sampler=custom_batch_sample, pin_memory=True, persistent_workers=True)

    # trainset = Vertebrae_Dataset(args.dataset, f"train_file_list.{args.dataset_tag}.yaml", transforms=t)
    # train_sampler = DistributedSampler(trainset, shuffle=True)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=24,
    #                                           num_workers=16, drop_last=True, pin_memory=True,
    #                                           sampler=train_sampler, persistent_workers=True)
    if local_rank == 0:
        t_0 = transforms.Compose([
            custom_transform.FixedAlign3D(128)
        ])
        testset = Vertebrae_Dataset(args.dataset, f"test_file_list.{args.dataset_tag}.yaml", transforms=t_0)
        test_sampler = SequentialSampler(testset)
        testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                                 num_workers=2, drop_last=False, pin_memory=True,
                                                 sampler=test_sampler, persistent_workers=True)

    if args.load_ckpt:
        checkpoint = logger.load_checkpoint(args.ckpt_path,local_rank)
        if checkpoint is None:
            args.load_ckpt = False

    if local_rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.tensorboard, args.tag))
        test_writer = SummaryWriter(log_dir=os.path.join(args.tensorboard, f"{args.tag}_test"))

    # net = resnet50(spatial_dims=3, n_input_channels=3, num_classes=4)#,pretrained=True)
    net =Seresnet50_Contrastive(spatial_dims=3, in_channels=3, num_classes=4,head='linear',feat_dim=128)

    net = nn.SyncBatchNorm.convert_sync_batchnorm(net).cuda(local_rank)

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 5) 封装
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])  # ,find_unused_parameters=True)
    if args.load_ckpt:
        net.load_state_dict(checkpoint['state_dict'])
    # else:
    #     pretrain = torch.load("/ckpt/Pretrain/resnet50_for_verse.pth")
    #     net.load_state_dict(pretrain['state_dict'], strict=False)
    #     print("Loaded pretrain model...")

    print(f'a:{local_rank}')

   # weight = torch.from_numpy(np.array([0.02, 1.0, 1.0, 1.0])).float().cuda(local_rank)
    criterion_cls = nn.CrossEntropyLoss().cuda(local_rank)
    criterion_con = SupConLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=0.005,
                          momentum=0.9,
                          weight_decay=1e-4)
    # optimizer = optim.Adam(net.parameters(), lr=0.05)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[800,900], gamma=0.1)
    confusion_matrix = torch.zeros([4, 4], device=torch.device('cuda', local_rank), requires_grad=False)

    if args.load_ckpt:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        g_step = checkpoint['g_step']
        epoch_init = g_step//len(trainloader)

    print(f'b:{local_rank}')

    dist.barrier()
    for epoch in range(5000):  # 多批次循环
        #train_sampler.set_epoch(epoch)
        net.train()
        for i, data in enumerate(trainloader, 0):
            # 获取输入
            inputs, GT, _ = data
            inputs = inputs.cuda(local_rank, non_blocking=True)
            GT = GT.cuda(local_rank, non_blocking=True)

            inputs_s1, inputs_s2 = torch.split(inputs,[128,128] , dim=3)

            inputs = torch.cat([inputs_s1, inputs_s2])
            GT2 = torch.cat([GT, GT])

            # 梯度置0
            optimizer.zero_grad()
            confusion_matrix = confusion_matrix * 0
            # 正向传播，反向传播，优化
            outputs,f = net(inputs)

            f1, f2 = torch.split(f, [12, 12], dim=0)
            f = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            l1 = criterion_cls(outputs, GT2)
            l2 = criterion_con(f, GT)
            loss = l1+l2
            loss.backward()
            optimizer.step()
            # 打印状态信息
            l_log = loss.detach()

            dist.all_reduce(l_log, op=dist.ReduceOp.SUM)
            ws = dist.get_world_size()
            l_log_num = l_log.item() / ws
            predict = torch.argmax(outputs, 1)
            calculate_confusion_matrix(GT, predict, confusion_matrix)
            dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
            if local_rank == 0:
                logger.result_log_train(confusion_matrix,l_log_num,epoch + epoch_init,g_step,i,writer)
                g_step += 1
        scheduler.step()
        # dist.barrier()
        if local_rank == 0:
            net.eval()
            y_true = []
            y_pred = []
            for i, data in enumerate(testloader, 0):
                # 获取输入Q
                inputs, GT_cpu, _ = data
                inputs = inputs.cuda(local_rank, non_blocking=True)
                GT = GT_cpu.cuda(local_rank, non_blocking=True)
                # print(f'd:{local_rank}')
                with torch.no_grad():
                    outputs,_ = net.module(inputs)
                    outputs = F.softmax(outputs,dim=1)
                    # predict = torch.argmax(outputs, 1)

                predict_np = outputs.cpu().numpy()
                GT_np = GT_cpu.numpy()

                y_true.append(GT_np)
                y_pred.append(predict_np)

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred, axis=0)
            # print(y_pred)
            fs = logger.result_log_test(y_true, y_pred, epoch + epoch_init,g_step,test_writer)
            save_dict = {'g_step': g_step, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),'scheduler':scheduler.state_dict()}
            best_bool = logger.save_checkpoint(save_dict,args.ckpt_path,fs,g_step)
            logger.draw_roc_auc(y_true, y_pred, g_step, os.path.join(args.ckpt_path, "last_ROC.jpg"))
            if best_bool:
                logger.draw_roc_auc(y_true, y_pred, g_step, os.path.join(args.ckpt_path, "best_ROC.jpg"))
        dist.barrier()

    print('Finished Training')
