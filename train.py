#-*-coding:utf-8-*-
## function: train


import argparse
import  sys

import numpy as np
from torch.utils.data import DataLoader
from utils.model_utils import *
from utils.common_utils import *
from data_iter.datasets import *
from face_multask_models.models.resnet import resnet50, resnet34, resnet18
from loss.loss import *
import time
import json
import random


def trainer(ops,f_log):
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

        if ops.log_flag:
            sys.stdout = f_log

        set_seed(ops.seed)
        #---------------------------------------------------------------- 构建模型
        print('use model : %s'%(ops.model))

        if ops.model == 'resnet_50':
            model_ = resnet50(pretrained=ops.pretrained, landmarks_num=ops.num_classes, img_size=ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_34':
            model_ = resnet34(pretrained=ops.pretrained, landmarks_num=ops.num_classes, img_size=ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_18':
            model_ = resnet18(pretrained=ops.pretrained, landmarks_num=ops.num_classes, img_size=ops.img_size[0],dropout_factor=ops.dropout)

        use_cuda = torch.cuda.is_available()

        device = torch.device(f"cuda:{ops.GPUS}" if use_cuda else "cpu")
        model_ = model_.to(device)

        # print(model_)# 打印模型结构
        # Dataset
        # dataset = LoadImagesAndLabels(ops=ops, img_size=ops.img_size, flag_agu=ops.flag_agu, fix_res=ops.fix_res, vis=False)  # 加载json格式
        dataset = LoadImagesAndLables_txt(ops=ops, img_size=ops.img_size, flag_agu=ops.flag_agu, fix_res=ops.fix_res, train=True)  # 加载txt格式
        val_datasets = LoadImagesAndLables_txt(ops, img_size=ops.img_size, flag_agu=False, fix_res=ops.fix_res, train=False)
        # print('len train datasets : %s'%(dataset.__len__()))
        # Dataloader
        dataloader = DataLoader(dataset,
                                batch_size=ops.batch_size,
                                num_workers=ops.num_workers,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)
        val_dataloader = DataLoader(val_datasets,batch_size=ops.batch_size,
                                    num_workers=ops.num_workers,
                                    shuffle=False,
                                    pin_memory=True,
                                    drop_last=True
                                    )
        print("完成数据集加载")
        # 优化器设计
        optimizer_Adam = torch.optim.Adam(model_.parameters(), lr=ops.init_lr, betas=(0.9, 0.99),weight_decay=ops.weight_decay)
        # optimizer_SGD = optim.SGD(model_.parameters(), lr=ops.init_lr, momentum=ops.momentum, weight_decay=ops.weight_decay)# 优化器初始化
        optimizer = optimizer_Adam
        # 加载 finetune 模型
        if os.access(ops.fintune_model,os.F_OK):# checkpoint
            chkpt = torch.load(ops.fintune_model, map_location='cpu')
            model_.load_state_dict(chkpt)
            print('load fintune model : {}'.format(ops.fintune_model))

        print('/**********************************************/')
        # 损失函数
        if ops.loss_define != 'wing_loss':
            criterion = nn.MSELoss(reduce=True, reduction='mean')

        criterion_gender = nn.CrossEntropyLoss()  #CrossEntropyLoss() 是 softmax 和 负对数损失的结合

        step = 0
        val_step = 0
        idx = 0

        # 变量初始化
        best_loss = np.inf
        loss_mean = 0. # 损失均值
        loss_idx = 0. # 损失计算计数器
        flag_change_lr_cnt = 0  # 学习率更新计数器
        init_lr = ops.init_lr  # 学习率

        val_best_loss = np.inf
        val_loss_mean = 0.
        val_loss_idx = 0.

        epochs_loss_dict = {}

        for epoch in range(0, ops.epochs):
            if ops.log_flag:
                sys.stdout = f_log
            print('\nepoch %d ------>>>'%epoch)

            # 学习率更新策略
            if loss_mean!=0.:
                if best_loss > (loss_mean/loss_idx):
                    flag_change_lr_cnt = 0
                    best_loss = (loss_mean/loss_idx)
                else:
                    flag_change_lr_cnt += 1

                    if flag_change_lr_cnt > 60:
                        init_lr = init_lr*ops.lr_decay
                        set_learning_rate(optimizer, init_lr)
                        flag_change_lr_cnt = 0

            loss_mean = 0. # 损失均值
            loss_idx = 0. # 损失计算计数器
            model_.train()
            for i, (imgs_,pts_,gender_,age_) in enumerate(dataloader):
                # print('imgs_, pts_,gender_,age_ : ',imgs_.size(), pts_.size(),gender_.size(),age_.size())
                # continue
                if use_cuda:
                    imgs_ = imgs_.cuda()  # pytorch 的 数据输入格式 ： (batch, channel, height, width)
                    pts_ = pts_.cuda()
                    gender_ = gender_.cuda()
                    age_ = age_.cuda()
                output_landmarks, output_gender, output_age = model_(imgs_.float())
                # print("output_gender,output_age : ",output_gender.size(),output_age.size())
                if ops.loss_define == 'wing_loss':
                    loss_pts = got_total_wing_loss(output_landmarks, pts_.float())   # 关键点
                    loss_age = got_total_wing_loss(output_age, age_.float())  # 年龄
                    # loss_eye_center = got_total_wing_loss(output_landmarks[:, 192:196], pts_[:, 192:196].float())*0.3
                    # loss_eye = got_total_wing_loss(output_landmarks[:, 120:152], pts_[:, 120:152].float())*0.3

                else:
                    loss_pts = criterion(output_landmarks, pts_.float())
                    loss_age = criterion(output_age, age_.float())

                loss_gender = criterion_gender(output_gender, gender_)  # 性别损失函数

                loss = loss_pts + 0.3*loss_age + 0.25*loss_gender
#                 loss = loss_pts

                loss_mean += loss.item()
                loss_idx += 1.
                if i%10 == 0:
                    loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print('  %s - %s - epoch [%s/%s] (%s/%s):'%(loc_time,ops.model,epoch,ops.epochs,i,int(dataset.__len__()/ops.batch_size)),\
                    'Mean Loss : %.6f - Loss: %.6f'%(loss_mean/loss_idx,loss.item()),\
                    " loss_pts:{:.4f},loss_age:{:.4f},loss_gender:{:.4f}".format(loss_pts.item(),loss_age.item(),loss_gender.item()),\
                    ' lr : %.5f'%init_lr, ' bs:', ops.batch_size,\
                    ' img_size: %s x %s'%(ops.img_size[0],ops.img_size[1]),' best_loss:%.6f'%best_loss)
                # 计算梯度
                loss.backward()
                # 优化器对模型参数更新
                optimizer.step()
                # 优化器梯度清零
                optimizer.zero_grad()
                step += 1

            # eval
            print("****************************val********************************\n")
            model_.eval()
            for i, (imgs_, pts_, gender_, age_) in enumerate(val_dataloader):
                # print('imgs_, pts_,gender_,age_ : ',imgs_.size(), pts_.size(),gender_.size(),age_.size())
                # continue
                if use_cuda:
                    imgs_ = imgs_.cuda()  # pytorch 的 数据输入格式 ： (batch, channel, height, width)
                    pts_ = pts_.cuda()
                    gender_ = gender_.cuda()
                    age_ = age_.cuda()
                output_landmarks, output_gender, output_age = model_(imgs_.float())
                # print("output_gender,output_age : ",output_gender.size(),output_age.size())
                if ops.loss_define == 'wing_loss':
                    val_loss_pts = got_total_wing_loss(output_landmarks, pts_.float())  # 关键点
                    val_loss_age = got_total_wing_loss(output_age, age_.float())  # 年龄
                    # loss_eye_center = got_total_wing_loss(output_landmarks[:, 192:196], pts_[:, 192:196].float())*0.3
                    # loss_eye = got_total_wing_loss(output_landmarks[:, 120:152], pts_[:, 120:152].float())*0.3

                else:
                    val_loss_pts = criterion(output_landmarks, pts_.float())
                    val_loss_age = criterion(output_age, age_.float())

                val_loss_gender = criterion_gender(output_gender, gender_)  # 性别损失函数

                val_loss = val_loss_pts + 0.3 * val_loss_age + 0.25 * val_loss_gender
                #                 loss = loss_pts

                val_loss_mean += val_loss.item()
                val_loss_idx += 1.
                if i % 10 == 0:
                    loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print('  %s - %s - epoch [%s/%s] (%s/%s):' % (
                    loc_time, ops.model, epoch, ops.epochs, i, int(dataset.__len__() / ops.batch_size)),
                          'Val Mean Loss : %.6f - Val Loss: %.6f' % (val_loss_mean / val_loss_idx, val_loss.item()),
                          " val loss_pts:{:.4f}, val loss_age:{:.4f}, val loss_gender:{:.4f}".format(val_loss_pts.item(), val_loss_age.item(),
                                                                                       val_loss_gender.item()),
                          ' lr : %.5f' % init_lr, ' bs:', ops.batch_size,
                          ' img_size: %s x %s' % (ops.img_size[0], ops.img_size[1]), ' val best_loss:%.6f' % val_best_loss)

                # 优化器梯度清零
                optimizer.zero_grad()
                step += 1
            if epoch % 3 == 0:  # 保存权重的周期
                torch.save(model_.state_dict(), ops.model_exp + '{}_epoch-{}.pth'.format(ops.model, epoch))
            set_seed(random.randint(0,65535))
    except Exception as e:
        print('Exception : ', e) # 打印异常
        print('Exception  file : ', e.__traceback__.tb_frame.f_globals['__file__'])# 发生异常所在的文件
        print('Exception  line : ', e.__traceback__.tb_lineno)  # 发生异常所在的行数


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Project Multi Task Train')
    parser.add_argument('--seed', type=int, default=15678, help='seed')  # 设置随机种子
    parser.add_argument('--model_exp', type=str, default='./model_exp', help='model_exp')  # 模型输出文件夹
    parser.add_argument('--model', type=str, default='resnet_34', help='model : resnet_50')  # 模型类型
    parser.add_argument('--num_classes', type=int, default=196, help='num_classes')  # landmarks 个数*2 98*2
    parser.add_argument('--GPUS', type=str, default='0', help='GPUS')  # GPU选择

    parser.add_argument('--train_path', type=str, default='F:/mydatasets/wiki_crop_face/train_list.txt', help='train_path')  # 训练集标注信息
    parser.add_argument('--val_path', type=str, default='F:/mydatasets/wiki_crop_face/val_list.txt', help='val_path')
    parser.add_argument('--pretrained', type=bool, default=False, help='imageNet_Pretrain')  # 初始化学习率
    parser.add_argument('--fintune_model', type=str, default='./weights/face_multitask-resnet_34_imgsize-256-20210425.pth', help='fintune_model')  # fintune model
    parser.add_argument('--loss_define', type=str, default='wing_loss', help='define_loss')  # 损失函数定义
    parser.add_argument('--init_lr', type=float, default=2e-4, help='init_learningRate')  # 初始化学习率
    parser.add_argument('--lr_decay', type=float, default=0.5, help='learningRate_decay')  # 学习率权重衰减率
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')  # 优化器正则损失权重
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')  # 优化器动量
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')  # 训练每批次图像数量
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')  # dropout
    parser.add_argument('--epochs', type=int, default=2000, help='epochs')  # 训练周期
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')  # 训练数据生成器线程数
    parser.add_argument('--img_size', type=tuple, default=(256, 256), help='img_size')  # 输入模型图片尺寸
    parser.add_argument('--flag_agu', type=bool, default=True, help='data_augmentation')  # 训练数据生成器是否进行数据扩增
    parser.add_argument('--fix_res', type=bool, default=False, help='fix_resolution')  # 输入模型样本图片是否保证图像分辨率的长宽比
    parser.add_argument('--clear_model_exp', type=bool, default=False, help='clear_model_exp')  # 模型输出文件夹是否进行清除
    parser.add_argument('--log_flag', type=bool, default=False, help='log flag')  # 是否保存训练 log

    # --------------------------------------------------------------------------
    args = parser.parse_args()  # 解析添加参数
    # --------------------------------------------------------------------------
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)
    loc_time = time.localtime()
    args.model_exp = args.model_exp + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)+'/'
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)

    f_log = None
    if args.log_flag:
        f_log = open(args.model_exp+'/train_{}.log'.format(time.strftime("%Y-%m-%d_%H-%M-%S",loc_time)), 'a+')
        sys.stdout = f_log

    print('---------------------------------- log : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", loc_time)))
    print('\n/******************* {} ******************/\n'.format(parser.description))

    unparsed = vars(args)  # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key, unparsed[key]))

    unparsed['time'] = time.strftime("%Y-%m-%d %H:%M:%S", loc_time)

    fs = open(args.model_exp+'train_ops.json', "w", encoding='utf-8')
    json.dump(unparsed, fs, ensure_ascii=False, indent=1)
    fs.close()

    trainer(ops=args, f_log= f_log)# 模型训练

    if args.log_flag:
        sys.stdout = f_log
    print('well done : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
