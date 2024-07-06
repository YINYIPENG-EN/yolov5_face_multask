#-*-coding:utf-8-*-
# function: data iter


import os
import argparse

import cv2
from IPython import embed

from tools.split_datasets import read_output_list
from torch.utils.data import Dataset
from data_iter.data_agu import *
import json


# 图像白化
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


# 图像亮度、对比度增强
def contrast_img(img, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img.shape
    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1-c, b)
    return dst


class LoadImagesAndLabels(Dataset):
    def __init__(self, ops, img_size=(224,224), flag_agu = False,fix_res = True,vis = False):

        print('img_size (height,width) : ',img_size[0],img_size[1])
        print("train_path : {}".format(ops.train_path))
        max_age = 0  # 用于记录数据集中最大年龄
        min_age = 65535.  # 用于记录数据集中最小年龄
        file_list = []  # 存放文件List
        landmarks_list = []  # 存放人脸关键点
        age_list = []  # 存放年龄list
        gender_list = []  # 存放性别
        idx = 0
        train_path_len = len(os.listdir(ops.train_path))
        for f_ in os.listdir(ops.train_path):
            f = open(ops.train_path + f_, encoding='utf-8')  # 读取 json文件
            dict = json.load(f)
            f.close()

            if dict["age"] > 100. or dict["age"] < 1.:
                continue
            idx += 1
            #-------------------------------------------------------------------
            img_path_ = (ops.train_path + f_).replace("label_new", "image").replace(".json", ".jpg")  # 读取对应的图像
            img = cv2.imread(img_path_)  # 读取图像
            file_list.append(img_path_)  # 存储图片路径

            # print("------> maker : {} ,age:{:.3f}, <{}/{}>".format(dict["maker"],dict["age"],idx,train_path_len))

            pts = [] # [[x,y],[x,y],[x,y],[x,y],...]  # 存放一个人的所有关键点
            for pt_ in dict["landmarks"]:
                x,y = pt_
                pts.append([x,y])
                # print("x,y : ",x,y)
                if vis:
                    cv2.circle(img, (int(x),int(y)), 2, (0,255,0),-1)
            # print(len(pts))
            landmarks_list.append(pts)  # [ [[x,y],[x,y],[x,y],[x,y],...]，.... ]  # 用于存储所有人的关键点
            if dict["gender"] == "male":
                gender_list.append(1)
            else:
                gender_list.append(0)
            age_list.append(dict["age"])
            if max_age < dict["age"]:
                max_age = dict["age"]
            if min_age > dict["age"]:
                min_age = dict["age"]

            if vis:
                # print(img.shape)
                x1,y1,x2,y2 = dict["loc"]
                # print("x1,y1,x2,y2",x1,y1,x2,y2)
                cv2.rectangle(img, (int(x1), int(y1)),(int(x2), int(y2)), (255, 255, 0),3)

                # draw_global_contour(img,dict["landmarks"])

                cv2.putText(img, 'age:{:.2f}'.format(dict["age"]), (2,20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)
                cv2.putText(img, 'gender:{}'.format(dict["gender"]), (2,40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)
                cv2.putText(img, 'age:{:.2f}'.format(dict["age"]), (2,20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 10, 25),1)
                cv2.putText(img, 'gender:{}'.format(dict["gender"]), (2,40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 10, 25),1)

                x1_ = x1 + random.randint(-5,5)
                y1_ = y1 + random.randint(-5,5)
                x2_ = x2 + random.randint(-5,5)
                y2_ = y2 + random.randint(-5,5)

                x1_ = max(0,x1_)
                x1_ = min(img.shape[1]-1,x1_)
                x2_ = max(0,x2_)
                x2_ = min(img.shape[1]-1,x2)
                y1_ = max(0,y1_)
                y1_ = min(img.shape[0]-1,y1_)
                y2_ = max(0,y2_)
                y2_ = min(img.shape[0]-1,y2_)

                cv2.namedWindow("result",0)
                cv2.imshow("result",img)
                cv2.waitKey(1)
        cv2.destroyAllWindows()

        print("max_age : {:.3f} ,min_age : {:.3f}".format(max_age,min_age))
        self.files = file_list
        self.landmarks = landmarks_list
        self.ages = age_list
        self.genders = gender_list
        self.img_size = img_size
        self.flag_agu = flag_agu
        self.fix_res = fix_res
        self.vis = vis

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]  # 读取图像绝对路径
        pts = self.landmarks[index]  #
        gender = self.genders[index]
        age = self.ages[index]

        img = cv2.imread(img_path)  # BGR
        if self.flag_agu == True:
            if random.random() > 0.35:
                left_eye = np.average(pts[60:68], axis=0)
                right_eye = np.average(pts[68:76], axis=0)

                angle_random = random.randint(-33,33)
                # 返回 crop 图 和 归一化 landmarks
                img_, landmarks_  = face_random_rotate(img, pts, angle_random, left_eye, right_eye,
                    fix_res = self.fix_res,img_size = self.img_size,vis = self.vis)
            else:
                x_max = -65535
                y_max = -65535
                x_min = 65535
                y_min = 65535

                for pt_ in pts:
                    x_,y_ = int(pt_[0]),int(pt_[1])
                    cv2.circle(img, (int(pt_[0]),int(pt_[1])), 2, (0,255,0),-1)
                    x_min = x_ if x_min>x_ else x_min
                    y_min = y_ if y_min>y_ else y_min
                    x_max = x_ if x_max<x_ else x_max
                    y_max = y_ if y_max<y_ else y_max

                #----------------------------------------
                face_w = x_max - x_min
                face_h = y_max - y_min
                x_min = int(x_min - random.randint(-6,int(face_w/10)))
                y_min = int(y_min - random.randint(-6,int(face_h/10)))
                x_max = int(x_max + random.randint(-6,int(face_w/10)))
                y_max = int(y_max + random.randint(-6,int(face_h/10)))

                x_min = np.clip(x_min,0,img.shape[1]-1)
                x_max = np.clip(x_max,0,img.shape[1]-1)
                y_min = np.clip(y_min,0,img.shape[0]-1)
                y_max = np.clip(y_max,0,img.shape[0]-1)

                face_w = x_max - x_min
                face_h = y_max - y_min

                face_crop = img[y_min:y_max,x_min:x_max,:]
                landmarks_ = []
                for pt_ in pts:
                    x_,y_ = int(pt_[0])-x_min,int(pt_[1])-y_min
                    if self.vis:
                        cv2.circle(face_crop, (x_,y_), 2, (0,255,0),-1)

                    landmarks_.append([float(x_)/float(face_w),float(y_)/float(face_h)])
                #----------------------------------------
                # cv2.rectangle(img, (x_min,y_min),(x_max,y_max), (255,0,0), 2)
                # if self.vis:
                #     cv2.namedWindow("face_crop",0)
                #     cv2.imshow("face_crop",face_crop)
                #     cv2.waitKey(1)
                # landmarks_ = pts
                # print(face_crop.shape)
                img_ = cv2.resize(face_crop, self.img_size, interpolation = random.randint(0,4))
        if self.flag_agu == True:
            if random.random() > 0.5:
                c = float(random.randint(80,120))/100.
                b = random.randint(-10,10)
                img_ = contrast_img(img_, c, b)
        if self.flag_agu == True:
            if random.random() > 0.7:
                # print('agu hue ')
                img_hsv=cv2.cvtColor(img_,cv2.COLOR_BGR2HSV)
                hue_x = random.randint(-10,10)
                # print(cc)
                img_hsv[:,:,0]=(img_hsv[:,:,0]+hue_x)
                img_hsv[:,:,0] =np.maximum(img_hsv[:,:,0],0)
                img_hsv[:,:,0] =np.minimum(img_hsv[:,:,0],180)#范围 0 ~180
                img_=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
        if self.flag_agu == True:
            if random.random() > 0.9:
                img_ = img_agu_channel_same(img_)
        # cv2.imwrite("./samples/{}.jpg".format(index),img_)
        if self.vis == True:
        # if True:

            cv2.putText(img_, 'age:{:.2f}'.format(age), (2,20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)
            if gender == 1.:
                cv2.putText(img_, 'gender:{}'.format("male"), (2,40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)
            else:
                cv2.putText(img_, 'gender:{}'.format("female"), (2,40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)

            cv2.namedWindow('crop',0)
            cv2.imshow('crop',img_)
            cv2.waitKey(1)
        img_ = img_.astype(np.float32)
        img_ = (img_-128.)/256.
        img_ = img_.transpose(2, 0, 1)
        landmarks_ = np.array(landmarks_).ravel()

        # gender = np.expand_dims(np.array(gender),axis=0)
        age = np.expand_dims(np.array(((age-50.)/100.)),axis=0) # 归一化年龄
        return img_, landmarks_, gender, age


class LoadImagesAndLables_txt(Dataset):
    def __init__(self, ops, img_size=(224, 224), flag_agu=False, fix_res=True, train=True):
        '''
        ops.train_path是train_list.txt路径
        '''
        print('img_size (height,width) : ', img_size[0], img_size[1])
        print("train_path : {}".format(ops.train_path))
        print("val_path : {}".format(ops.val_path))
        max_age = 0  # 用于记录数据集中最大年龄
        min_age = 65535.  # 用于记录数据集中最小年龄
        self.dataset_root = str(os.path.dirname(ops.train_path)) if train else str(os.path.dirname(ops.val_path))
        # file_list = []  # 存放文件List
        # landmarks_list = []  # 存放人脸关键点
        # age_list = []  # 存放年龄list
        # gender_list = []  # 存放性别
        # idx = 0
        # 读取数据集长度
        self.data = read_output_list(ops.train_path if train else ops.val_path)
        for i in range(len(self.data)):
            if float(self.data[i][1].split(';')[1]) > 100. or float(self.data[i][1].split(';')[1]) < 1.:
                continue  # 去除无效年龄
            if max_age < float(self.data[i][1].split(';')[1]):
                max_age = float(self.data[i][1].split(';')[1])
            if min_age > float(self.data[i][1].split(';')[1]):
                min_age = float(self.data[i][1].split(';')[1])
            # file_list.append(self.data[i][0].split('# '))
        print("max_age : {:.3f} ,min_age : {:.3f}".format(max_age, min_age))
        self.flag_agu = flag_agu  # 是否采用数据增强
        self.fix_res = fix_res
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # 读取图像绝对路径
        img_path = str(os.path.join(self.dataset_root, 'image', self.data[item][0].split('# ')[-1].replace('txt', 'jpg')))
        pts = eval(self.data[item][1].split(';')[3])  # str->list
        gender = 1 if self.data[item][1].split(';')[2] == 'male' else 0
        age = float(self.data[item][1].split(';')[1])
        img = cv2.imread(img_path)
        if self.flag_agu == True:
            if random.random() > 0.35:
                left_eye = np.average(pts[60:68], axis=0)
                right_eye = np.average(pts[68:76], axis=0)

                angle_random = random.randint(-33,33)
                # 返回 crop 图 和 归一化 landmarks
                img_, landmarks_ = face_random_rotate(img, pts, angle_random, left_eye, right_eye,
                    fix_res=self.fix_res, img_size=self.img_size, vis=False)
            else:
                x_max = -65535
                y_max = -65535
                x_min = 65535
                y_min = 65535

                for pt_ in pts:
                    x_, y_ = int(pt_[0]),int(pt_[1])
                    cv2.circle(img, (int(pt_[0]),int(pt_[1])), 2, (0,255,0),-1)
                    x_min = x_ if x_min>x_ else x_min
                    y_min = y_ if y_min>y_ else y_min
                    x_max = x_ if x_max<x_ else x_max
                    y_max = y_ if y_max<y_ else y_max

                #----------------------------------------
                face_w = x_max - x_min
                face_h = y_max - y_min
                x_min = int(x_min - random.randint(-6,int(face_w/10)))
                y_min = int(y_min - random.randint(-6,int(face_h/10)))
                x_max = int(x_max + random.randint(-6,int(face_w/10)))
                y_max = int(y_max + random.randint(-6,int(face_h/10)))

                x_min = np.clip(x_min,0,img.shape[1]-1)
                x_max = np.clip(x_max,0,img.shape[1]-1)
                y_min = np.clip(y_min,0,img.shape[0]-1)
                y_max = np.clip(y_max,0,img.shape[0]-1)

                face_w = x_max - x_min
                face_h = y_max - y_min

                face_crop = img[y_min:y_max,x_min:x_max,:]
                landmarks_ = []
                for pt_ in pts:
                    x_, y_ = int(pt_[0])-x_min,int(pt_[1])-y_min
                    # if self.vis:
                    #     cv2.circle(face_crop, (x_,y_), 2, (0,255,0),-1)

                    landmarks_.append([float(x_)/float(face_w),float(y_)/float(face_h)])
                img_ = cv2.resize(face_crop, self.img_size, interpolation = random.randint(0,4))

        if self.flag_agu == True:
            if random.random() > 0.5:
                c = float(random.randint(80,120))/100.
                b = random.randint(-10,10)
                img_ = contrast_img(img_, c, b)
        if self.flag_agu == True:
            if random.random() > 0.7:
                # print('agu hue ')
                img_hsv=cv2.cvtColor(img_,cv2.COLOR_BGR2HSV)
                hue_x = random.randint(-10,10)
                # print(cc)
                img_hsv[:,:,0]=(img_hsv[:,:,0]+hue_x)
                img_hsv[:,:,0] =np.maximum(img_hsv[:,:,0],0)
                img_hsv[:,:,0] =np.minimum(img_hsv[:,:,0],180)#范围 0 ~180
                img_=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
        if self.flag_agu == True:
            if random.random() > 0.9:
                img_ = img_agu_channel_same(img_)

        img_ = img_.astype(np.float32)
        img_ = (img_ - 128.) / 256.
        img_ = img_.transpose(2, 0, 1)
        landmarks_ = np.array(landmarks_).ravel()

        # gender = np.expand_dims(np.array(gender),axis=0)
        age = np.expand_dims(np.array(((age - 50.) / 100.)), axis=0)  # 归一化年龄
        return img_, landmarks_, gender, age



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--train_path', type=str, default='F:/mydatasets/wiki_crop_face/train_list.txt')
    parse.add_argument('--img_size', type=tuple, default=(256, 256))
    args = parse.parse_args()
    print(args)
    datasets = LoadImagesAndLables_txt(args, args.img_size)
