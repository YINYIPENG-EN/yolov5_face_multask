import cv2
import torch
import numpy as np
from utils.draw_utils import draw_face_landmarks
from face_multask_models.models.resnet import resnet34

model = resnet34(pretrained=False, landmarks_num=196, img_size=256)
ckpt = torch.load(r'../weights/face_multitask-resnet_34_imgsize-256-20210425.pth', map_location='cpu')
print(model.load_state_dict(ckpt, strict=False))
model = model.cuda()
img = cv2.imread('../img/2.jpg')
algo_image = img.copy()
img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
img = img.astype(np.float32)
img = (img - 128) / 256.
img_tensor = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).cuda()  # hwc->bchw
landmarks, gender, age = model(img_tensor)
# print(landmarks.size(), gender.size(), age.size())
landmarks = landmarks.cpu().detach().numpy()
gender = np.array(gender.cpu().detach().numpy())
age = (age.cpu().detach().numpy() * 100. + 50.)
# print("性别："+gender + " 年龄： " + int(age[0]))
num_target = gender.shape[0]
# 人脸宽度和高度
face_h = algo_image.shape[0]
face_w = algo_image.shape[1]
for i in range(num_target):
    # 寻找性别最大索引
    gender_max_index = np.argmax(gender[i])
    score_gender = gender[i][gender_max_index]
    dict_landmarks, eyes_center, face_area = draw_face_landmarks(algo_image, landmarks[i], face_w, face_h, 0, 0, True)
cv2.imshow("face", algo_image)
cv2.waitKey(0)








