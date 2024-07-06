import json
import os
from tqdm import tqdm
Json_label_path = 'F:/mydatasets/wiki_crop_face/label_new/'  # Json标签存储路径
Txt_label_path = 'F:/mydatasets/wiki_crop_face/label_txt/'   # Json转txt保存路径
os.makedirs(Txt_label_path, exist_ok=True)
txt_name = 'train_list.txt'
for f_ in tqdm(os.listdir(Json_label_path)):
    # 读取json文件
    f = open(Json_label_path + f_, encoding='utf-8')
    # 加载json
    dict = json.load(f)
    f.close()
    txt_name = f_.replace('json', 'txt')
    with open(Txt_label_path + txt_name, 'w') as txt_file:
        for item in dict:
            # 数据集存储转换格式为：
            '''
            以图片名为txt文件，内容为：
            loc,age,gender,landmarks
            '''
            # txt_file.write(f"#{f_}\n")
            if item in ['loc', 'age', 'gender', 'landmarks']:
                txt_file.write(str(dict[item]) + ';')
