import os
from tqdm import tqdm
import random

def create_datasets_list():
    label_root_path = 'F:/mydatasets/wiki_crop_face/label_txt'
    output_file = 'datasets_list.txt'

    with open(output_file, 'w') as out_f:
        for f_ in tqdm(os.listdir(label_root_path)):
            with open(str(os.path.join(label_root_path, f_)), 'r') as f:
                line = f.readline().strip()
                lines = line.split(';')
                loc = lines[0]  # bbox
                age = lines[1]
                gender = lines[2]
                landmarks = lines[3]

                # 写入图像名称
                out_f.write(f"# {f_}\n")
                # 写入 loc;age;gender;landmarks
                out_f.write(f"{loc};{age};{gender};{landmarks}\n")

    print(f"数据已写入 {output_file}")


def read_output_list(output_file):
    with open(output_file, 'r') as f:
        lines = f.readlines()

    # 将数据按图像为单位进行分组
    data = []
    for i in range(0, len(lines), 2):
        data.append((lines[i].strip(), lines[i + 1].strip()))
    return data

def save_list(file_path, data_list, des=''):
    with open(file_path, 'w') as f:
        for name, info in tqdm(data_list, desc=des):
            f.write(f"{name}\n{info}\n")

def split_datasets():
    output_file = 'datasets_list.txt'
    train_file = 'train_list.txt'
    val_file = 'val_list.txt'
    # 读取 output_list.txt 的内容
    data = read_output_list(output_file)
    # 打乱数据顺序
    random.shuffle(data)

    # 划分训练集和验证集 (80% 训练集, 20% 验证集)
    train_split = int(0.8 * len(data))
    train_data = data[:train_split]
    val_data = data[train_split:]

    # 保存训练集和验证集列表到文件
    save_list(train_file, train_data, 'train')
    save_list(val_file, val_data, 'val')


if __name__ == '__main__':
    create_dataset = True
    split_dataset = True
    # 如果需要将所有数据集信息保存在一个txt中，运行下面函数即可
    if create_dataset:
        create_datasets_list()
    # 从datasets_list.txt中划分训练集和验证集，运行下面函数即可
    if split_dataset:
        split_datasets()