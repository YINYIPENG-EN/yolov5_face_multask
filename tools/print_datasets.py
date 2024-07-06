import torch
from data_iter.datasets import LoadImagesAndLabels
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='F:/mydatasets/wiki_crop_face/label_new/')
    parser.add_argument('--img_size', type=tuple, default=(256, 256))
    parser.add_argument('--flag_agu', type=bool, default=True)
    parser.add_argument('--fix_res', type=bool, default=False)
    ops = parser.parse_args()
    print(ops)
    dataset = LoadImagesAndLabels(ops=ops, img_size=ops.img_size, flag_agu=ops.flag_agu, fix_res=ops.fix_res, vis=False)
    print(dataset[1])
    print('len train datasets : %s' % (dataset.__len__()))