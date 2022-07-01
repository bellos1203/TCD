# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

ROOT_DATASET = '../data/'

def return_ucf101():
    filename_categories = 'ucf101/classInd.txt'
    root_data = ROOT_DATASET + 'ucf101/frames'
    filename_imglist_train = 'ucf101/splits/ucf101_rgb_train_split_1.txt'
    filename_imglist_val = 'ucf101/splits/ucf101_rgb_val_split_1.txt'
    prefix = 'img_{:05d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51():
    filename_categories = 51
    root_data = ROOT_DATASET + 'hmdb51/frames'
    filename_imglist_train = 'hmdb51/splits/hmdb51_rgb_train_split_1.txt'
    filename_imglist_val = 'hmdb51/splits/hmdb51_rgb_val_split_1.txt'
    prefix = 'img_{:05d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2():
    filename_categories = 'sthsth2/splits/category.txt'

    root_data = ROOT_DATASET + 'sthsth2/frames'
    filename_imglist_train = 'sthsth2/splits/train_videofolder.txt'
    filename_imglist_val = 'sthsth2/splits/val_videofolder.txt'
    prefix = 'img_{:05d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset):
    dict_single = {'ucf101': return_ucf101, 'hmdb51': return_hmdb51,
                'somethingv2': return_somethingv2}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset]()
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)

    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
