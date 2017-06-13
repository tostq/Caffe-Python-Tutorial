# -*- coding:utf-8 -*-
# 将图像数据生成lmdb数据集
# 1. 生成分类图像数据集
# 2. 生成目标检测图像数据集
import os
import sys
import numpy as np
import random
from caffe.proto import caffe_pb2
from xml.dom.minidom import parse

# 生成分类标签文件
def labelmap(labelmap_file, label_info):
    labelmap = caffe_pb2.LabelMap()
    for i in range(len(label_info)):
        labelmapitem = caffe_pb2.LabelMapItem()
        labelmapitem.name = label_info[i]['name']
        labelmapitem.label = label_info[i]['label']
        labelmapitem.display_name = label_info[i]['display_name']
        labelmap.item.add().MergeFrom(labelmapitem)
    with open(labelmap_file, 'w') as f:
        f.write(str(labelmap))

def rename_img(Img_dir):
    # 重新命名Img,这里假设图像名称表示为000011.jpg、003456.jpg、000000.jpg格式，最高6位，前补0
    # 列出图像，并将图像改为序号名称
    listfile=os.listdir(Img_dir) # 提取图像名称列表
    total_num = 0
    for line in listfile:  #把目录下的文件都赋值给line这个参数
        if line[-4:] == '.jpg':
            newname = '{:0>6}'.format(total_num) +'.jpg'
            os.rename(os.path.join(Img_dir, line), os.path.join(Img_dir, newname))
            total_num+=1         #统计所有图像

def get_img_size():
    pass

def create_annoset(anno_args):
    if anno_args.anno_type == "detection":
        cmd = "E:\Code\windows-ssd/Build/x64/Release/convert_annoset.exe" \
              " --anno_type={}" \
              " --label_type={}" \
              " --label_map_file={}" \
              " --check_label={}" \
              " --min_dim={}" \
              " --max_dim={}" \
              " --resize_height={}" \
              " --resize_width={}" \
              " --backend={}" \
              " --shuffle={}" \
              " --check_size={}" \
              " --encode_type={}" \
              " --encoded={}" \
              " --gray={}" \
              " {} {} {}" \
            .format(anno_args.anno_type, anno_args.label_type, anno_args.label_map_file, anno_args.check_label,
                    anno_args.min_dim, anno_args.max_dim, anno_args.resize_height, anno_args.resize_width, anno_args.backend, anno_args.shuffle,
                    anno_args.check_size, anno_args.encode_type, anno_args.encoded, anno_args.gray, anno_args.root_dir, anno_args.list_file, anno_args.out_dir)
    elif anno_args.anno_type == "classification":
        cmd = "E:\Code\windows-ssd/Build/x64/Release/convert_annoset.exe" \
              " --anno_type={}" \
              " --min_dim={}" \
              " --max_dim={}" \
              " --resize_height={}" \
              " --resize_width={}" \
              " --backend={}" \
              " --shuffle={}" \
              " --check_size={}" \
              " --encode_type={}" \
              " --encoded={}" \
              " --gray={}" \
              " {} {} {}" \
            .format(anno_args.anno_type, anno_args.min_dim, anno_args.max_dim, anno_args.resize_height,
                    anno_args.resize_width, anno_args.backend, anno_args.shuffle, anno_args.check_size, anno_args.encode_type, anno_args.encoded,
                    anno_args.gray, anno_args.root_dir, anno_args.list_file, anno_args.out_dir)
    print cmd
    os.system(cmd)

def detection_list(Img_dir, Ano_dir, Data_dir, test_num):
    # 造成目标检测图像数据库
    # Img_dir表示图像文件夹
    # Ano_dir表示图像标记文件夹，用labelImg生成
    # Data_dir生成的数据库文件地址
    # test_num测试图像的数目
    # 列出图像
    listfile=os.listdir(Img_dir) # 提取图像名称列表

    # 列出图像，并将图像改为序号名称
    total_num = 0
    for line in listfile:  #把目录下的文件都赋值给line这个参数
        if line[-4:] == '.jpg':
            total_num+=1         #统计所有图像

    trainval_num = total_num-test_num # 训练图像数目

    # 生成训练图像及测试图像列表
    test_list_file=open(Data_dir+'/test.txt','w')
    train_list_file=open(Data_dir+'/trainval.txt','w')

    test_list = np.random.randint(0,total_num-1, size=test_num)

    train_list = range(total_num)
    for n in range(test_num):
        train_list.remove(test_list[n])
    random.shuffle(train_list)

    # 测试图像排序，而训练图像不用排序
    test_list = np.sort(test_list)
    # train_list = np.sort(train_list)

    for n in range(trainval_num):
        train_list_file.write(Img_dir + '{:0>6}'.format(train_list[n]) +'.jpg '+ Ano_dir + '{:0>6}'.format(train_list[n]) +'.xml\n')

    for n in range(test_num):
        test_list_file.write(Img_dir + '{:0>6}'.format(test_list[n]) +'.jpg '+ Ano_dir + '{:0>6}'.format(test_list[n]) +'.xml\n')


caffe_root = 'E:/Code/Github/windows_caffe/'
data_root = caffe_root + 'data/mnist/'
Img_dir = data_root + 'JPEGImages/'
Ano_dir = data_root + 'Annotations/'
anno_type = "detection"
test_num = 100

# 第一步，预处理图像，重命名图像名，生成各图像标记信息
# rename_img(Img_dir)
# 然后通过labelImg(可以通过pip install labelImg安装，出现错误可以删除PyQt4的描述）来生成图像的标记

# 第二步，生成分类标签文件
# 编辑label信息
label_info = [
    dict(name='none', label=0, display_name='background'),  # 背景
    dict(name="cat",label=1, display_name='cat'),  # 背景
    dict(name="dog",label=2, display_name='dog'),  # 背景
]
labelmap(data_root+'labelmap_voc.prototxt', label_info)

# 第三步，生成图像及标记的列表文件
if anno_type == "detection":
    detection_list(Img_dir, Ano_dir, data_root, test_num)
else:
    # 分类，生成
    pass

# 第四步，生成lmdb文件
# 初始化信息
anno_args = {}
anno_args['anno_type'] = anno_type
# 仅用于目标检测，lable文件的类型：{xml, json, txt}
anno_args['label_type'] = "xml"
# 仅用于目标检测，label文件地址
anno_args['label_map_file'] = data_root+"labelmap_voc.prototxt"
# 是否检测所有数据有相同的大小.默认False
anno_args['check_size'] = False
# 检测label是否相同的名称，默认False
anno_args['check_label'] = False
# 为0表示图像不用重新调整尺寸
anno_args['min_dim'] = 0
anno_args['max_dim'] = 0
anno_args['resize_height'] = 0
anno_args['resize_width'] = 0
anno_args['backend'] = "lmdb"  # 数据集格式（lmdb, leveldb）
anno_args['shuffle'] = False  # 是否随机打乱图像及对应标签
anno_args['encode_type'] = ""  # 图像编码格式('png','jpg',...)
anno_args['encoded'] = False  # 是否编码，默认False
anno_args['gray'] = False  # 是否视为灰度图，默认False
anno_args['root_dir'] = data_root  # 存放图像文件夹及标签文件夹的根目录
anno_args['list_file'] = data_root + ''  # listfile文件地址
anno_args['out_dir'] = data_root  # 最终lmdb的存在地址

# 生成训练数据集train_lmdb
anno_args['list_file'] = data_root + 'trainval.txt'
create_annoset(anno_args)

# 生成测试数据集train_lmdb
anno_args['list_file'] = data_root + 'test.txt'
create_annoset(anno_args)





