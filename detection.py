# -*- coding:utf-8 -*-
# 用于模型的单张图像分类操作
import os
os.environ['GLOG_minloglevel'] = '2' # 将caffe的输出log信息不显示，必须放到import caffe前
import caffe # caffe 模块
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# 分类单张图像img
def detection(img, net, transformer, labels_file):
    im = caffe.io.load_image(img)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)

    start = time.clock()
    # 执行测试
    net.forward()
    end = time.clock()
    print('detection time: %f s' % (end - start))

    # 查看目标检测结果
    file = open(labels_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    loc = net.blobs['detection_out'].data[0][0]
    confidence_threshold = 0.5
    for l in range(len(loc)):
        if loc[l][2] >= confidence_threshold:
            xmin = int(loc[l][3] * im.shape[1])
            ymin = int(loc[l][4] * im.shape[0])
            xmax = int(loc[l][5] * im.shape[1])
            ymax = int(loc[l][6] * im.shape[0])
            img = np.zeros((512, 512, 3), np.uint8)  # 生成一个空彩色图像
            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (55 / 255.0, 255 / 255.0, 155 / 255.0), 2)

            # 确定分类类别
            class_name = labelmap.item[int(loc[l][1])].display_name
            # text_font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)
            cv2.putText(im, class_name, (xmin, ymax), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)

    # 显示结果
    plt.imshow(im, 'brg')
    plt.show()

#CPU或GPU模型转换
caffe.set_mode_cpu()
#caffe.set_device(0)
#caffe.set_mode_gpu()

caffe_root = '../../'
# 网络参数（权重）文件
caffemodel = caffe_root + 'models/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'
# 网络实施结构配置文件
deploy = caffe_root + 'models/SSD_300x300/deploy.prototxt'


img_root = caffe_root + 'data/VOCdevkit/VOC2007/JPEGImages/'
labels_file = caffe_root + 'data/VOC0712/labelmap_voc.prototxt'

# 网络实施分类
net = caffe.Net(deploy,  # 定义模型结构
                caffemodel,  # 包含了模型的训练权值
                caffe.TEST)  # 使用测试模式(不执行dropout)

# 加载ImageNet图像均值 (随着Caffe一起发布的)
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # 对所有像素值取平均以此获取BGR的均值像素值

# 图像预处理
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

# 处理图像
while 1:
    img_num = raw_input("Enter Img Number: ")
    if img_num == '': break
    img = img_root + '{:0>6}'.format(img_num) + '.jpg'
    detection(img,net,transformer,labels_file)