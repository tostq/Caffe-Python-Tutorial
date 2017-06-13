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
def classification(img, net, transformer, synset_words):
    im = caffe.io.load_image(img)
    # 导入输入图像
    net.blobs['data'].data[...] = transformer.preprocess('data', im)

    start = time.clock()
    # 执行测试
    net.forward()
    end = time.clock()
    print('classification time: %f s' % (end - start))

    # 查看目标检测结果
    labels = np.loadtxt(synset_words, str, delimiter='\t')

    category = net.blobs['prob'].data[0].argmax()

    class_str = labels[int(category)].split(',')
    class_name = class_str[0]
    # text_font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)
    cv2.putText(im, class_name, (0, im.shape[0]), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)

    # 显示结果
    plt.imshow(im, 'brg')
    plt.show()

#CPU或GPU模型转换
caffe.set_mode_cpu()
#caffe.set_device(0)
#caffe.set_mode_gpu()

caffe_root = '../../'
# 网络参数（权重）文件
caffemodel = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
# 网络实施结构配置文件
deploy = caffe_root + 'models/bvlc_alexnet/deploy.prototxt'


img_root = caffe_root + 'data/VOCdevkit/VOC2007/JPEGImages/'
synset_words = caffe_root + 'data/ilsvrc12/synset_words.txt'

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
    classification(img,net,transformer,synset_words)

