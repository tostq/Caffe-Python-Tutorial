# -*- coding:utf-8 -*-
# 生成Minst网络结构文件train.prototxt、test.prototxt及deploy.prototxt
from __future__ import print_function
import caffe
from caffe import layers as L
from caffe import params as P

# 此函数生成LeNet5的主体结构
def lenet5_body(net, from_layer):
    # 网络参数
    kwargs = {
        # param定义学习率，这里是指基础学习率step的情况，lt_mult乘以基础学习率为实际学习率，为0表示权重不更新，decay_mult同权重衰减相关
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='xavier'), # 权重初始化模式
        'bias_filler': dict(type='constant', value=0)} # 权重偏差初始化模式

    # 判断是否存在from_layer层
    assert from_layer in net.keys()
    # conv1
    net.conv1 = L.Convolution(net[from_layer], kernel_size=5, stride=1, num_output=20, pad=0, **kwargs)
    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv2 = L.Convolution(net.pool1, kernel_size=5, stride=1, num_output=50, pad=0, **kwargs)
    net.pool2 = L.Pooling(net.conv2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.ip1 = L.InnerProduct(net.pool2, num_output=500, **kwargs)
    net.relu1 = L.ReLU(net.ip1, in_place=True)
    net.ip2 = L.InnerProduct(net.relu1, name='ip2', num_output=10, **kwargs)

caffe_root = '../../'
# caffe_root = 'E:/Code/Github/windows_caffe/'
model_root = caffe_root + 'models/mnist/'

# 训练数据
train_data = caffe_root + "data/mnist/mnist_train_lmdb"
# 测试数据
test_data = caffe_root + "data/mnist/mnist_test_lmdb"

# 训练网络
train_net = caffe.NetSpec()  # 基础网络
# 带标签的数据输入层
train_net.data, train_net.label = L.Data(source=train_data,backend=P.Data.LMDB, batch_size=64,ntop=2,transform_param=dict(scale=0.00390625))
# 生成LeNet5的主体结构
lenet5_body(train_net, 'data')
# 生成误差损失层
train_net.loss = L.SoftmaxWithLoss(train_net.ip2, train_net.label)

# 测试网络
test_net = caffe.NetSpec()  # 基础网络
# 带标签的数据输入层
test_net.data, test_net.label = L.Data(source=test_data, batch_size=100, backend=P.Data.LMDB, ntop=2,transform_param=dict(scale=0.00390625))
# 生成LeNet5的主体结构
lenet5_body(test_net, 'data')
# 生成误差损失层
test_net.loss = L.SoftmaxWithLoss(test_net.ip2, test_net.label)
# 添加一个精确层
test_net.accuracy = L.Accuracy(test_net.ip2, test_net.label)

# 实施网络
deploy_net = caffe.NetSpec()  # 基础网络
# 带标签的数据输入层
deploy_net.data = L.Input(input_param=dict(shape=dict(dim=[64,1,28,28])))
# 生成LeNet5的主体结构
lenet5_body(deploy_net, 'data')
deploy_net.prob = L.Softmax(deploy_net.ip2)

# 保存训练文件
with open(model_root+'train.prototxt', 'w') as f:
    print('name: "LenNet5_train"', file=f)
    print(train_net.to_proto(), file=f)

with open(model_root+'test.prototxt', 'w') as f:
    print('name: "LenNet5_test"', file=f)
    print(test_net.to_proto(), file=f)

with open(model_root+'deploy.prototxt', 'w') as f:
    print('name: "LenNet5_test"', file=f)
    print(deploy_net.to_proto(), file=f)