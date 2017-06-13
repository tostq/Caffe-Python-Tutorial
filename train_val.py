# -*- coding:utf-8 -*-
# 训练及测试文件
# 训练网络
import caffe
import numpy as np
import matplotlib.pyplot as plt
import math

def crop_network(prune_proto, caffemodel, prune_caffemodel):
    # 截取已知网络的部分层
    #  caffemodel网络权重值并不要求其结构与proto相对应
    # 网络只会取train_proto中定义的结构中权重作为网络的初始权重值
    # 因此，当我们需要截取某些已训练网络的特定层作为新网络的某些层的权重初始值，只需要在其train_proto定义同名的层
    # 之后caffe将在caffemodel中找到与train_proto定义的同名结构，并将其权重作为应用权重初始值。
    # prune_deploy: 选择保留的网络结构层:prototxt
    # caffemodel: 已知网络的权重连接
    # prune_caffemodel：截断网络的权重连接文件
    net = caffe.Net(prune_proto, caffemodel, caffe.TEST)
    net.save(prune_caffemodel)

def train(solver_proto, caffemodel='', is_step=True, savefig=''):
    # 训练模型函数
    # solver_proto: 训练配置文件
    # caffemodel：预设权重值或者快照等，并不要求其结构与网络结构相对应，但只会取与训练网络结构相对应的权重值
    # is_step: True表示按步训练，False表示直接完成训练
    # savefig: 表示要保存的图像训练时损失变化图
    # 设置训练器：随机梯度下降算法
    solver = caffe.SGDSolver(solver_proto)
    if caffemodel!='':
        solver.net.copy_from(caffemodel)

    if is_step==False:
        # 直接完成训练
        solver.solve()
    else:
        # 迭代次数
        max_iter = 10000
        # 每隔100次收集一次数据
        display = 100

        # 每次测试进行100次解算，10000/100
        test_iter = 100
        # 每500次训练进行一次测试（100次解算），60000/64
        test_interval = 500

        # 初始化
        train_loss = np.zeros(int(math.ceil(max_iter * 1.0 / display)))
        test_loss = np.zeros(int(math.ceil(max_iter * 1.0 / test_interval)))
        test_acc = np.zeros(int(math.ceil(max_iter * 1.0 / test_interval)))

        # iteration 0，不计入
        solver.step(1)

        # 辅助变量
        _train_loss = 0
        _test_loss = 0
        _accuracy = 0

        # 分步训练
        for it in range(max_iter):
            # 进行一次解算
            solver.step(1)
            # 每迭代一次，训练batch_size张图片
            _train_loss += solver.net.blobs['loss'].data # 最后一层的损失值
            if it % display == 0:
                # 计算平均train loss
                train_loss[int(it / display)] = _train_loss / display
                _train_loss = 0

            # 测试
            if it % test_interval == 0:
                for test_it in range(test_iter):
                    # 进行一次测试
                    solver.test_nets[0].forward()
                    # 计算test loss
                    _test_loss += solver.test_nets[0].blobs['loss'].data
                    # 计算test accuracy
                    _accuracy += solver.test_nets[0].blobs['accuracy'].data
                    # 计算平均test loss
                test_loss[it / test_interval] = _test_loss / test_iter
                # 计算平均test accuracy
                test_acc[it / test_interval] = _accuracy / test_iter
                _test_loss = 0
                _accuracy = 0

                # 绘制train loss、test loss和accuracy曲线
        print '\nplot the train loss and test accuracy\n'
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # train loss -> 绿色
        ax1.plot(display * np.arange(len(train_loss)), train_loss, 'g')
        # test loss -> 黄色
        ax1.plot(test_interval * np.arange(len(test_loss)), test_loss, 'y')
        # test accuracy -> 红色
        ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')

        ax1.set_xlabel('iteration')
        ax1.set_ylabel('loss')
        ax2.set_ylabel('accuracy')

        if savefig!='':
            plt.savefig(savefig)
        plt.show()

#CPU或GPU模型转换
#caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()

caffe_root = '../../'
# caffe_root = 'E:/Code/Github/windows_caffe/'
model_root = caffe_root + 'models/mnist/'
solver_proto = model_root + 'solver.prototxt'
train(solver_proto, caffemodel='', is_step=True)
