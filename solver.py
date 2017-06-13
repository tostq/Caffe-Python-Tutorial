# -*- coding:utf-8 -*-
# 生成solver文件
from caffe.proto import caffe_pb2

def solver_file(model_root, model_name):
    s = caffe_pb2.SolverParameter() # 声明solver结构
    s.train_net = model_root+'train.prototxt' # 训练网络结构配置文件
    s.test_net.append(model_root+'test.prototxt') # 测试时网络结构配置文件，测试网络可有多个
    # 每训练迭代test_interval次进行一次测试。
    s.test_interval = 500
    # 每次测试时的批量数，测试里网络可有多个
    s.test_iter.append(100)
    # 最大训练迭代次数
    s.max_iter = 10000
    # 基础学习率
    s.base_lr = 0.01
    # 动量，记忆因子
    s.momentum = 0.9
    # 权重衰减值，遗忘因子
    s.weight_decay = 5e-4
    # 学习率变化策略。可选参数：fixed、step、exp、inv、multistep
    # fixed: 保持base_lr不变；
    # step: 学习率变化规律base_lr * gamma ^ (floor(iter / stepsize))，其中iter表示当前的迭代次数；
    # exp: 学习率变化规律base_lr * gamma ^ iter；
    # inv: 还需要设置一个power，学习率变化规律base_lr * (1 + gamma * iter) ^ (- power)；
    # multistep: 还需要设置一个stepvalue，这个参数和step相似，step是均匀等间隔变化，而multistep则是根据stepvalue值变化；
    #   stepvalue参数说明：
    #       poly: 学习率进行多项式误差，返回base_lr (1 - iter/max_iter) ^ (power)；
    #       sigmoid: 学习率进行sigmod衰减，返回base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))。
    s.lr_policy = 'inv'
    s.gamma = 0.0001
    s.power = 0.75

    s.display = 100 # 每迭代display次显示结果
    s.snapshot = 5000 # 保存临时模型的迭代数
    s.snapshot_prefix = model_root+model_name+'shapshot' # 模型前缀，就是训练好生成model的名字
    s.type = 'SGD' # 训练方法（各类梯度下降法），可选参数：SGD，AdaDelta，AdaGrad，Adam，Nesterov，RMSProp
    s.solver_mode = caffe_pb2.SolverParameter.GPU # 训练及测试模型，GPU或CPU

    solver_file=model_root+'solver.prototxt' # 要保存的solver文件名

    with open(solver_file, 'w') as f:
        f.write(str(s))

caffe_root = '../../'
model_name = 'LeNet5_Mnist_'
# caffe_root = 'E:/Code/Github/windows_caffe/'
model_root = caffe_root + 'models/mnist/'
solver_file(model_root, model_name)

