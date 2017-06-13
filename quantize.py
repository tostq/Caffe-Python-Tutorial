# -*- coding:utf-8 -*-
# 通过Kmeans聚类的方法来量化权重
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.vq as scv
import pickle
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import time

# 获得各层的量化码表
def kmeans_net(net, layers, num_c=16, initials=None):
    # net: 网络
    # layers: 需要量化的层
    # num_c: 各层的量化级别
    # initials: 初始聚类中心
    codebook = {} # 量化码表
    if type(num_c) == type(1):
        num_c = [num_c] * len(layers)
    else:
        assert len(num_c) == len(layers)

    # 对各层进行聚类分析
    print "==============Perform K-means============="
    for idx, layer in enumerate(layers):
        print "Eval layer:", layer
        W = net.params[layer][0].data.flatten()
        W = W[np.where(W != 0)] # 筛选不为0的权重
        # 默认情况下，聚类中心为线性分布中心
        if initials is None:  # Default: uniform sample
            min_W = np.min(W)
            max_W = np.max(W)
            initial_uni = np.linspace(min_W, max_W, num_c[idx] - 1)
            codebook[layer], _ = scv.kmeans(W, initial_uni)
        elif type(initials) == type(np.array([])):
            codebook[layer], _ = scv.kmeans(W, initials)
        elif initials == 'random':
            codebook[layer], _ = scv.kmeans(W, num_c[idx] - 1)
        else:
            raise Exception

        # 将0权重值附上
        codebook[layer] = np.append(0.0, codebook[layer])
        print "codebook size:", len(codebook[layer])

    return codebook

# 随机量化权重值
def stochasitc_quantize2(W, codebook):
    # mask插入新维度：(W.shape,1)
    mask = W[:, np.newaxis] - codebook

    mask_neg = mask
    mask_neg[mask_neg > 0.0] -= 99999.0
    max_neg = np.max(mask_neg, axis=1)
    max_code = np.argmax(mask_neg, axis=1)

    mask_pos = mask
    mask_pos += 99999.0
    min_code = np.argmin(mask_pos, axis=1)
    min_pos = np.min(mask_pos, axis=1)

    rd = np.random.uniform(low=0.0, high=1.0, size=(len(W)))
    thresh = min_pos.astype(np.float32) / (min_pos - max_neg)

    max_idx = thresh < rd
    min_idx = thresh >= rd

    codes = np.zeros(W.shape)
    codes[max_idx] += min_code[max_idx]
    codes[min_idx] += max_code[min_idx]

    return codes.astype(np.int)

# 得到网络的量化权重值
def quantize_net(net, codebook):
    layers = codebook.keys()
    codes_W = {}
    print "================Perform quantization=============="
    for layer in layers:
        print "Quantize layer:", layer
        W = net.params[layer][0].data
        codes, _ = scv.vq(W.flatten(), codebook[layer]) # 根据码表得到量化权重值
        # codes = stochasitc_quantize2(W.flatten(), codebook[layer]) # 采用随机量化的方式
        codes = np.reshape(codes, W.shape)
        codes_W[layer] = np.array(codes, dtype=np.uint32)
        # 将量化后的权重保存到网络中
        W_q = np.reshape(codebook[layer][codes], W.shape)
        np.copyto(net.params[layer][0].data, W_q)

    return codes_W


def quantize_net_with_dict(net, layers, codebook, use_stochastic=False, timing=False):
    start_time = time.time()
    codeDict = {} # 记录各个量化中心所处的位置
    maskCode = {} # 各层量化结果
    for layer in layers:
        print "Quantize layer:", layer
        W = net.params[layer][0].data
        if use_stochastic:
            codes = stochasitc_quantize2(W.flatten(), codebook[layer])
        else:
            codes, _ = scv.vq(W.flatten(), codebook[layer])
        W_q = np.reshape(codebook[layer][codes], W.shape)
        net.params[layer][0].data[...] = W_q

        maskCode[layer] = np.reshape(codes, W.shape)
        codeBookSize = len(codebook[layer])
        a = maskCode[layer].flatten()
        b = xrange(len(a))

        codeDict[layer] = {}
        for i in xrange(len(a)):
            codeDict[layer].setdefault(a[i], []).append(b[i])

    if timing:
        print "Update codebook time:%f" % (time.time() - start_time)

    return codeDict, maskCode

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(step_cache={}, step_cache2={}, count=0)
def update_codebook_net(net, codebook, codeDict, maskCode, args, update_layers=None, snapshot=None):

    start_time = time.time()
    extra_lr = args['lr'] # 基础学习速率
    decay_rate = args['decay_rate'] # 衰减速率
    momentum = args['momentum'] # 遗忘因子
    update_method = args['update'] # 更新方法
    smooth_eps = 0

    normalize_flag = args['normalize_flag'] # 是否进行归一化


    if update_method == 'rmsprop':
        extra_lr /= 100

    # 对码表与量化结果的初始化
    if update_codebook_net.count == 0:
        step_cache2 = update_codebook_net.step_cache2
        step_cache = update_codebook_net.step_cache
        if update_method == 'adadelta':
            for layer in update_layers:
                step_cache2[layer] = {}
                for code in xrange(1, len(codebook[layer])):
                    step_cache2[layer][code] = 0.0
            smooth_eps = 1e-8

        for layer in update_layers:
            step_cache[layer] = {}
            for code in xrange(1, len(codebook[layer])):
                step_cache[layer][code] = 0.0

        update_codebook_net.count = 1

    else:
        # 读入上次运算的结果
        step_cache2 = update_codebook_net.step_cache2
        step_cache = update_codebook_net.step_cache
        update_codebook_net.count += 1

    # 所有层名
    total_layers = net.params.keys()
    if update_layers is None: # 所有层都需要进行更新
        update_layers = total_layers

    # 权重码表的更新
    for layer in total_layers:
        if layer in update_layers:
            diff = net.params[layer][0].diff.flatten() # 误差梯度
            codeBookSize = len(codebook[layer])
            dx = np.zeros((codeBookSize)) # 编码表的误差更新
            for code in xrange(1, codeBookSize):
                indexes = codeDict[layer][code] # codeDict保存属于某编码的权重的序号
                #diff_ave = np.sum(diff[indexes]) / len(indexes)
                diff_ave = np.sum(diff[indexes]) # 统计该编码所有的误差更新和

                # 针对于不同方法进行更新
                if update_method == 'sgd':
                    dx[code] = -extra_lr * diff_ave
                elif update_method == 'momentum':
                    if code in step_cache[layer]:
                        dx[code] = momentum * step_cache[layer][code] - (1 - momentum) * extra_lr * diff_ave
                        step_cache[layer][code] = dx
                elif update_method == 'rmsprop':
                    if code in step_cache[layer]:
                        step_cache[layer][code] = decay_rate * step_cache[layer][code] + (1.0 - decay_rate) * diff_ave ** 2
                        dx[code] = -(extra_lr * diff_ave) / np.sqrt(step_cache[layer][code] + 1e-6)
                elif update_method == 'adadelta':
                    if code in step_cache[layer]:
                        step_cache[layer][code] = step_cache[layer][code] * decay_rate + (1.0 - decay_rate) * diff_ave ** 2
                        dx[code] = -np.sqrt((step_cache2[layer][code] + smooth_eps) / (step_cache[layer][code] + smooth_eps)) * diff_ave
                        step_cache2[layer][code] = step_cache2[layer][code] * decay_rate + (1.0 - decay_rate) * (dx[code] ** 2)

            # 是否需要进行归一化更新参数
            if normalize_flag:
                codebook[layer] += extra_lr * np.sqrt(np.mean(codebook[layer] ** 2)) / np.sqrt(np.mean(dx ** 2)) * dx
            else:
                codebook[layer] += dx
        else:
            pass

        # maskCode保存编码结果
        W2 = codebook[layer][maskCode[layer]]
        net.params[layer][0].data[...] = W2 # 量化后权重值

    print "Update codebook time:%f" % (time.time() - start_time)

# 保存量化结果
def store_all(net, codebook, dir_t, idx=0):
    net.save(dir_t + 'caffemodel%d' % idx)
    # 量化网络及码表
    pickle.dump(codebook, open(dir_t + 'codebook%d' % idx, 'w'))

# 恢复权重值
def recover_all(net, dir_t, idx=0):
    layers = net.params.keys()
    net.copy_from(dir_t + 'caffemodel%d' % idx)
    codebook = pickle.load(open(dir_t + 'codebook%d' % idx))
    maskCode = {}
    codeDict = {}
    for layer in layers:
        W = net.params[layer][0].data
        # 码表结果
        codes, _ = scv.vq(W.flatten(), codebook[layer])
        # 编码结果重新排列
        maskCode[layer] = np.reshape(codes, W.shape)
        codeBookSize = len(codebook[layer])
        a = maskCode[layer].flatten()
        b = xrange(len(a))

        codeDict[layer] = {}
        for i in xrange(len(a)):
            # codeDict保存每个码有哪些位置，而maskCode保存每个位置属于哪个码
            codeDict[layer].setdefault(a[i], []).append(b[i])

    return codebook, maskCode, codeDict


def analyze_log(fileName):
    data = open(fileName, "r")
    y = []
    for line in data:
        y.append(float(line.split()[0]))
    return y

# 读入测试数据
def parse_caffe_log(log):
    lines = open(log).readlines()
    try:
        res = map(lambda x: float(x.split()[-1]), lines[-3:-1])
    except Exception as e:
        print e
        res = [0.0, 0.0]
    return res

# 检测量化后网络的精度
def test_quantize_accu(test_net):
    test_iter = 100
    test_loss = 0
    accuracy = 0
    for test_it in range(test_iter):
        # 进行一次测试
        test_net.forward()
        # 计算test loss
        test_loss += test_net.blobs['loss'].data
        # 计算test accuracy
        accuracy += test_net.blobs['accuracy'].data

    return (test_loss / test_iter), (accuracy / test_iter)


def save_quantize_net(codebook, maskcode, net_filename, total_layers):
    # 编码
    quantizeNet = {}
    for layer in total_layers:
        quantizeNet[layer+'_codebook'] = np.float32(codebook[layer])
        quantizeNet[layer + '_maskcode'] = np.int8(maskcode[layer])

    np.savez(net_filename,quantizeNet)

# 保存修剪量化的网络参数
def save_pruned_quantize_net(codebook, maskcode, net_filename, total_layers):
    # W_flatten: 扁平化的权重矩阵
    # num_level: 量化级别
    quantizeNet = {}
    for layer in total_layers:
        W_flatten = maskCode[layer].flatten()
        indx = 0
        num_level = 8
        csc_W = []
        csc_indx = []
        for n in range(len(W_flatten)):
            if W_flatten[n]!=0 or indx == 2**num_level:
                csc_W.append(W_flatten[n])
                csc_indx.append(indx)
                indx = 0
            else:
                indx += 1
        if indx!=0:
            csc_W.append(0)
            csc_indx.append(indx-1)
        print max(csc_indx)
        quantizeNet[layer + '_codebook'] = np.float32(codebook[layer])
        quantizeNet[layer + '_maskcode_W'] = np.array(csc_W, dtype=np.int8)
        print max(csc_indx)
        quantizeNet[layer + '_maskcode_indx'] = np.array(csc_indx, dtype=np.int8)

    np.savez(net_filename, quantizeNet)


caffe.set_mode_gpu()
caffe.set_device(0)

caffe_root = '../../'
model_dir = caffe_root + 'models/mnist/'
deploy = model_dir + 'deploy.prototxt'
solver_file = model_dir + 'solver.prototxt'
# model_name = 'LeNet5_Mnist_shapshot_iter_10000'
model_name = 'LeNet5_Mnist_shapshot_iter_10000_pruned'
caffemodel = model_dir + model_name + '.caffemodel'

dir_t = '/weight_quantize/'

# 运行测试命令
args = dict(lr=0.01, decay_rate = 0.0009, momentum = 0.9, update = 'adadelta', normalize_flag = False)

start_time = time.time()

solver = caffe.SGDSolver(solver_file)
solver.net.copy_from(caffemodel)
# 需要量化的权重
total_layers = ['conv1','conv2','ip1','ip2']

num_c = 2 ** 8 # 量化级别，由8位整数表示
codebook = kmeans_net(solver.test_nets[0], total_layers, num_c)

codeDict, maskCode = quantize_net_with_dict(solver.test_nets[0], total_layers, codebook)
quantize_net_caffemodel = model_dir + model_name + '_quantize.caffemodel'
solver.test_nets[0].save(quantize_net_caffemodel)

quantize_net_npz = model_dir + model_name + '_quantize_net'
save_pruned_quantize_net(codebook, maskCode, quantize_net_npz , total_layers)

# 迭代训练编码表
accuracys = []
co_iters = 40
ac_iters = 10
for i in xrange(2500):
    if (i % (co_iters + ac_iters) == 0 and i > 0):
        # 重新量化
        # 导入训练后的
        codebook = kmeans_net(solver.net, total_layers, num_c)
        codeDict, maskCode = quantize_net_with_dict(solver.net, total_layers, codebook)
        solver.net.save(quantize_net_caffemodel)
        solver.test_nets[0].copy_from(quantize_net_caffemodel)
        _, accu = test_quantize_accu(solver.test_nets[0])
        accuracys.append(accu)

    solver.step(1)
    if (i % (co_iters + ac_iters) < co_iters):
        # 码表更新
        update_codebook_net(solver.net, codebook, codeDict, maskCode, args=args, update_layers=total_layers)

    print "Iter:%d, Time cost:%f" % (i, time.time() - start_time)

plt.plot(accuracys, 'r.-')
plt.show()