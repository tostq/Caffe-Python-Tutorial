# -*- coding:utf-8 -*-
# 用于修剪网络模型
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe

# 由稠密变成CSC稀疏矩阵
def dense_to_sparse_csc(W_flatten, num_level):
    # W_flatten: 扁平化的权重矩阵
    # num_level: 量化级别
    csc_W = [] # 存储稀疏矩阵
    csc_indx = []
    indx = 0
    for n in range(len(W_flatten)):
        if W_flatten[n]!=0 or indx == 2**num_level:
            csc_W.append(W_flatten[n])
            csc_indx.append(indx)
            indx = 0
        else:
            indx += 1
    if indx!=0:
        csc_W.append(0.0)
        csc_indx.append(indx-1)
    return np.array(csc_W, dtype=np.float32),np.array(csc_indx, dtype=np.int8)

# 由稠密变成CSC稀疏矩阵
def sparse_to_dense_csc(csc_W, csc_W_indx):
    # W_flatten: 扁平化的权重矩阵
    # num_level: 量化级别
    W_flatten = [] # 存储稠密矩阵
    indx = 0
    for n in range(len(csc_W)):
        if csc_W_indx[n]!=0:
            W_flatten.extend([0]*(csc_W_indx[n]))
        W_flatten.append(csc_W[n])
    return np.array(W_flatten, dtype=np.float32)


def read_sparse_net(filename, net, layers):
    pass

def write_sparse_net(filename, net):
    pass

# 画出各层参数的直方图
def draw_hist_weight(net, layers):
    plt.figure()  # 画图
    layer_num = len(layers)
    for i, layer in enumerate(layers):
        i += 1
        W = net.params[layer][0].data

        plt.subplot(layer_num/2, 2, i)
        numBins = 2 ^ 5
        plt.hist(W.flatten(), numBins, color='blue', alpha=0.8)
        plt.title(layer)
        plt.show()

# 网络模型的参数
def analyze_param(net, layers):

    print '\n=============analyze_param start==============='
    total_nonzero = 0
    total_allparam = 0
    percentage_list = []
    for i, layer in enumerate(layers):
        i += 1
        W = net.params[layer][0].data
        b = net.params[layer][1].data

        print 'W(%s) range = [%f, %f]' % (layer, min(W.flatten()), max(W.flatten()))
        print 'W(%s) mean = %f, std = %f' % (layer, np.mean(W.flatten()), np.std(W.flatten()))
        non_zero = (np.count_nonzero(W.flatten()) + np.count_nonzero(b.flatten())) # 参数非零值
        all_param = (np.prod(W.shape) + np.prod(b.shape)) # 所有参数的数目
        this_layer_percentage = non_zero / float(all_param) # 参数比例
        total_nonzero += non_zero
        total_allparam += all_param
        print 'non-zero W and b cnt = %d' % non_zero
        print 'total W and b cnt = %d' % all_param
        print 'percentage = %f\n' % (this_layer_percentage)
        percentage_list.append(this_layer_percentage)

    print '=====> summary:'
    print 'non-zero W and b cnt = %d' % total_nonzero
    print 'total W and b cnt = %d' % total_allparam
    print 'percentage = %f' % (total_nonzero / float(total_allparam))
    print '=============analyze_param ends ==============='
    return (total_nonzero / float(total_allparam), percentage_list)

def prune(threshold, test_net, layers):
    sqarse_net = {}

    for i, layer in enumerate(layers):

        print '\n============  Pruning %s : threshold=%0.2f   ============' % (layer,threshold[i])
        W = test_net.params[layer][0].data
        b = test_net.params[layer][1].data
        hi = np.max(np.abs(W.flatten()))
        hi = np.sort(-np.abs(W.flatten()))[int((len(W.flatten())-1)* threshold[i])]

        # abs(val)  = 0         ==> 0
        # abs(val) >= threshold ==> 1
        interpolated = np.interp(np.abs(W), [0, hi * threshold[i], 999999999.0], [0.0, 1.0, 1.0])

        # 小于阈值的权重被随机修剪
        random_samps = np.random.rand(len(W.flatten()))
        random_samps.shape = W.shape

        # 修剪阈值
        # mask = (random_samps < interpolated)
        mask = (np.abs(W) > (np.abs(hi)))
        mask = np.bool_(mask)
        W = W * mask

        print 'non-zero W percentage = %0.5f ' % (np.count_nonzero(W.flatten()) / float(np.prod(W.shape)))
        # 保存修剪后的阈值
        test_net.params[layer][0].data[...] = W
        # net.params[layer][0].mask[...] = mask
        csc_W, csc_W_indx = dense_to_sparse_csc(W.flatten(), 8)
        dense_W = sparse_to_dense_csc(csc_W, csc_W_indx)
        sqarse_net[layer + '_W'] = csc_W
        sqarse_net[layer + '_W_indx'] = csc_W_indx

    # 计算修剪后的权重稀疏度
    # np.savez(model_dir + model_name +"_crc.npz",sqarse_net) # 保存存储成CRC格式的稀疏网络
    (total_percentage, percentage_list) = analyze_param(test_net, layers)
    test_loss, accuracy = test_net_accuracy(test_net)
    return (threshold, total_percentage, percentage_list, test_loss, accuracy)

def test_net_accuracy(test_net):
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


def eval_prune_threshold(threshold_list, test_prototxt, caffemodel, prune_layers):
    def net_prune(threshold, test_prototx, caffemodel, prune_layers):
        test_net = caffe.Net(test_prototx, caffemodel, caffe.TEST)
        return prune(threshold, test_net, prune_layers)

    accuracy = []
    for threshold in threshold_list:
        results = net_prune(threshold, test_prototxt, caffemodel, prune_layers)
        print 'threshold: ', results[0]
        print '\ntotal_percentage: ', results[1]
        print '\npercentage_list: ', results[2]
        print '\ntest_loss: ', results[3]
        print '\naccuracy: ', results[4]
        accuracy.append(results[4])
    plt.plot(accuracy,'r.')
    plt.show()

# 迭代训练修剪后网络
def retrain_pruned(solver, pruned_caffemodel, threshold, prune_layers):
    #solver = caffe.SGDSolver(solver_proto)
    retrain_iter = 20

    accuracys = []
    for i in range(retrain_iter):
        solver.net.copy_from(pruned_caffemodel)
        # solver.solve()
        solver.step(500)
        _,_,_,_,accuracy=prune(threshold, solver.test_nets[0], prune_layers)
        solver.test_nets[0].save(pruned_caffemodel)
        accuracys.append(accuracy)

    plt.plot(accuracys, 'r.-')
    plt.show()


#CPU或GPU模型转换
#caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()

caffe_root = '../../'
#model_dir = caffe_root + 'models/SSD_300x300/'
#deploy = model_dir + 'deploy.prototxt'
#model_name = 'VGG_VOC0712_SSD_300x300_iter_60000'
#caffemodel = model_dir + model_name + '.caffemodel'

model_dir = caffe_root + 'models/mnist/'
deploy = model_dir + 'deploy.prototxt'
model_name = 'LeNet5_Mnist_shapshot_iter_10000'
caffemodel = model_dir + model_name + '.caffemodel'
test_prototxt = model_dir + 'test.prototxt'
solver_proto = model_dir + 'solver.prototxt'

solver = caffe.SGDSolver(solver_proto)

# 要修剪的层
prune_layers = ['conv1','conv2','ip1','ip2']
# 测试修剪率
test_threshold_list = [[0.3, 1 ,1 ,1], [0.4, 1 ,1 ,1], [0.5, 1 ,1 ,1], [0.6, 1 ,1 ,1], [0.7, 1 ,1 ,1],
                  [1, 0.05, 1, 1], [1, 0.1, 1, 1], [1, 0.15, 1, 1], [1, 0.2, 1, 1], [1, 0.3, 1, 1],
                  [1, 1, 0.05, 1], [1, 1, 0.1, 1], [1, 1, 0.15, 1], [1, 1, 0.2, 1], [1, 1, 0.3, 1],
                  [1, 1, 1, 0.05], [1, 1, 1, 0.1], [1, 1, 1, 0.15], [1, 1, 1, 0.2], [1, 1, 1, 0.3]]

# 验证修剪率
#eval_prune_threshold(test_threshold_list, test_prototxt, caffemodel, prune_layers)

threshold = [0.3, 0.1, 0.01, 0.2]
prune(threshold, solver.test_nets[0], prune_layers)
pruned_model = model_dir + model_name +'_pruned' + '.caffemodel'
solver.test_nets[0].save(pruned_model)

retrain_pruned(solver, pruned_model, threshold, prune_layers)



"""
# 各层对应的修剪率
threshold = [0.3, 0.1, 0.01, 0.2]

net = caffe.Net(deploy, caffemodel, caffe.TEST)
# 修剪
prune(threshold, net, prune_layers, test_prototxt)

# 保存修剪后的稀疏网络模型
output_model = model_name +'_pruned' + '.caffemodel'
net.save(output_model)
"""