import torch
import numpy
import matplotlib.pyplot as plt
import random
import numpy as np
import math
import xlrd
from sklearn import datasets
import copy
list_1 = []
list_2 = []

def loadDataSet(path, dim):
    """
    加载所需的数据集
    path:数据集路径
    dim:数据维度
    return：数据data, 距离矩阵Dis, dic_eps, dis_mean, Plist
    """
    with open(path, 'r') as fd:
        raw_data = fd.readlines()
    headline = raw_data[0].strip().split(",")[1:]
    data0 = [it.strip().split(",") for it in raw_data[1:]]
    data = np.float16(data0)
    coordinate = [i[1:] for i in data]
    count = len(data)
    keys = [it[0] for it in data]   # 行名是一级索引（序号）
    sub_keys = headline[:]    # 列名是二级索引（行号）
    dic = dict(zip(keys, [{} for i in range(len(keys))]))
    for line in data:
        key = line[0]
        if dic[key] != {}:
            print("Error: repeated values", key, dic[key])
        else:
            value = dict(zip(sub_keys, [int(it) for it in line[1:]]))
            dic[key] = value  
    Dis = [i for i in range(count)]    # 距离分布矩阵
    Dis_sort = [i for i in range(count)]   # 距离分布排序后的矩阵
    for row in range(count):
        Dis[row] = [np.linalg.norm(coordinate[row] - coordinate[r], ord=2) for r in range(count)]
        Dis_sort[row] = sorted(Dis[row])
    a = np.array(Dis_sort) # 转为数组
    dis_mean = np.mean(a, axis = 0)  # 列平均值，即不同的Eps值
    Plist = [0 for i in range(count)]  # 存储不同Eps值下的MinPts
    keys_eps = dis_mean   # eps是一级索引
    subKeys_eps = [i for i in range(count)]   # 样本的序号是二级索引
    dic_eps = dict(zip(keys_eps, [{} for i in range(count)]))
    for index in range(count):
        for i in range(count):
            value = []
            for j in range(count):
                if(Dis[i][j] <= dis_mean[index]):
                    value.append(j)
                    Plist[index] += 1
            dic_eps[dis_mean[index]][i] = value
    Plist = np.divide(Plist, count);
    return data, Dis, dic_eps, dis_mean, Plist
 
# 计算两个点之间的欧式距离，参数为两个元组
#def dist(t1, t2):
#    dis = math.sqrt((np.power((t1[0]-t2[0]),2) +
#    np.power((t1[1]-t2[1]),2)))#power(x,y)函数，计算x的y次方
#    # print("两点之间的距离为："+str(dis))
#    return dis

def DBscan(Dic, Eps, MinPts):
    """
    Dic:数据集字典
    Eps:邻域半径参数
    MinPts:制定邻域密度阈值（核心点阈值）
    """
    count = len(Eps)
    value_k = []
    C = [i for i in range(count)]
    for eps in Eps:
        index = np.where(Eps == eps)[0][0]
        minpts = MinPts[index]
        UV = [i for i in range(count)]     #总列表中没有访问的点
        visited = []
        C[index] = [-1 for i in range(count)]    # C为输出结果，默认是一个长度为num的值全为-1的列表
        k = -1     # 用k来标记不同的簇，k = -1表示噪声点
        while len(UV) > 0:   # 如果还有没访问的点
            p = random.choice(UV)   # 随机选择一个unvisited对象;
            UV.remove(p)
            unvisited = Dic[eps][p]
            visited = []   # 已经访问的点的列表
            CorePoints = []  # 核心点的列表
            #unvisited.remove(p); # 所有的点，自己的eps范围内的值包含了自己
            visited.append(p)
            if(len(unvisited) >= minpts): # 判断是否是核心点
                k += 1
                C[index][p] = k
                CorePoints.append(p)   # 将核心点放入列表
                for pi in unvisited:    # 对于p的epsilon邻域中的每个对象pi
                    if(pi in UV):
                        UV.remove(pi)
                        visited.append(pi)
                        if(len(Dic[eps][pi]) >= minpts):
                            unvisited.extend(Dic[eps][pi])
                            unvisited = list(set(unvisited))                        
                    if C[index][pi] == -1:# 若pi不属于任何簇，C[pi] == -1说明C中第pi个值没有改动
                        C[index][pi] = k              
            else:   # 如果p的epsilon邻域中的对象数小于指定阈值
                C[index][p] = -1
        value_k.append(k)
        if(index > 3):
            if(value_k[-4] == value_k[-5] and value_k[-2] == value_k[-3] and value_k[-2] == value_k[-4] and value_k[-1] != value_k[-2]):
                 return C[index - 1], Eps[index - 1], MinPts[index - 1]
    #return value_k

if  __name__ == "__main__":
    #导入数据集
    data_file = u'dataset.csv'
    dim = 2   # 数据维度
    dataSet, Dislist, DicEps, Epslist, Minptslist = loadDataSet(data_file, dim)   # count是指数据量
    C, eps, minpts = DBscan(DicEps, Epslist, Minptslist)
    print(C)
    x = []
    y = []
    for data in dataSet:
        x.append(data[1])
        y.append(data[2])
    plt.figure(figsize=(8, 6), dpi=80)
    plt.scatter(x,y, c=C, marker='o')
    plt.show()
    x = []
    y = []
    for i in range(99):
        x.append(i)
        y.append(Minptslist[i])
    plt.figure(figsize=(8, 6), dpi=80)
    plt.scatter(x,y, c=C, marker='o')
    plt.show()
    print(x)
    print(y)



