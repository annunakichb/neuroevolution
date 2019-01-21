
import pygraphviz as pgv
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

import numpy as np
import math
import queue
import copy

import utils.collections as collections



class Node:
    span_rate = 0.45     # 有效的宽度边界，正态分布计算结果小于该值表示无效
    grid_num = 10        # 网络每维的数量
    max_spilt_count=3    # 一个节点最多分成三个子节点
    width_scale = 0.8    # 宽度缩小系数
    max_sigma = 1.0      # 最大宽度
    min_sigma = 0.1      # 最小宽度
    '''
    网络节点
    '''
    def __init__(self,center=np.array([0.0]),sigma=1.0):
        '''
        网络节点，每个节点类似RBF的隐节点
        '''

        # 节点本身特性
        self.center = center     # 均值
        self.sigma = sigma       # 跨度
        self.childs = []          # 子节点
        self.parent = None        # 父节点

        # 归属节点的样本及其特性
        self.samples = []        # 属于该节点的样本
        self.initGrids()


    @property
    def dimension(self):
        return len(self._center)

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self,c):
        '''
        设置中心点
        :param center: Union(float,list,ndarray)
        :return:
        '''
        if isinstance(c,float):
            c = [c]
        if not isinstance(c,np.ndarray):
            c = np.array(c)
        self._center = c

    @property
    def sigma(self):
        '''
        节点的作用宽度，是指节点发放率大于Node.span_rate的取值范围
        :return:
        '''
        return self._sigma

    @sigma.setter
    def sigma(self,value):
        '''
        设置节点的宽度，以及方差
        :param Union(float,list,np.array) value
        :return:
        '''
        if isinstance(value,float):
            if self.dimension == 1:
                value = [[value]]
            else:
                temp = value
                value = np.ones((self.dimension,self.dimension))
                for i in range(self.dimension):
                    value[i][i] = 1
                    for j in range(self.dimension):
                        if i == j: continue
                        value[i][j] = temp

        if not isinstance(value,np.ndarray):
            value = np.array(value)

        self._sigma = value

    def initGrids(self):
        '''
        初始化网格
        :return:
        '''
        dimension = len(self._center)
        shape = [Node.grid_num] * dimension
        self.grids = np.zeros(shape)
        self.gridranges = np.zeros((dimension,2))

        # 计算各个维度方向的边际分布,并根据边际分布，计算各个维度方向上的网格开始位置和网格宽度
        for i in range(dimension):
            # 第i个维度方向上的方差和均值
            s = self._sigma[i][i]
            u = self._center[i]
            # 计算第i个方向上高斯分布值为Node.span_rate的位置,作为网格在该维度的开始位置
            begin = -np.sqrt(-1*np.log(Node.span_rate)) * 2 * s + u
            end = np.sqrt(-1*np.log(Node.span_rate)) * 2 * s + u
            self.gridranges[i][0] = begin  # 网格起始点
            self.gridranges[i][1] = abs((end - begin) / Node.grid_num)  # 网格宽度


    def value(self, input):
        '''
        按照高斯分布计算值
        :param input:  Union(float,list,np.ndarray) 输入样本向量
        :return: 高斯分布计算值
        '''
        if isinstance(input,float):
            input = [input]
        if not isinstance(input,np.ndarray):
            input = np.array(input)

        diff = input - self._center
        value = np.dot(np.mat(diff),np.mat(self._sigma).I)
        value = np.dot(value,np.mat(diff).T)
        if isinstance(value,float):
            print(value)
        #print(value)
        value = np.exp(-1 * value / 2)
        if value < Node.span_rate:
            return 0.
        return value[0][0]


    @property
    def firerate(self):
        if len(self.samples)<=0:
            return 0.0
        return np.sum([self.value(s) for s in self.samples])


    def scale(self,value):
        '''
        按比例修改sigma
        :param value:
        :return:
        '''
        self._sigma *= value
        self.initGrids()
        collections.foreach(self.samples,lambda s:self.putgrid(s))

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self,c):
        self._cost = c

    def getgridcenter(self,gridpos):
        '''
        取得网格中心
        :param gridpos: tuple 网格编号
        :return:  ndarray 网格中心点坐标
        '''
        return np.array([r[0]+gridpos[i]*r[1]+r[1]/2 for i,r in enumerate(self.gridranges)])

    def putgrid(self,sample):
        '''
        放置网格
        :param sample:
        :return:
        '''
        indexes = []
        for i in range(self.dimension):
            s = sample[i]
            c = self._center[i]
            gridwidth = self.gridranges[i][1]
            pos = int((s - c) / gridwidth)
            if pos <= -1 * Node.grid_num / 2:
                pos = 0
            elif pos >= Node.grid_num / 2:
                pos = Node.grid_num - 1
            else:
                if s < c:
                    pos = Node.grid_num / 2 - 1 + pos
                else:
                    pos = Node.grid_num / 2 + pos
            indexes.append(pos)

        indexes = tuple(indexes)

        old_value = self.grids.take(indexes)
        self.grids.put(indexes, old_value + 1)

    def put_sample(self,sample):
        '''
        记录一个样本，并计算样本在网格中的位置
        :param sample:
        :return:
        '''
        self.samples.append(sample)
        self.putgrid(sample)

    def __str__(self):
        return str(self.center)+","+str(self.width)

class Net:
    '''
    网络
    '''
    eplison = 0.01          # 样本距离系数，样本的映射向量距离大于它乘以样本书的平方根，则是可区分样本对
    min_sensitivty = 0.85   # 识别度下限，当识别度低于这个值的时候，就要进行分裂操作
    sensitivty_lower_num = 1# 识别度连续下降的次数，当识别度连续下降的次数超过该值，则进行分裂操作
    cost_k1 = 1.0           # 计算能耗中节点数量的比例参数
    cost_k2 = 1.0           # 计算能耗中发放耗能的比例参数
    max_energy_cost = 0.33  # 能耗上限系数
    node_sample_density=0.08# 节点样本密度


    def __init__(self,root=None):
        '''
        网络是由节点构成的树结构
        :param root:
        '''
        self.root = Node() if root is None else root    #根节点
        self.nodescount = 1   #节点总数
        self.leafs = [self.root]       #叶子节点数

        self.reset()
        self.spiltcount = 0  # 发生分裂的次数
        self.reconstructcount = 0  # 发生重构的次数
        self.sensitivty = []  # 网络灵敏度
        self.energycost = []            #网络运行所有样本的能量消耗

    def reset(self):
        '''
        重置所有有记录在网络内部的输入样本
        :return:
        '''
        self.samples = []               #所有样本，在线会动态变化，离线则不变
        self.differentablecount = 0     #所有样本对中属于可区别集的数量
        self.activation_energy_cost = 0 # 节点激活的能耗

    def put(self,sample):
        '''
         online learning method is adopted in the network, with one sample input at a time
        :param sample: one sample
        :return:
        '''
        # 计算样本的隐藏层映射向量
        h = self.__hidden_vector__(sample)

        # 如果h所有分量都为0,则添加新节点
        if collections.all(h,lambda i:i<=0):
            n = Node(center=sample,sigma=1.0)
            n.parent = self.root
            self.leafs.append(n)
            return

        # 计算归属节点,即该样本归属于发放概率最大的那个节点
        i = np.argmax(h)                 # h中多个相同的时候argmax只返回第一个
        self.leafs[i].put_sample(sample)

        # 如果是第一个样本，则并计算所有节点的激活能耗
        if len(self.samples)<=0:
            self.samples.append(sample)
            self.evulate()
            return
        # 将新样本的激活能耗加入到总能耗中去
        #self.activation_energy_cost += np.sum(h)
        # 　计算新样本与已有样本可区分度
        #for s in self.samples:
        #    h1 = self.__hidden_vector__(s)
        #    dis = np.linalg.norm(np.array(h1) - np.array(h))
        #    if dis > Net.eplison*max(1.0,math.sqrt(len(self.leafs))):
        #        self.differentablecount += 1
        # 新样本加入集合
        self.samples.append(sample)

    def dochange(self):
        # 计算网络平均能耗
        if len(self.energycost) <= 0:
            self.energycost.append(self.computeEnergycost())

        # 计算目前网络的数据敏感度
        if len(self.sensitivty) <= 0:
            self.sensitivty.append(self.computeSenstivity())

        #print('sensitivty=%.2f,cost=%.2f\n' % (self.sensitivty[-1], self.energycost[-1]))

        action  = ''
        # 敏感度低于下限，执行网络节点分裂过程
        if self.sensitivty[-1] != 0 and self.sensitivty[-1] < Net.min_sensitivty:
            action = '分裂(敏感度低于阈值)'
            print('sensitivty=%.2f,cost=%.2f,action=%s\n' % (self.sensitivty[-1], self.energycost[-1], action))
            self.__spilt()
            self.spiltcount += 1
        # 敏感度连续Net.sensitivty_lower_num次不升高
        elif len(self.sensitivty) > Net.sensitivty_lower_num + 1 and self.isDescending(
                self.sensitivty[-1 - Net.sensitivty_lower_num:]):
            action = '分裂(敏感度连续下降)'
            print('sensitivty=%.2f,cost=%.2f,action=%s\n' % (self.sensitivty[-1], self.energycost[-1], action))
            self.__spilt()
            self.spiltcount += 1
        # 如果网络能耗高于上限，执行网络重构操作
        #elif self.energycost[-1] > Net.max_energy_cost:
        elif self.energycost[-1] > Net.cost_k1 * len(self.leafs) + Net.cost_k2 * math.exp(-len(self.leafs) / 5) * len(
                self.leafs):
            action = '重构'
            print('sensitivty=%.2f,cost=%.2f,action=%s\n' % (self.sensitivty[-1], self.energycost[-1], action))
            self.__reconstruct()
            self.reconstructcount += 1

        else:
            print('sensitivty=%.2f,cost=%.2f,action=%s\n' % (self.sensitivty[-1], self.energycost[-1], action))




        #print('sample=%s,sensitivty=%.2f,cost=%.2f,action=%s\n' % (str(sample), self.sensitivty[-1], self.energycost[-1],action))

    def isDescending(self,vs):
        return all(x > y for x, y in zip(vs, vs[1:]))

    def evulate(self):
        '''
        计算所有样本对于当前网络的可区分样本对和样本总能耗
        :return:
        '''
        self.differentablecount = 0
        self.activation_energy_cost = 0
        for i in range(len(self.samples)):
            hi = self.__hidden_vector__(self.samples[i])
            self.activation_energy_cost += np.sum(hi)
            for j in range(i+1,len(self.samples)):
                hj = self.__hidden_vector__(self.samples[j])
                dis = np.linalg.norm(np.array(hi) - np.array(hj))
                if dis > Net.eplison * max(1.0, math.sqrt(len(self.leafs))):
                    self.differentablecount += 1

        return self.differentablecount,self.activation_energy_cost

    def computeEnergycost(self):
        '''
        计算网络能耗指数
        :return:
        '''
        return Net.cost_k1 * len(self.leafs) + Net.cost_k2 * self.activation_energy_cost / len(self.samples)

    def computeSenstivity(self):
        '''
        计算网络敏感度
        :return:
        '''
        length = len(self.samples)
        return self.differentablecount / (length * (length - 1) / 2)

    def __hidden_vector__(self,sample):
        '''
        计算某个样本的隐藏层向量值
        :param sample: list 输入样本
        :return: list  隐藏层计算的向量
        '''
        return [leaf.value(sample) for leaf in self.leafs]

    def __spilt(self,node=None):
        '''
        执行节点分裂，找到
        :return:
        '''
        # 找到待分裂的节点，是所有节点中发放率总和最高的节点
        if node is None:
            firerates = np.array(list(map(lambda leaf:leaf.firerate,self.leafs)))
            index = np.argmax(firerates)
            node = self.leafs[index]
        # 找到该节点的分裂区，是样本出现最频繁的网格
        spiltareas = []
        spiltpositions = []
        while len(spiltareas) < Node.max_spilt_count:            # 分裂区数量不能大于设定
            maxvalue = np.amax(node.grids)                       # 取得节点中网格统计的最大值
            if maxvalue<=0:break
            position = np.where(node.grids == np.max(node.grids))
            if len(position[0]) >= 2:   # 如果有多个位置点索引，就只取得第一个 [[0,0],[1,7]]
                if len(position) == 1:
                    position = tuple(np.array([position[0][0]]))
                elif len(position)>=2:
                    position = tuple([np.array([position[0][0]]),np.array([position[1][0]])])

            pos_center = node.getgridcenter(position)
            #position = np.delete(position, range(1, len(position[0])), axis=1)
            # 如果是第一个分裂区，直接加入
            if len(spiltareas)<=0:
                spiltareas.append(pos_center)
                spiltpositions.append(copy.deepcopy(position))
                node.grids.put(position,0.0)
                continue
            # 否则需要计算距离
            if not collections.any(spiltpositions,lambda pos:np.sum([abs(p[0]-p[1]) for p in zip(pos,position)])<=len(position)):
                spiltareas.append(pos_center)
                spiltpositions.append(copy.deepcopy(position))
            node.grids.put(position, 0.0)

        if len(spiltareas) <=0 :
            return
        # 如果只有一个分裂区，则不创建子节点，而调整当前节点
        if len(spiltareas)<=1:
            node._center = spiltareas[0]
            node.scale(Node.width_scale)
            #self.__evulate()
            return


        # 创建子节点
        self.leafs.remove(node)
        for spiltarea in spiltareas:
            sigma = node._sigma * Node.width_scale
            n = Node(spiltarea,sigma)
            n.parent = node
            node.childs.append(n)
            self.leafs.append(n)


        # 将属于父节点的样本分配到子节点
        for s in node.samples:
            index = np.argmin(list(map(lambda c:np.linalg.norm(c-s),spiltareas)))
            if collections.all(node.childs[index].samples,lambda x:np.linalg.norm(s,x)!=0):
                node.childs[index].put_sample(s)

        node.samples = []
        self.nodescount += len(spiltareas)
        #self.__evulate()

        #self.drawdata()
    def __reconstruct(self):
        '''
        重构以减少平均能耗
        :return:
        '''
        if len(self.leafs)<=2:
            return

        # 找到样本数量最少的节点
        minIndex = np.argmin(list(map(lambda l:len(l.samples),self.leafs)))
        minNode = self.leafs[minIndex]
        minNode.parent.childs.remove(minNode)

        # 在叶子节点中找到中心点距离这个宽度最小节点最近的节点
        mergeIndex = np.argmin(list(map(lambda l:np.linalg.norm(np.array(l._center) - np.array(self.leafs[minIndex]._center)),self.leafs[:minIndex]+self.leafs[minIndex+1:])))
        mergeNode = self.leafs[mergeIndex]

        # 将宽度最小节点并入到最近节点中去
        newcenter = np.average([minNode.center,mergeNode.center],axis=0)
        newsigma = mergeNode.sigma / Node.width_scale
        mergeNode.center = newcenter
        mergeNode.sigma = newsigma

        # 删除待合并的叶节点
        self.leafs.remove(minNode)

        # 判断是否要将待合并节点的父节点加入到叶子中去
        if len(minNode.parent.childs) <= 0:
            self.leafs.append(minNode.parent)

        # 重新分配样本到叶子节点中
        for s in minNode.samples:
            h = self.__hidden_vector__(s)
            i = np.argmax(h)  # h中多个相同的时候argmax只返回第一个
            self.leafs[i].put_sample(s)

        self.nodescount -= 1
        #self.__evulate()
        #self.drawdata()

    def drawdata(self,sn):
        fig = plt.figure(3)
        asp = fig.add_subplot(111)
        for node in self.leafs:
            mu, sigma = node.center, node.sigma
            sampleNo = 200
            np.random.seed(0)
            s = np.random.normal(mu, sigma, sampleNo)
            asp.hist(s, bins=200)
        plt.show()
        fig.savefig('distribution'+str(sn)+'.png')

    def drawnode(self,node):
        if node is None:
            return
        mu, sigma = node.center, node.sigma
        sampleNo = 1000
        np.random.seed(0)
        s = np.random.normal(mu, sigma, sampleNo)
        plt.hist(s, bins=100, normed=True)

        for child in node.childs:
            self.drawnode(child)


    def draw(self,sn):
        # 用队列做BFS
        director_queue = queue.Queue()
        director_queue.put(self.root)
        # 初始化一个图
        tree_graph = pgv.AGraph(directed=True, strict=True)
        tree_graph.node_attr['style'] = 'filled'
        tree_graph.node_attr['shape'] = 'square'

        # 创建图
        self.__createGraph(tree_graph,self.root,None)

        # 生成图
        tree_graph.graph_attr['epsilon'] = '0.001'
        tree_graph.string()  # print dot file to standard output
        tree_graph.write('nettree'+str(sn)+'.dot')
        tree_graph.layout('dot')  # layout with dot
        tree_graph.draw('nettree'+str(sn)+'.png')  # write to file



    def __createGraph(self,tree_graph,node,label):
        if node is Node:
            return
        if label is None:
            label = "{:.3f}\n%{:.3f}\n{:d}".format(float(node.center),node.width,len(node.samples))
            tree_graph.add_node(label)

        clabels = []
        for child in node.childs:
            clabel = "{:.3f}\n%{:.3f}\n{:d}".format(float(child.center),child.width,len(child.samples))
            clabels.append(clabel)
            tree_graph.add_node(clabel)
            tree_graph.add_edge(label,clabel)

        for child,clabel in zip(node.childs,clabels):
            self.__createGraph(tree_graph,child,clabel)



def createSamples(type):
    if type == 1: # 一维正态分布
        values = np.random.normal(0.0, 1.0, 50)
        np.random.shuffle(values)
        # 正规化到0-1之间
        values = (values - values.min()) / (values.max() - values.min())
        values = np.array([[v] for v in values])
        return values
    elif type == 2: # 两个一维正态分布混合
        # 根据标准正态分布采样100个值
        values1 = np.random.normal(0.0, 1.0, 30)
        values2 = np.random.normal(10.0, 1.0, 30)
        values = np.append(values1, values2)
        np.random.shuffle(values)

        # 正规化到0-1之间
        values = (values - values.min()) / (values.max() - values.min())
        values = np.array([[v] for v in values])
        return values
    elif type == 4: # 二维正态分布
        mean1 = [0, 0]
        cov1 = [[1, 0], [0, 10]]
        values = np.random.multivariate_normal(mean1, cov1, 30)

        mean2 = [10, 10]
        cov2 = [[10, 0], [0, 1]]
        values = np.append(values, np.random.multivariate_normal(mean2, cov2, 30), 0)
        np.random.shuffle(values)
        values = (values - values.min()) / (values.max() - values.min())
        return values
    elif type == 5:
        df = pd.read_excel("/Volumes/data/shared/perfume_data.xlsx", header=None)
        rownum, colnum = df.shape
        values = []
        lables = []
        for index in df.index:
            classname = df.loc[index].values[0]
            print(classname)
            row = df.loc[index].values[1:-1]
            data1 = [v // 1000 for v in row]
            data1 = (data1 - data1.min()) / (data1.max() - data1.min())
            data2 = [v % 1000 for v in row]
            data2 = (data2 - data2.min()) / (data2.max() - data2.min())
            datas = zip(data1, data2)
            values.append(datas)
            #lables.append(classname)
            # plt.scatter(data1, data2)
        np.random.shuffle(values)
        return values
        #return values, lables

if __name__ == '__main__':

    Net.max_energy_cost = 0.0
    max_cost = []
    max_sen = []
    real_cost = []
    node_cost = []
    count_node = []
    count_leaf = []

    opitmal_net = None
    for k in range(10):
        # 构造初始网络
        root = Node(center=np.array([0.5]), sigma=1.0)
        net = Net(root)
        leafcount = len(net.leafs)

        Net.max_energy_cost += 1.0
        max_cost.append(Net.max_energy_cost)

        for i in range(50):
            # 根据标准正态分布采样100个值
            values = createSamples(1)

            # 逐个输入样本
            for k,value in enumerate(values):
                net.put(value)

            # 计算能耗
            net.evulate()
            net.energycost.append(net.computeEnergycost())
            net.sensitivty.append(net.computeSenstivity())


            # 结构改变
            net.dochange()

            # 显示结果
            #net.draw(i)
            #net.drawdata(i)



            # 准备下一次迭代
            net.reset()

        print("分裂次数=" + str(net.spiltcount) + ",重构次数=" + str(net.reconstructcount) + ",节点总数=" + str(
            net.nodescount) + ",叶子节点数=" + str(len(net.leafs)) + '\n')

        if opitmal_net is None or (opitmal_net.sensitivty[-1] < net.sensitivty[-1] and net.energycost[-1] < Net.max_energy_cost):
            opitmal_net = net

        max_sen.append(net.sensitivty[-1])
        real_cost.append(net.energycost[-1])
        node_cost.append((net.energycost[-1]-Net.cost_k1*len(net.leafs))/(Net.cost_k2*len(net.leafs)))
        count_node.append(net.energycost[-1])
        count_leaf.append(len(net.leafs))

    fig = plt.figure(1)
    sp = fig.add_subplot(321)
    sp.plot(max_cost,max_sen)

    sp = fig.add_subplot(322)
    sp.plot(max_cost, real_cost)

    sp = fig.add_subplot(323)
    sp.plot(max_cost, node_cost)

    #sp = fig.add_subplot(324)
    #sp.plot(max_cost,count_node)

    sp = fig.add_subplot(324)
    sp.plot(max_cost,count_leaf)

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(max_sen, real_cost, count_leaf, label=' ')

    plt.show()

    opitmal_net.drawdata(1)
