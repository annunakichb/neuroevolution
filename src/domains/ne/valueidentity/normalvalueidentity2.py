
import pygraphviz as pgv
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import math
import queue

import utils.collections as collections



class Node:
    span_rate = 0.35     # 有效的宽度边界
    grid_num = 10        # 网络每维的数量
    max_spilt_count=3    # 一个节点最多分成三个子节点
    width_scale = 0.8    # 宽度缩小系数
    max_width = 1.0      # 最大宽度
    min_width = 0.1      # 最小宽度
    '''
    网络节点
    '''
    def __init__(self,center=np.array([0.0]),width=1.0):
        '''
        网络节点，每个节点类似RBF的隐节点
        '''

        # 节点本身特性
        self.center = center     # 均值
        self.sigma = 1.0         # 方差
        self.childs = []         # 子节点
        self.parent = None       # 父节点
        self.span = 0.0
        self.setWidth(width)  # 跨度

        # 归属节点的样本及其特性
        self.grids = []          # 节点内部区域被划分成网格,每个网格值是落入该格子内的样本数量
        self.samples = []        # 属于该节点的样本




    def value(self,input):
        '''
        按照高斯分布计算值
        :param input:  输入样本向量
        :return: 高斯分布计算值
        '''
        return (np.exp((-1 * np.power(input - self.center, 2)) / (2 * self.sigma * self.sigma)))
        #return (np.exp((-1*np.power(input - self.center,2))/(2*self.sigma*self.sigma)))/(self.sigma * np.sqrt(2*3.14159))

    @property
    def width(self):
        '''
        节点的作用宽度，是指节点发放率大于Node.span_rate的取值范围
        :return:
        '''
        return self.span

    def setWidth(self,value):
        '''
        设置节点的宽度，以及方差
        :param value:
        :return:
        '''

        self.span = value
        x = np.array([c-value/2 for c in self.center])
        self.sigma = math.sqrt(np.power(x-self.center,2)*-1/(2*np.log(Node.span_rate)))

        dimension = len(self.center)
        shape = [Node.grid_num] * dimension
        self.grids = np.zeros(shape)

    @property
    def firerate(self):
        if len(self.samples)<=0:
            return 0.0
        return np.sum([self.value(s) for s in self.samples])

    def put_sample(self,sample):
        '''
        记录一个样本，并计算样本在网格中的位置
        :param sample:
        :return:
        '''
        dimension = len(sample)
        if len(self.grids)<=0:
            shape = [Node.grid_num]*dimension
            self.grids = np.zeros(shape)
        gridwidth = self.width / Node.grid_num

        indexes = []
        for s,c in zip(sample,self.center):
            pos = int((s-c)/gridwidth)
            if pos <= -1*Node.grid_num/2:pos = 0
            elif pos >= Node.grid_num/2:pos = Node.grid_num-1
            else:
                if s<c:pos = Node.grid_num/2-1+pos
                else:pos = Node.grid_num/2+pos
            indexes.append(pos)
        indexes = tuple(indexes)

        old_value = self.grids.take(indexes)
        self.grids.put(indexes,old_value+1)

        self.samples.append(sample)

    def __str__(self):
        return str(self.center)+","+str(self.width)

class Net:
    '''
    网络
    '''
    eplison = 0.01          # 样本距离系数，样本的映射向量距离大于它乘以样本书的平方根，则是可区分样本对
    min_sensitivty = 0.85   # 敏感度下限
    sensitivty_lower_num = 1# 敏感度连续下降的次数
    cost_k1 = 0.5           # 计算能耗中节点数量的比例参数
    cost_k2 = 0.5           # 计算能耗中发放耗能的比例参数
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
        重置所有输入样本
        :return:
        '''
        self.samples = []               #所有样本，在线会动态变化，离线则不变
        self.differentablecount = 0     #所有样本对中属于可区别集的数量
        self.activation_energy_cost = 0 # 节点激活的能耗
        #self.sensitivty = []            #网络灵敏度
        #self.energycost = []            #网络运行所有样本的能量消耗

        #self.spiltcount = 0             # 发生分裂的次数
        #self.reconstructcount = 0       # 发生重构的次数

    def put(self,sample):
        '''
         online learning method is adopted in the network, with one sample input at a time
        :param sample: one sample
        :return:
        '''
        # 计算样本的隐藏层映射向量
        h = self.__hidden_vector__(sample)

        # 计算归属节点,即该样本归属于发放概率最大的那个节点
        i = np.argmax(h)                 # h中多个相同的时候argmax只返回第一个
        self.leafs[i].put_sample(sample)

        # 如果是第一个样本，则并计算所有节点的激活能耗
        if len(self.samples)<=0:
            self.samples.append(sample)
            self.__evulate()
            return
        # 新样本与已有样本比较可区分度,并计算所有节点的激活能耗
        self.activation_energy_cost += np.sum(h)
        for s in self.samples:
            h1 = self.__hidden_vector__(s)
            dis = np.linalg.norm(np.array(h1) - np.array(h))
            if dis > Net.eplison*max(1.0,math.sqrt(len(self.leafs))):
                self.differentablecount += 1
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
        elif self.energycost[-1] > Net.max_energy_cost:
        #elif self.energycost[-1] > Net.cost_k1 * len(self.leafs) + Net.cost_k2 * math.exp(-len(self.leafs) / 5) * len(
        #        self.leafs):
            action = '重构'
            print('sensitivty=%.2f,cost=%.2f,action=%s\n' % (self.sensitivty[-1], self.energycost[-1], action))
            self.__reconstruct()
            self.reconstructcount += 1

        else:
            print('sensitivty=%.2f,cost=%.2f,action=%s\n' % (self.sensitivty[-1], self.energycost[-1], action))




        #print('sample=%s,sensitivty=%.2f,cost=%.2f,action=%s\n' % (str(sample), self.sensitivty[-1], self.energycost[-1],action))

    def isDescending(self,vs):
        return all(x > y for x, y in zip(vs, vs[1:]))

    def __evulate(self):
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
        while len(spiltareas) < Node.max_spilt_count:            #分裂区数量不能大于设定
            maxvalue = np.amax(node.grids)                       #取得节点中网格统计的最大值
            if maxvalue<=0:break
            position = np.where(node.grids == np.max(node.grids))
            pos_center = [node.center[i]-node.width/2+p[0]*node.width/Node.grid_num+node.width/(Node.grid_num*2) for i,p in enumerate(position)]

            position = np.delete(position, range(1, len(position[0])), axis=1)
            # 如果是第一个分裂区，直接加入
            if len(spiltareas)<=0:
                spiltareas.append(pos_center)
                node.grids.put(position,0.0)
                continue
            # 否则需要计算距离
            dis = [np.linalg.norm(np.array(sa) - np.array(pos_center)) for sa in spiltareas]
            if collections.any(dis,lambda  d:d<node.width/4):
                node.grids.put(position, 0.0)
                continue
            spiltareas.append(pos_center)
            node.grids.put(position, 0.0)


        if len(spiltareas) <=0 :
            return
        # 如果只有一个分裂区，则不创建子节点，而调整当前节点
        if len(spiltareas)<=1:
            node.center = np.array(spiltareas[0])
            node.setWidth(node.width*Node.width_scale)
            samples = node.samples
            node.samples = []
            if len(samples)>0:
                collections.foreach(samples,lambda s:node.put_sample(s))
            self.__evulate()
            return


        # 创建子节点
        self.leafs.remove(node)
        for spiltarea in spiltareas:
            n = Node(np.array(spiltarea),max(node.width*Node.width_scale,Node.min_width))
            n.parent = node
            node.childs.append(n)
            self.leafs.append(n)

        # 将属于父节点的样本分配到子节点
        for s in node.samples:
            index = np.argmin(list(map(lambda c:np.linalg.norm(c-s),spiltareas)))
            if s not in node.childs[index].samples:
                node.childs[index].put_sample(s)

        node.samples = []
        self.nodescount += len(spiltareas)
        self.__evulate()

        #self.drawdata()
    def __reconstruct(self):
        '''
        重构以减少平均能耗
        :return:
        '''
        if len(self.leafs)<=2:
            return

        # 找到宽度最小的节点
        minIndex = np.argmin(list(map(lambda l:l.width,self.leafs)))
        minNode = self.leafs[minIndex]
        minNode.parent.childs.remove(minNode)

        # 在叶子节点中找到中心点距离这个宽度最小节点最近的节点
        mergeIndex = np.argmin(list(map(lambda l:np.linalg.norm(np.array(l.center) - np.array(self.leafs[minIndex].center)),self.leafs[:minIndex]+self.leafs[minIndex+1:])))
        mergeNode = self.leafs[mergeIndex]

        # 将宽度最小节点并入到最近节点中去
        newcenter = [np.average([minNode.center,mergeNode.center])]
        newwidth = max(minNode.width,mergeNode.width)/Node.width_scale
        mergeNode.center = np.array(newcenter)
        mergeNode.setWidth(newwidth)


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
        self.__evulate()
        #self.drawdata()

    def drawdata(self,sn):
        fig = plt.figure(3)
        asp = fig.add_subplot(111)
        for node in self.leafs:
            mu, sigma = node.center, node.sigma
            sampleNo = 1000
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





if __name__ == '__main__':

    Net.max_energy_cost = 0.0
    max_cost = []
    max_sen = []
    real_cost = []
    node_cost = []
    count_node = []
    count_leaf = []

    opitmal_net = None
    for k in range(30):
        # 构造初始网络
        root = Node(center=np.array([0.5]), width=1.0)
        net = Net(root)
        leafcount = len(net.leafs)

        Net.max_energy_cost += 1.0
        max_cost.append(Net.max_energy_cost)

        for i in range(50):
            # 根据标准正态分布采样100个值
            values1 = np.random.normal(0.0, 1.0, 30)
            values2 = np.random.normal(2.0, 1.0, 30)
            values = np.append(values1,values2)
            np.random.shuffle(values)

            # 正规化到0-1之间
            values = (values - values.min()) / (values.max() - values.min())

            # 逐个输入样本
            for k,value in enumerate(values):
                net.put(np.array([value]))

            # 计算能耗
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
