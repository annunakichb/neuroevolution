# 该实验测试关注逻辑网络控制ant移动的效果
import pybullet
import gym
import numpy as np
import utils.collections as collections

class BoxGene:
    def __init__(self,inputboxs,outputboxs,initnodecount,attributes,desctemplate):
        self.inputs = inputboxs
        self.outputs = outputboxs
        self.initnodecount = initnodecount
        self.attributes = attributes
        self.desctemplate = desctemplate
class Box:
    def __init__(self,desc=''):
        self.desc = desc
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.attributes = {}
        self.samples = []
class Node:
    firerate_activate = 0.55
    def __init__(self,center,sigma):
        self._center = center
        self._sigma = sigma
        self.states['firerate'] = 0.
        self.states['activation'] = 0
        self.states['firecount'] = 0
        self.states['firetime'] = -1.
        self.samples = []
    def __init__(self,dimension):
        self._center = [0.5]*dimension
        self.sigma = 1.0
        self.states['firerate'] = 0.
        self.states['activation'] = 0
        self.states['firecount'] = 0
        self.states['firetime'] = -1.

        self.samples = []

    def check(self,value,time):
        firerate = self._compute_firerate(value)
        if self.firerate >= Node.firerate_activate:
            self.states['firerate'] = firerate
            self.states['activation'] = 1
            self.states['firecount'] += 1
            self.states['firetime'] = time
            self.samples.append(value)
        else:
            self.states['firerate'] = firerate
            self.states['activation'] = 0

        return self.states['activation'],self.states['firerate']

    @property
    def dimension(self):
        return len(self._center)

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, c):
        '''
        设置中心点
        :param center: Union(float,list,ndarray)
        :return:
        '''
        if isinstance(c, float):
            c = [c]
        if not isinstance(c, np.ndarray):
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
    def sigma(self, value):
        '''
        设置节点的宽度，以及方差
        :param Union(float,list,np.array) value
        :return:
        '''
        if isinstance(value, float):
            if self.dimension == 1:
                value = [[value]]
            else:
                temp = value
                value = np.ones((self.dimension, self.dimension))
                for i in range(self.dimension):
                    value[i][i] = 1
                    for j in range(self.dimension):
                        if i == j: continue
                        value[i][j] = temp

        if not isinstance(value, np.ndarray):
            value = np.array(value)

        self._sigma = value

    def _compute_firerate(self, input):
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
        return value[0][0]

class Receptor(Box):
    def check(self,value,time):
        states = [node.check(value,time) for node in self.nodes]
        activation_count = collections.count(states,lambda s:s[0]==1)
        exploration = self._exploration_or_exploitation(time)
        if exploration == 0:return


    def _get_sample_count(self):
        return np.sum([len(node.samples) for node in self.nodes])

    def _do_learn(self):
        '''
        学习过程，包括节点分裂，节点合并，节点移动，方差调整
        学习的原则是满足任务需要的区分度的情况下，使得能耗最低
        该学习过程类似数据流动态聚类问题或者进化聚类模型（Evolving cluster models）
        不同的是，节点进化要考虑上面的原则，
        当一个Box中有节点频繁的激活，且对任务是有益的，节点可以进一步分裂，否则节点
        :return:
        '''
        # 判断是否要分裂节点：当节点容量达到最大，且能耗没有超过上限，节点分裂，采用带权重样本改进EM算法
        for node in self.nodes:
            if node.states['firecount'] >= Node.
    def _exploration_or_exploitation(self,t):
        '''
        # 判断启动学习过程的条件是否满足，有几种判断方法：
        # 1.样本数是否达到一定数量，要求的样本数随着时间呈指数增长
        # 2.指定的时间间隔到达，时间间隔随着时间呈指数增长
        # 在生物神经系统中，发育期间存在自适应调整的过程，神经网络自适应调整很快
        # 但是随着成年以后神经网络调整速率变慢
        # ，学习包括节点移动，方差调整，节点分裂，节点合并
        :return int: 0 表示利用，否则为探索
        '''
        #eplison = 1.0/np.sqrt(t)
        #return 1 if np.random.uniform(0,1) < eplison else 0
        alpha = 1.0
        return 1 if self._get_sample_count() >= alpha*np.exp(t) else 0


def fitness(ind,session):
    pybullet.connect(pybullet.DIRECT)
    env = gym.make("AntBulletEnv-v0")
    env.render(mode="human")
    env.reset()

def run():
