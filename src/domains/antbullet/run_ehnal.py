# -*- coding: UTF-8 -*-

# 该实验测试关注逻辑网络控制ant移动的效果
import pybullet
import gym
import numpy as np
import utils.collections as collections

class BoxGene:
    def __init__(self,id,inputboxs,outputboxs,initdistribution,type,attributes,desctemplate):
        '''
        神经元盒子基因
        :param id:               int  神经元盒子编号
        :param inputboxs:        list of int 输入盒子的编号集
        :param outputboxs:       list of int 输出盒子的编号集
        :param initdistribution: list of tuple 盒子中神经元的特征分布，
                                               例如[(0.1,1.5),(2.,1.),(10.3)],
                                               像RBF一样，每个神经元包含一个特征的的高斯分布，tuple中为高斯分布的均值和协方差矩阵
                                               基因中记录的初始分布与实际神经元细胞中的分布不同的是，只有神经元的稳定度（见后面神经元节点中的定义）
                                               大于0.6时，才会记录在基因中来。因此，初始分布中高斯分布的数量是少于神经元盒子发育之后的高斯分布数量的
        :param type              str   类型用于区分不同神经元盒子的值类型，type不可变，例如感知自身力气的神经元盒子与感知自身坐标的盒子，类型不同
                                       类型用于神经元进化的变异过程，同类型的盒子之间优先建立连接
        :param attributes:       dict  属性记录神经元盒子的基本特征，它的值是不可变的，例如一个神经元盒子如果包含接收图像的一组感光细胞，则
                                        这个盒子里的感光细胞将对图像中某个固定位置的像素的值做出响应，图像的位置坐标就称为该神经元盒子的空间属性
                                        如果盒子对某个时序序列的固定时间点做出响应，则该盒子的属性为该时间点
                                        字典的key为属性名，其中's'和't'固定代表空间属性和时间属性
        :param desctemplate:     str 完全是为了打印输出。当需要打印输出盒子的属性或状态的时候，该字符串为输出模版，输出模版都是在网络进化任务完成以后，
                                     人为的补充的
        '''
        self.id = id
        self.inputs = inputboxs
        self.outputs = outputboxs
        self.initdistribution = initdistribution
        self.attributes = attributes
        self.desctemplate = desctemplate

class Box:
    def __init__(self,gene):
        self.gene = gene
        self.nodes = []
class Sensor(Box):
    pass

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
        for node in self.nodes:pass
            #if node.states['firecount'] >= Node.
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
    # 初始神经网络的输入为28个姿态数据和8个动作数据（参见ant.py因为机器人应能够感知自身动作）
    ## 机器人位置坐标感受器
    sensor_pos = Sensor(BoxGene(10,[],[],[],'pos',{},''))
    ## 机器人移动速度感受器
    sensor_speed = Sensor(BoxGene(11,[],[],[],'speed',{},''))
    #s3s = [Sensor(BoxGene(12,[],[],[],'pos',{'partno':i+1},'')) for i in range(11)] # 身体各部位位置感受器
    ## 朝向感受器
    sensor_direction = Sensor(BoxGene(12,[],[],[],'direction',{},''))
    ## 关节感受器,其中no的三个数字最高位表示前后，中间位表示左右，最低位表示上下，上下距离最近，左右距离次近，前后距离最远
    s4_konts_posture_front_left_up = Sensor(BoxGene(13,[],[],[],'posture',{'no':'111'},''))
    s4_konts_posture_front_left_down = Sensor(BoxGene(15, [], [], [], 'posture', {'no': '112'}, ''))
    s4_konts_posture_front_right_up = Sensor(BoxGene(17, [], [], [], 'posture', {'no': '121'}, ''))
    s4_konts_posture_front_right_down = Sensor(BoxGene(19, [], [], [], 'posture', {'no': '122'}, ''))
    s4_konts_posture_back_left_up = Sensor(BoxGene(21, [], [], [], 'posture', {'no': '211'}, ''))
    s4_konts_posture_back_left_down = Sensor(BoxGene(23, [], [], [], 'posture', {'no': '212'}, ''))
    s4_konts_posture_back_right_up = Sensor(BoxGene(25, [], [], [], 'posture', {'no': '221'}, ''))
    s4_konts_posture_back_right_down = Sensor(BoxGene(27, [], [], [], 'posture', {'no': '222'}, ''))
    # 关节动作感受器
    s4_konts_action_front_left_up = Sensor(BoxGene(14, [], [], [], 'action', {'no': '111'}, ''))
    s4_konts_action_front_left_down = Sensor(BoxGene(16, [], [], [], 'action', {'no': '112'}, ''))
    s4_konts_action_front_right_up = Sensor(BoxGene(18, [], [], [], 'action', {'no': '121'}, ''))
    s4_konts_action_front_right_down = Sensor(BoxGene(20, [], [], [], 'action', {'no': '122'}, ''))
    s4_konts_action_back_left_up = Sensor(BoxGene(22, [], [], [], 'action', {'no': '211'}, ''))
    s4_konts_action_back_left_down = Sensor(BoxGene(24, [], [], [], 'action', {'no': '212'}, ''))
    s4_konts_action_back_right_up = Sensor(BoxGene(26, [], [], [], 'action', {'no': '221'}, ''))
    s4_konts_action_back_right_down = Sensor(BoxGene(28, [], [], [], 'action', {'no': '222'}, ''))
    # 脚丫子沾地感受器
    s5_foot_front_left = Sensor(BoxGene(29,[],[],[],'state',{'no':11},''))
    s5_foot_front_right = Sensor(BoxGene(30, [], [], [], 'state', {'no': 12}, ''))
    s5_foot_back_left = Sensor(BoxGene(31, [], [], [], 'state', {'no': 21}, ''))
    s5_foot_back_right = Sensor(BoxGene(32, [], [], [], 'state', {'no': 22}, ''))

    # 初始神经网络的输出为8个动作数据，每个关节感受器连接到对应的动作效应器
    r_knots_front_left_up = Receptor(BoxGene(29, [13,14], [], [], 'action', {'no': '111'}, ''))
    r_knots_front_left_down = Receptor(BoxGene(30, [15, 16], [], [], 'action', {'no': '111'}, ''))
    r_knots_front_right_up = Receptor(BoxGene(31, [17,18], [], [], 'action', {'no': '121'}, ''))
    r_knots_front_right_down = Sensor(BoxGene(32, [19,20], [], [], 'action', {'no': '122'}, ''))
    r_knots_back_left_up = Sensor(BoxGene(33, [21,22], [], [], 'action', {'no': '211'}, ''))
    r_knots_back_left_down = Sensor(BoxGene(34, [23,24], [], [], 'action', {'no': '212'}, ''))
    r_knots_back_right_up = Sensor(BoxGene(35, [25,26], [], [], 'action', {'no': '221'}, ''))
    r_knots_back_right_down = Sensor(BoxGene(36, [27,28], [], [], 'action', {'no': '222'}, ''))