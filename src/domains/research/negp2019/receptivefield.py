import numpy as np
from ne.negp.elements import FeatureNeuronBox


class FeatureBox:
    ATTENTION = ['average', 'sigma', 'joint', 'margin']

    def __init__(self, id, birth, name,coord):
        self.points = []
        self.state['main'] = None
        self.state['follows'] = []
        self.name = name
        self.inputs = []
        self.outputs = []
        self.properties = {'coord':coord}
    def put(self,value,birth):
        if len(self.points)<=0:
            p = FeaturePoint(1,birth,self)
            p.state['avtivation'] = 1.
            p.state['feature'] = {'value':value,'sigma':np.eye(len(value))}
            p.state['livenss'] = np.exp(20/(-1))
            p.state['count'] = 1
            self.state['main'] = p
            self.state['follows'] = []
            return
        # 寻找最近的points




class FeaturePoint:
    def __init__(self, id, birth, box):
        self.state['activation'] = False
        self.state['liveness'] = 0.
        self.state['feature'] = None
        self.state['time'] = 0
        self.state['count'] = 0
        self.box = box

    def prob(self, value):
        '''
        按照高斯分布计算值
        :param input:  输入样本向量
        :return: 高斯分布计算值
        '''
        center = self.state['feature']['value']
        sigma = self.state['feature']['sigma']

        dim = np.shape(np.cov(value.T))[0]
        X = value.T - center.reshape(-1, 1)

        Z = np.array(X.T).dot(np.linalg.inv(sigma)).dot(X) * -.5
        #A = 1 / (np.linalg.det(sigma) * (2 * np.pi) ** dim) ** .5
        #return np.diag(A * np.exp(Z))
        return np.diag(np.exp(Z))


class Link:
    def __init__(self, from_, to_):
        self._from = from_
        self._to = to_


class FeatureNet:
    def __init__(self, id,inputBoxes, outputBoxes):
        self.id = id
        self.inputBoxs = inputBoxes
        self.outputBosx = outputBoxes

def receptivefield():
    # 图像尺寸是27*27
    size = [27, 27]

    # 种群尺寸
    pop_size = 100
    pops = []
    for i in pop_size:
        # 创建输入特征
        inputs = [FeatureBox(row * (col + 1), 0, str(row) + "-" + str(col)) for col in range(size[1]) for row in
                  range[size[0]]]
        # 创建输出特征
        output = [FeatureBox(1000, 0, 'edge')]
        # 创建网络
        net = FeatureNet(i,inputs,output)
        pops.append(net)

    for generation in range(10000):
        # 计算适应度
        for ind in pops:


            # 选择
            # 交叉
            # 变异