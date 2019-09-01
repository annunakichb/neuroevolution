import numpy as np
from ne.negp.elements import FeatureNeuronBox


class FeatureBox:
    ATTENTION = ['average','sigma','joint','margin']
    def __init__(self,id,birth,name):
        self.points = []
        self.state['main'] = None
        self.state['follows'] = []
        self.name = name
        self.inputs = []
        self.outputs = []

class FeaturePoint:
    def __init__(self,id,birth,box):
        self.state['activation'] = False
        self.state['liveness'] = 0.
        self.state['features'] = None
        self.state['time'] = 0
        self.box = box




class Link:
    def __init__(self,from_,to_):
        self._from = from_
        self._to = to_

class FeatureNet:
    def __init__(self,inputBoxes,outputBoxes):
        self.inputBoxs = inputBoxes
        self.outputBosx = outputBoxes


class Food:
    def __init__(self, point, birth, duration):
        self.point = point
        self.birth = birth
        self.duration = duration

class NodularLife:
    def __init__(self,id,birth,type,head,nodes,tail,net):
        self.id = id
        self.birth = birth
        self.type = type
        self.body = {}
        self.body['head'] = head
        self.body['nodes'] = nodes
        self.body['tail'] = tail
        self.net = net
class LifeA(NodularLife):
    def __init__(self,id,birth,type,head,nodes,tail,net):
        super(LifeA, self).__init__(id,birth,type,head,nodes,tail,net)

class Predator(NodularLife):
    def __init__(self,id,birth,type,head,nodes,tail,net):
        super(LifeA, self).__init__(id,birth,type,head,nodes,tail,net)

class World:
    def __init__(self,size):
        self.size = size
    def randomposition(self):
        return [np.random.uniform(10, self.size[0]-10),np.random.uniform(10, self.size[1]-10)]
    def randomdireciotn(self):
        # 随机选择一个方向，-pi到pi之间
        return np.random.uniform(-np.pi,np.pi)

def receptivefield():
    # 图像尺寸是27*27
    size = [27,27]

    # 种群尺寸
    pop_size = 100
    for i in pop_size:
        # 创建输入特征
        inputs = [[FeatureBox(row*(col+1),0,str(row)+"-"+str(col)) for col in range(size[1])] for row in range[size[0]]]
        # 创建输出特征
        output = [FeatureBox(1000, 0, 'edge')]
        # 创建网络

    inputs = []
    for i in range(size[0]):
        row = []
        for j in range[size[1]]:
            box = FeatureBox(i*(j+1),0,str(i*(j+1)))
            row.append(box)
        inputs.append(row)







context = {}
context['world'] = {}
context['unitengery'] = 0.1                       # 单位（每厘米）能量
context['unitconsumption'] = 0.01                # 单位（每厘米）能耗
context['predator']={}                            #  捕食者参数
context['predator']['initsize'] = 50            #  捕食者初始种群数量
context['predator']['initknots'] = 2            #  捕食者初始活动关节数
context['predator']['bodysize'] = 1.            #  捕食者每节身体长度
context['predator']['visualradius']=50         #  捕食者视觉半径,cm
context['predator']['receptivefield']=0.5     # 捕食者感受野大小 cm
if __name__ == '__main__':
    # The scene is a square two-dimensional world with one kilometer width and height.
    world = World([1000.0,1000.0])

    # The predator's body is divided into three parts by two joints. The average length of each part is 10 cm, with a standard Gauss noise.
    # The size of population of predators is 50
    id,birth,type= 1,0,'predator'
    for i in range(context['predator']['initsize']):
        node1 = world.randomposition()
        d = world.randomdireciotn()
        node2 = [node1[0]+node1[0]*np.cos(context['predator']['bodysize']),node1[1]+node1[1]*np.sin(context['predator']['bodysize'])]
        tail = [node1[0]+node1[0]*np.cos(context['predator']['bodysize']*2),node1[1]+node1[1]*np.sin(context['predator']['bodysize']*2)]
        head = [node1[0]-node1[0]*np.cos(context['predator']['bodysize']),node1[1]-node1[1]*np.sin(context['predator']['bodysize'])]

        boxInput