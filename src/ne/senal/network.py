
from brain.networks import NeuralNetwork
from ne.senal.box import Box

class SENetwork(NeuralNetwork):
    def __init__(self):
        self.boxes = []
    def putBox(self,box):
        self.boxes.append(box)


    def findBox(self,type=None,group=None,expressType=None):
        '''
        查找特定类型和分组条件的盒子
        :param type:
        :param group:
        :return:
        '''
        return [b for b in self.boxes if
                (group is not None and b.gene.group.startWith(group)) and
                (type is not None and b.gene.type == type) and
                (expressType is not None and b.gene.expression.startWith(expressType))]



    def findTBox(self,cause,effect):
        return [b for b in self.boxes if
                (b.getExpressionOperation() == 'T') and
                (cause is not None and b.isInExpressionParam(cause,parts='cause')) and
                (effect if not None and b.isInExpressionParam(effect,parts='effect'))]

    def findABox(self,params):
        return [b for b in self.boxes if
                (b.getExpressionOperation() == 'T') and
                (params if not None and b.isInExpressionParam(params))]

    def findEnvSensorBox(self):
        '''
        查找负责环境感知的所有盒子
        :return:
        '''
        return self.findBox('sensor','env.')