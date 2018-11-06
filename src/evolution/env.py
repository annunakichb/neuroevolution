

__all__ = ['EvaluationValue','Evaluator']
#个体评估值
class EvaluationValue:
    def __init__(self,maxsize=3):
        '''
        个体评估值
        :param maxsize:  int 存储历史评估值的最大量
        '''
        self.maxsize = maxsize
        self.values = []

    def __getitem__(self, item):
        '''
        取得历史值
        :param item:
        :return:
        '''
        if len(self.values)<=0:return 0
        return self.values[item]
    def __setitem__(self, key, value):
        '''
        添加新值，值一旦添加后，不能再修改
        :param key:   无论填写什么，都是添加一个新值
        :param value:
        :return:
        '''
        self.append(value)

    def getValue(self):
        '''
        取得最新值
        :return:
        '''
        if len(self.values)<=0:return 0
        return self.values[-1]

    def getValues(self):
        '''
        取得所有历史值
        :return:
        '''
        return self.values

    def __float__(self):
        '''
        取得最新值
        :return:
        '''
        return self.getValue()


    def append(self,value):
        '''
        添加最新值
        :param value:
        :return:
        '''
        self.values.append(value)
        if len(self.values)>self.maxsize:
            self.values = self.values[1:]

# 评估器
class Evaluator:
    def __init__(self,key,evaulateFunctions):
        '''
        评估器
        :param key:                 str 评估器名称，如fitness
        :param evaulateFunctions:   dict key是函数，value是权重float
        '''
        self.key = key
        if evaulateFunctions is None:raise  RuntimeError('创建评估器对象失败：评估函数参数无效')
        if evaulateFunctions is function:
            self.evaulateFunctions = {evaulateFunctions:1.0}
        elif evaulateFunctions is dict:
            self.evaulateFunctions = evaulateFunctions

    def getFuncsAndWeights(self):
        '''
        取得函数和权重
        :return: dict key是函数，value是权重float
        '''
        return self.evaulateFunctions

    def getFuncs(self):
        '''
        取得所有函数
        :return:
        '''
        return self.evaulateFunctions.keys()

    def calacute(self,ind,session):
        '''
        对个体进行评估计算
        :param ind:        个体
        :param session:
        :return:           float 评估值
        '''
        result = 0.0
        for func,weight in self.evaulateFunctions:
            result += weight * func(ind,session)
        return result
