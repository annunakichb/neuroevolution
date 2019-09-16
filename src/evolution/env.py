# -*- coding: UTF-8 -*-

from functools import reduce
from utils.collections import ExtendJsonEncoder
import  utils.strs as strs
from operator import mul, truediv

__all__ = ['EvaluationValue','Evaluator','EvaluationValues']
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

    def __str__(self):
        return str(self.getValue())

    @property
    def value(self):
        return self.getValue()

    @value.setter
    def value(self,v):
        self.append(v)

    @property
    def Value(self):
        return self.getValue()

    @Value.setter
    def Value(self, v):
        self.append(v)

    def append(self,value):
        '''
        添加最新值
        :param value:
        :return:
        '''
        self.values.append(value)
        if len(self.values)>self.maxsize:
            self.values = self.values[1:]
        return self

class EvaluationValues(object):
    """主要用于多目标优化同时获取多个适应度值
    这段代码来自DEAP：https://github.com/DEAP/deap/blob/master/deap/base.py
    """
    # 每种适应度的权重
    weights = None

    """Contains the weighted values of the fitness, the multiplication with the
    weights is made when the values are set via the property :attr:`values`.
    Multiplication is made on setting of the values for efficiency.
    Generally it is unnecessary to manipulate wvalues as it is an internal
    attribute of the fitness used in the comparison operators.
    """

    def __init__(self, values=()):
        if self.weights is None:raise TypeError('创建EvaluationValue之前需要设置权重' % (self.__class__))
        self.wvalues = ()
        if len(values) > 0:
            self.setValues(values)

    def getValues(self):
        '''
        得到权重后的评估值
        :return:
        '''
        return tuple(map(truediv, self.wvalues, self.weights))

    def setValues(self, values):
        self.wvalues = tuple(map(mul, values, self.weights))

    def delValues(self):
        self.wvalues = ()

    values = property(getValues, setValues, delValues,
                      ("Fitness values. Use directly ``individual.fitness.values = values`` "
                       "in order to set the fitness and ``del individual.fitness.values`` "
                       "in order to clear (invalidate) the fitness. The (unweighted) fitness "
                       "can be directly accessed via ``individual.fitness.values``."))

    def dominates(self, other, obj=slice(None)):
        """Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.
        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        not_equal = False
        for self_wvalue, other_wvalue in zip(self.wvalues[obj], other.wvalues[obj]):
            if self_wvalue > other_wvalue:
                not_equal = True
            elif self_wvalue < other_wvalue:
                return False
        return not_equal

    @property
    def valid(self):
        """Assess if a fitness is valid or not."""
        return len(self.wvalues) != 0

    def __hash__(self):
        return hash(self.wvalues)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __le__(self, other):
        return self.wvalues <= other.wvalues

    def __lt__(self, other):
        return self.wvalues < other.wvalues

    def __eq__(self, other):
        return self.wvalues == other.wvalues

    def __ne__(self, other):
        return not self.__eq__(other)

    def __deepcopy__(self, memo):
        """Replace the basic deepcopy function with a faster one.
        It assumes that the elements in the :attr:`values` tuple are
        immutable and the fitness does not contain any other object
        than :attr:`values` and :attr:`weights`.
        """
        copy_ = self.__class__()
        copy_.wvalues = self.wvalues
        return copy_

    def __str__(self):
        """Return the values of the Fitness object."""
        return str(self.values if self.valid else tuple())

    def __repr__(self):
        """Return the Python code to build a copy of the object."""
        return "%s.%s(%r)" % (self.__module__, self.__class__.__name__,
                              self.values if self.valid else tuple())
# 评估器
class Evaluator:
    def __init__(self,key,evaulateFunctions):
        '''
        评估器
        :param key:                 str 评估器名称，如fitness
        :param evaulateFunctions:   list,每项是一个长度为2的元组(评估函数，权重）
        '''
        self.key = key
        if evaulateFunctions is None:raise  RuntimeError('创建评估器对象失败：评估函数参数无效')
        if isinstance(evaulateFunctions,list):
            self.evaulateFunctions = evaulateFunctions
        elif callable(evaulateFunctions):
            self.evaulateFunctions = [(evaulateFunctions,1.0)]
        else:
            self.evaulateFunctions = None

    def __str__(self):
        return "Evaluator('" + self.key + "',[" + reduce(lambda x,y:x+","+y,map(lambda x:"("+x[0].__name__+","+str(x[1])+")",self.evaulateFunctions)) + "])"


    def getFuncsAndWeights(self):
        '''
        取得函数和权重
        :return: [(key,value),(key,value)] key是函数，value是权重float
        '''
        return self.evaulateFunctions

    def getFuncs(self):
        '''
        取得所有函数
        :return:
        '''
        return list(map(lambda x:x[0],self.evaulateFunctions))

    def calacute(self,ind,session):
        '''
        对个体进行评估计算
        :param ind:        个体
        :param session:
        :return:           float 评估值
        '''
        result = 0.0
        for (func,weight) in self.evaulateFunctions:
            result += weight * func(ind,session)
        return result

ExtendJsonEncoder.autostrTypes.append(Evaluator)