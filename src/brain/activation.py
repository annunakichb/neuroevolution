#/usr/bin/env python
#coding:utf-8

import math
import random
from ..utils.properties import NameInfo
from ..utils.collections import *

#region 激活函数管理

# 抽象激活函数
class ActivationFunction:
    def __init__(self,name):
        '''
        激活函数
        '''
        self.nameInfo = name if name is NameInfo else NameInfo(str(name))
        self.vars = {}  #激活函数的参量

    @classmethod
    def calculate(cls,name,inputs,modelparams):
        '''
        激活函数执行
        :param name:     str or NameInfo
        :param inputs:   输入
        :param context:  执行上下文
        :return:
        '''
        activationObj = activations.get(name)
        if activationObj is None:
            raise RuntimeError('activationFunction ' + name + " not existed")
        return activationObj.calculate(inputs,modelparams)

    @classmethod
    def find(cls,name,cataory=''):
        '''
        寻找指定名称或者类型的激活函数
        :param name:         激活函数名称
        :param cataory:      激活函数类型，如果通过name找不到，则尝试寻找指定类型的所有
        :return:
        '''
        if name is not None:
            key = first(activations.keys(),lambda key: key.hasName(name))
            if key is not None:return activations[key]

        if cataory is not None and cataory != '':
            findall(activations.values(),lambda func:func.nameInfo.cataory == cataory)

    @classmethod
    def register(cls,func):
        '''
        注册或者修改激活函数
        :param func:  激活函数
        :return:
        '''
        if func is None:return
        activations[func.nameInfo] = func

    @classmethod
    def findAll(cls):
        return activations;

#endregion


#region gaussina类激活函数

# 高斯激活函数
class Gaussian(ActivationFunction):
    nameInfo = NameInfo(name='gaussian', caption='高斯激活函数', description='exp(-(x-center)^2/sigma^2)',cataory='gaussian')
    def __init__(self,center = 0.0,sigma = 1.0):
        '''
        高斯激活函数
        :param center:  float 均值缺省值,注意这里只是缺省值，实际计算会从上下文中传入
        :param sigma:   float 方差缺省值
        '''

        self.center = center
        self.sigma = sigma
        self.nameInfo = Gaussian.nameInfo

    def getParamNames(self):
        '''
        计算所需参数名
        :return:
        '''
        return ['center','sigma']

    # 激活函数执行
    def calculate(self,inputs,params):
        '''
        计算函数
        :param inputs:      输入
        :param modelparams: 模型参数
        :return: 状态值(float)和是否激活(bool)
        '''
        if params is None : params = {}
        center = params.get('center',self.center)
        sigma = params.get('sigma',self.sigma)
        value = math.exp(-math.pow(inputs-center,2)/math.pow(sigma,2))
        return random() < value



#endregion

#region sigmod类激活函数

# S函数
class Sigmod(ActivationFunction):
    nameInfo = NameInfo(name='sigmod', caption='Sigmod激活函数', description='a/(b*(1+exp(-T*x)))',cataory='sigmods')
    def __init__(self,a = 1.0,b = 1.0,T=1.0):
        '''
        高斯激活函数
        :param a:   float 参量a的缺省值
        :param b:   float 参量b的缺省值
        :param T:   float 参量T的缺省值
        '''
        self.a = a
        self.b = b
        self.T = T
        self.nameInfo = Sigmod.nameInfo

    def getParamNames(self):
        '''
        计算所需参数名
        :return:
        '''
        return ['a','b','T']

    def calculate(self,inputs,params):
        '''
        计算函数
        :param inputs:      输入
        :param modelparams: 模型参数
        :return:状态值(float)和是否激活(bool)
        '''
        if params is None : params = {}
        a = params.get('a',self.a)
        b = params.get('b',self.b)
        T = params.get('T', self.T)
        value = a/(b*(1+math.exp(-T*inputs)))
        return value,value>0.5


#endregion

#注册的激活函数
activations = {
                  Gaussian.nameInfo:Gaussian(),
                  Sigmod.nameInfo:Sigmod()

              }