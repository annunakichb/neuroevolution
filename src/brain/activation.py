#/usr/bin/env python
#coding:utf-8

import math
import random
import  numpy as np
from utils.properties import NameInfo
from utils.properties import Registry
from utils.collections import *

__all__ = ['ActivationFunction','Gaussian','Sigmod','activations']
#region 激活函数管理

# 抽象激活函数
class ActivationFunction:

    def __init__(self,name):
        '''
        激活函数
        '''
        self.nameInfo = name if isinstance(name,NameInfo) else NameInfo(str(name))
        self.vars = {}  #激活函数的参量

    @classmethod
    def calculate(cls,name,inputs,modelparams):
        '''
        激活函数执行
        :param name:     str or NameInfo
        :param inputs:   输入
        :param modelparams:  dict 执行上下文
        :return:
        '''
        activationObj = activations.get(name)
        if activationObj is None:
            raise RuntimeError('activationFunction ' + name + " not existed")
        return activationObj.calculate(inputs,modelparams)

    @classmethod
    def find(cls,name):
        return activations.find(name)

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
        return value,value > 0.5


class BipolarGaussian(ActivationFunction):
    nameInfo = NameInfo(name='bipolargaussian', caption='BipolarGaussian激活函数', description='2*e^(-(input*2.5)^2) - 1)',cataory='gaussian')
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
        value = 2 * math.exp(-(inputs * 2.5) * (inputs * 2.5)) - 1
        return value,value > 0

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

class BipolarSigmoid(ActivationFunction):
    nameInfo = NameInfo(name='bipolarsigmoid', caption='BipolarSigmoid激活函数', description='2.0/(1.0 + exp(-4.9*x)) - 1.0',cataory='sigmods')
    def __init__(self,a = 2.0,b = 1.0,T=1.0):
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
        value = 2.0/(1.0 + math.exp(-4.9*inputs)) - 1.0
        return value,value>0.5
#endregion




#region 线性激活函数

# S函数
class Linear(ActivationFunction):
    nameInfo = NameInfo(name='linear', caption='线性激活函数', description='y = max(-1,min(1,x))',cataory='linear')
    def __init__(self):
        '''
        线性激活函数
        '''
        self.nameInfo = Linear.nameInfo

    def getParamNames(self):
        '''
        计算所需参数名
        :return:
        '''
        return []

    def calculate(self,inputs,params):
        '''
        计算函数
        :param inputs:      输入
        :param modelparams: 模型参数
        :return:状态值(float)和是否激活(bool)
        '''
        value = max(-1.,min(1.,inputs))
        return value,value > 0


#endregion

# 三角函数
class Sin(ActivationFunction):
    nameInfo = NameInfo(name='sin', caption='线性激活函数', description='k*sin(omega*x+f)',cataory='triangle')
    def __init__(self,k=1.,omega=1.,f=0.):
        '''
        线性激活函数
        '''
        self.k = k
        self.omega = omega
        self.f = f
        self.nameInfo = Sin.nameInfo

    def getParamNames(self):
        '''
        计算所需参数名
        :return:
        '''
        return [self.k,self.omega,self.f]

    def calculate(self,inputs,params):
        '''
        计算函数
        :param inputs:      输入
        :param modelparams: 模型参数
        :return:状态值(float)和是否激活(bool)
        '''
        value = self.k * np.sin(self.omega * inputs + self.f)
        return value,value > 0

class Sine(ActivationFunction):
    nameInfo = NameInfo(name='sine', caption='Sine激活函数', description='sin(2*x)',cataory='triangle')
    def __init__(self,k=1.,omega=2.,f=0.):
        '''
        线性激活函数
        '''
        self.k = k
        self.omega = omega
        self.f = f
        self.nameInfo = Sin.nameInfo

    def getParamNames(self):
        '''
        计算所需参数名
        :return:
        '''
        return [self.k,self.omega,self.f]

    def calculate(self,inputs,params):
        '''
        计算函数
        :param inputs:      输入
        :param modelparams: 模型参数
        :return:状态值(float)和是否激活(bool)
        '''
        value = self.k * np.sin(self.omega * inputs + self.f)
        return value,value > 0
#endregion

#注册的激活函数
activations = Registry()
activations.register(Gaussian(),Gaussian.nameInfo.name)
activations.register(BipolarGaussian(),BipolarGaussian.nameInfo.name)
activations.register(Sigmod(),Sigmod.nameInfo.name)
activations.register(BipolarSigmoid(),BipolarSigmoid.nameInfo.name)
activations.register(Linear(),Linear.nameInfo.name)
activations.register(Sin(),Sin.nameInfo.name)
activations.register(Sine(),Sine.nameInfo.name)