# -*- coding: UTF-8 -*-


import copy
import numpy as np
from utils import strs as strs
from utils import collections as collections
from utils.properties import Range
import  brain.models as models
from brain.activation import ActivationFunction

__all__ = ['NeuralElement', 'Neuron', 'Synapse']
# 神经系统基本元素
class NeuralElement:
    def __init__(self,id,birth,modelConfiguration,coord=None):
        '''
        神经系统元素，神经元和突触的父类型
        :param id:                     int或者str     ID
        :param birth:                  int或者double  创建时间
        :param modelConfiguration:     dict          计算模型配置信息
        :param coord:                  geopy.Point   位置
        '''
        self.id = id
        self.birth = birth
        self.modelConfiguration = modelConfiguration
        self.coord = coord
        self.states = {}        # 特指活动状态，例如神经元的激活强度
        self.params = {}        # 计算过程涉及到的参数
        self.variables = []     # 变量
        self.features = {}      # 统计特征

        self._initModel()

    def _initModel(self):
        '''
        初始化计算模型：找到注册的计算模型，根据模型中的状态信息初始化神经系统元素状态，
        并将计算模型中登记的变量克隆一份在元素对象中（计算模型中本身的变量只是模版，并不在实际计算中使用，这样可以使得计算模型成为单体对象）
        :return:
        '''

        # 根据模型配置取得模型对象
        if self.modelConfiguration is None: raise RuntimeError('初始化计算模型失败(NueralElement.initModel):模型配置无效')
        if not strs.isVaild(self.modelConfiguration.modelid): raise RuntimeError('初始化计算模型失败(NueralElement.initModel):模型配置中modelId无效')
        model = models.nervousModels.find(self.modelConfiguration.modelid)
        if model is None:raise RuntimeError('初始化计算模型失败(NueralElement.initModel):找不到模型:'+self.modelConfiguration.modelid)
        # 如果模型规定了要有初始状态，则设置初始状态
        self.states = {} if model.initStates is None else copy.deepcopy(model.initStates)

        # 拷贝模型中规定的所有变量，并初始号变量的值
        self.variables = [var.clone() for var in model.variables]
        self._initVariableValue()

    def _initVariableValue(self):
        '''
        根据模型配置初始号变量的值
        :return:
        '''
        if not collections.isEmpty(self.variables):
            for var in self.variables:
                if var.type is not float:
                    continue
                if var.nameInfo.name in self.modelConfiguration:
                    var.range = Range(self.modelConfiguration[var.nameInfo.name])
                    var.value = 0 if var.range is None else var.range.sample()

    def getModel(self):
        '''
        取得计算模型
        :return:
        '''
        return models.nervousModels.find(self.modelConfiguration.modelid)

    def reset(self,resetvar=False):
        '''
        重置计算状态
        :return:
        '''
        model = self.getModel()
        self.states = {} if model.initStates is None else copy.deepcopy(model.initStates)
        if resetvar:
            self._initVariableValue()

    def getVariable(self,name):
        '''
        取得变量
        :param name: str 变量名
        :return:
        '''
        return collections.first(self.variables, lambda var: var.nameInfo.hasName(name))

    def getVariableValue(self,name,default=0.0):
        '''
        取得变量的值
        :param name:    str   变量名
        :param default: float 缺省值 0.0
        :return:
        '''
        var = collections.first(self.variables, lambda var: var.nameInfo.hasName(name))
        if var is None: return default
        return var.value


    # 取得变量或者状态的值
    def __getitem__(self, item):
        '''
        取索引
        :param item:如果索引项是变量集合中的名字，则返回该变量的值；如果是状态集合中状态的名称，则返回状态的值
        :return:
        '''
        # 在变量集合中查找
        var = collections.first(self.variables,lambda var:var.nameInfo.hasName(item))
        if var is not None:return var.value

        # 在状态集合中查找
        #if item in self.states.keys():
        if item in self.states:
            return self.states[item]

        # 在参数集合中查找
        if item in self.params.keys():
            return self.params[item]

        return super.__getitem__(item)

    # 设置变量或者状态的值
    def __setitem__(self, key, value):

        # 在变量集合中查找
        var = collections.first(self.variables, lambda var: var.nameInfo.hasName(key))
        if var is not None:
            var.value = value
            return


        # 在状态集合中查找
        self.states[key] = value
        #super(NueralElement,self).__setitem__(key,value)

#神经元
class Neuron(NeuralElement):
    default_activation_function_name = 'sigmod'
    def __init__(self,id,layer,birth,modelConfiguration,coord=None,activationFunction=None):
        '''
        神经元
        :param id:                       int或者str     ID
        :param layer:                    int           层编号
        :param birth:                    int或者double  创建时间
        :param modelConfiguration:       dict          计算模型配置信息
        :param coord:                    geopy.Point   位置
        '''
        super(Neuron, self).__init__(id,birth,modelConfiguration,coord)
        #super(id,birth,modelConfiguration,coord)
        self.layer = layer
        #if activationFunction is None and 'activationFunction' in modelConfiguration.keys() and  \
        #        'selection' in modelConfiguration.activationFunction.keys() and \
        #        modelConfiguration.activationFunction.selection is not None:
        if activationFunction is None and 'activationFunction' in modelConfiguration and \
                'selection' in modelConfiguration.activationFunction and \
                modelConfiguration.activationFunction.selection is not None:
            activationFunctionIndex = int(np.random.uniform(0, len(modelConfiguration.activationFunction.selection)))
            activationFunctionName = modelConfiguration.activationFunction.selection[activationFunctionIndex]
            activationFunction = ActivationFunction.find(activationFunctionName)
        self.activationFunction = activationFunction if activationFunction is not None else ActivationFunction.find(Neuron.default_activation_function_name)

    def _doSelectActiovationFunction(self):
        activationFunctionIndex = int(np.random.uniform(0, len(self.modelConfiguration .activationFunction.selection)))
        activationFunctionName = self.modelConfiguration .activationFunction.selection[activationFunctionIndex]
        self.activationFunction = ActivationFunction.find(activationFunctionName)
        return self.activationFunction

    def __str__(self):
        #stateStr = collections.dicttostr(self.states)
        varStr = collections.listtostr(list(map(lambda v:v.__repr__(),self.variables)))
        return 'Neuron'+str(self.id)+'[layer='+str(self.layer) \
               + (',' + varStr if strs.isVaild(varStr) else '') + ']'
                              # +(','+stateStr if strs.isVaild(stateStr) else '') \


    def __repr__(self):
        return str(self)


#突触
class Synapse(NeuralElement):
    def __init__(self,id,birth,fromId,toId,modelConfiguration,coord=None):
        '''
        连接突触
        :param id:                       int或者str     ID
        :param birth:                    int或者double  创建时间
        :param fromId:                   int或者str     输入神经元ID
        :param toId:                     int或者str     输出神经元ID
        :param modelConfiguration:       dict          计算模型配置信息
        :param coord:                    geopy.Point   位置
        '''
        super(Synapse,self).__init__(id, birth, modelConfiguration, coord)
        #super(id, birth, modelConfiguration, coord)
        self.fromId = fromId
        self.toId = toId

    def __str__(self):
        varStr = collections.listtostr(list(map(lambda v: v.__repr__(), self.variables)))
        varStr = '[' + varStr + "]" if strs.isVaild(varStr) else ''
        return 'Synapse'+str(self.id)+'('+str(self.fromId)+'->'+str(self.toId)+')' + varStr

    def __repr__(self):
        #stateStr = collections.dicttostr(self.states)
        varStr = collections.listtostr(list(map(lambda v: v.__repr__(), self.variables)))
        return 'Synapse' + str(self.id) + '[' + str(self.fromId) + '->' + str(self.toId) \
               + (',' + varStr if strs.isVaild(varStr) else '') + ']'
               #+ (',' + stateStr if strs.isVaild(stateStr) else '') \

class Module:
    def __init__(self,id,birth,fromModuleIds,toModuleIds,neuronIds):
        '''
        神经模块
        :param id:               Union(int,str)  模块id
        :param birth:            float           创建时间
        :param fromModuleIds:    list            上游模块Ids
        :param toModuleIds:      list            下游模块Ids
        :param neuronIds:        list            神经元Ids
        '''
        self.id = id
        self.birth = birth
        self.fromModuleIds = fromModuleIds
        self.toModuleIds = toModuleIds
        self.neuronIds = neuronIds
        self.attributes = {}