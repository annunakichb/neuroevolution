

import copy
from ..utils import strs as strs
from ..utils import collections as collections
import  brain.models as models

__all__ = ['NueralElement','Neuron','Synapse']
# 神经系统基本元素
class NueralElement:
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
        self.states = {}
        self.variables = []

        self.initModel()

    def initModel(self):
        '''
        初始化计算模型：找到注册的计算模型，根据模型中的状态信息初始化神经系统元素状态，
        并将计算模型中登记的变量克隆一份在元素对象中（计算模型中本身的变量只是模版，并不在实际计算中使用，这样可以使得计算模型成为单体对象）
        :return:
        '''
        if self.modelConfiguration is None: raise RuntimeError('初始化计算模型失败(NueralElement.initModel):模型配置无效')
        if not strs.isVaild(self.modelConfiguration.modelId): raise RuntimeError('初始化计算模型失败(NueralElement.initModel):模型配置中modelId无效')
        model = models.nervousModels.find(self.modelConfiguration.modelId)
        if model is None:raise RuntimeError('初始化计算模型失败(NueralElement.initModel):找不到模型:'+self.modelConfiguration.modelId)
        self.states = {} if model.initStates is None else model.initStates
        self.variables = copy.deepcopy(model.variables)

    def getModel(self):
        '''
        取得计算模型
        :return:
        '''
        return models.nervousModels.find(self.modelConfiguration.modelId)

    def reset(self):
        '''
        重置计算状态
        :return:
        '''
        self.states = {}

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
        if item in self.states.keys():
            return self.states[item]

        return super.__getitem__(item)

    # 设置变量或者状态的值
    def __setitem__(self, key, value):

        # 在变量集合中查找
        var = collections.first(self.variables, lambda var: var.nameInfo.hasName(item))
        if var is not None:
            var.value = value
            return

        # 在状态集合中查找
        if key in self.states.keys():
            self.states[key] = value
            return

        super.__setitem__(key,value)

#神经元
class Neuron(NueralElement):
    def __init__(self,id,layer,birth,modelConfiguration,coord=None):
        '''
        神经元
        :param id:                       int或者str     ID
        :param layer:                    int           层编号
        :param birth:                    int或者double  创建时间
        :param modelConfiguration:       dict          计算模型配置信息
        :param coord:                    geopy.Point   位置
        '''
        super(id,birth,modelConfiguration,coord)
        self.layer = layer

    def __str__(self):
        stateStr = collections.dicttostr(self.states)
        varStr = collections.listtostr(map(lambda v:v.__repr(),self.variables))
        return 'Neuron'+str(id)+'[layer='+str(self.layer)\
                               +(','+stateStr if strs.isVaild(stateStr) else '') \
                               + (','+varStr if strs.isVaild(varStr) else '') + ']'

    def __repr__(self):
        return str(self)


#突触
class Synapse(NueralElement):
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
        super(id, birth, modelConfiguration, coord)
        self.fromId = fromId
        self.toId = toId

    def __str__(self):
        return 'Synapse'+self.id+'['+self.fromId+'->'+self.toId+']'

    def __repr__(self):
        stateStr = collections.dicttostr(self.states)
        varStr = collections.listtostr(map(lambda v: v.__repr(), self.variables))
        return 'Synapse' + self.id + '[' + self.fromId + '->' + self.toId \
               + (',' + stateStr if strs.isVaild(stateStr) else '') \
               + (',' + varStr if strs.isVaild(varStr) else '') + ']'

