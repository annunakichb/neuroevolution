from ..utils import strs as strs
from ..utils import collections as collections
from ..utils.properties import *
from .activation import ActivationFunction
from utils.properties import Registry



#region 普通神经元计算模型

#普通输入模型
class CommonInputNeurnModel:
    nameInfo = NameInfo('input', cataory='common')
    def __init__(self,**configuration):
        '''
        普通输入模型
        :param configuration: 模型缺省配置
        '''
        self.nameInfo = CommonInputNeurnModel.nameInfo
        self.configuration = configuration if not collections.isEmpty(configuration) else {}

    def execute(self,neuron,net,**context):
        '''
        执行：对于没有输入的神经元，记录值为0，状态为未激活，返回0
             对于输入不完全的神经元，不做记录，返回None
             对于输入完全的神经元，记录计算值和激活状态，返回值
        :param neuron:  计算的神经元
        :param net:     网络
        :param context: 上下文,对于输入神经元，应包含value
        :return:
        '''
        if context is None or not context.keys().__contains__('value'):raise RuntimeError('神经元计算失败(CommonInputNeurnModel):没有有效的输入')
        value = context['value']
        neuron.states['value'] = value
        neuron.states['activation'] = context['value'] != 0
        return value,neuron.states['activation']

    def getVariables(self):
        return []
    def getInitState(self):
        return {}


# 基本神经元计算模型（权重和加激活函数）
class CommonHiddenNeuronModel:
    nameInfo = NameInfo('hidden', cataory='common')
    def __init__(self,**configuration):
        self.nameInfo = CommonHiddenNeuronModel.nameInfo
        self.configuration = configuration

    def execute(self,neuron,net,**context):
        '''
        执行：对于没有输入的神经元，记录值为0，状态为未激活，返回0
             对于输入不完全的神经元，不做记录，返回None
             对于输入完全的神经元，记录计算值和激活状态，返回值
        :param neuron:  计算的神经元
        :param net:     网络
        :param context: 上下文（保留）
        :return:
        '''

        # 取得待计算突触的输入突触
        synapses = net.getInputs(neuron.id)
        if synapses is None or len(synapses)<=0:return

        # 检查突触是否都有值
        if not collections.all(synapses,lambda s:'value' in s.states.keys()):
            return None

        # 取得突触所有输入值并求和(权重已经计算)
        inputs = map(lambda s:s.states['value'],synapses)
        sum = sum(inputs)

        # 加偏置
        sum += neuron['bias']

        # 取得激活函数
        activationFunctionConfig = self.configuration.get('activationFunction')
        activationFunction = ActivationFunction.find(activationFunctionConfig.name)
        if activationFunction is None:raise RuntimeError('神经元计算失败(CommonNeuronModel),激活函数无效:'+activationFunctionConfig.name)

        # 组合出激活函数参数(参数可能是网络)
        activationParamNames = activationFunction.getParamNames()
        activationParams = {}
        for name in activationParamNames:
            if name in activationFunctionConfig:activationParams[name] = activationFunctionConfig[name]
            v = neuron[name]
            if v is not None:activationParams[name] = v

        # 用输入和计算激活函数
        value,activation = activationFunction.calculate(sum,activationParams) #?这里有问题，激活函数的参数无法传入

        # 记录状态
        neuron.states['value'] = value
        neuron.states['activation'] = activation

        return value

#endregion

#region 普通突触计算模型

class CommonSynapseModel:
    nameInfo = NameInfo('synapse', cataory='common')
    def __init__(self,**configuration):
        self.nameInfo = CommonSynapseModel.nameInfo
        self.configuration = configuration

    def execute(self,synapse,net,**context):
        '''
        执行：对于输入没有到达的突触，记录值为0，返回0
             对于输入完全的突触，记录计算值，返回值
        :param neuron:  计算的神经元
        :param net:     网络
        :param time:    计算时间
        :param context: 上下文（保留）
        :return:
        '''
        neuron = net.getNeuron(id=synapse.fromId)
        if neuron is None:raise RuntimeError('突触计算失败：找不到接入神经元:'+synapse.fromId)
        if 'value' not in neuron.states.keys():return None
        w = synapse['weight']
        value = w * neuron['value']
        synapse['vlaue'] = value
        return value

#endregion


# 神经计算模型管理
nervousModels = Registry()
nervousModels.register(CommonInputNeurnModel())
nervousModels.register(CommonHiddenNeuronModel())
nervousModels.register(CommonSynapseModel())