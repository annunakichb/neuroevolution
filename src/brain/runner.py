# -*- coding: UTF-8 -*-

import numpy as np
from enum import Enum
from utils import collections
from utils.properties import Registry
from utils.collections import ExtendJsonEncoder

import operator
import copy
import sys

import brain.models as models

__all__ = ['EndCondition','NeuralNetworkTask','SimpleNeuralNetworkRunner','runners']

cur_generation = 1

# 训练终止条件类型
class EndCondition(Enum):
    MAXEPOCH = 1,
    MINDICATOR = 2,
    MAXDICATOR = 4,
    MINERROR = 8,
    MAXERRORUNCHANGED = 16

# 神经网络训练任务
class NeuralNetworkTask:

    #region 测试指标名称
    #测试指标：测试样本数
    INDICATOR_TEST_COUNT = 'testCount'
    #测试指标：测试正确数
    INDICATOR_CORRECT_COUNT = 'correctCount'
    # 测试指标：准确率
    INDICATOR_ACCURACY = 'accuracy'
    # 测试指标：MAE
    INDICATOR_MEAN_ABSOLUTE_ERROR = 'MAE'
    # 测试指标：MSE
    INDICATOR_MEAN_SQUARED_ERROR = 'MSE'
    #endregion

    #region 初始化

    def __init__(self,train_x=[],train_y=[],test_x=[],test_y=[],**kwargs):
        '''
        神经网训练任务
        :param train_x:   训练样本
        :param train_y:   训练标签
        :param test_x:    测试样本
        :param test_y:    测试标签
        :param kwargs     配置参数:
                           'deviation' : 0.00001 float or list of float 度量准确性的允许误差
                           'multilabel': False   bool                   是否是多标签
        '''
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.test_result = [[]]*len(test_y)
        self.kwargs = {} if kwargs is None else kwargs
        #if 'deviation' not in self.kwargs.keys():self.kwargs['deviation'] = 0.00001
        #if 'multilabel' not in self.kwargs.keys():self.kwargs['multilabel'] = False
        if 'deviation' not in self.kwargs:self.kwargs['deviation'] = 0.00001
        if 'multilabel' not in self.kwargs:self.kwargs['multilabel'] = False




# 简单前馈神经网络运行期
class SimpleNeuralNetworkRunner:
    def __init__(self):
        '''
        简单前馈神经网络运行器（要求神经网络无自连接，无循环，全部都是前馈连接）
        '''
        pass

    def doLearn(self,net,task):
        '''
        未实现
        :param net:
        :param task:
        :return:
        '''
        pass


    def doTest(self,net,task):
        '''
        执行测试
        :param net:  测试网络
        :param task: 测试任务
        :return: None
        '''
        # 取得输入
        inputNeurons = net.getInputNeurons()
        #对每一个输入样本
        for index,value in enumerate(task.test_x):
            outputs = self.activate(net,value)
            net.setTestResult(index,outputs)

    def activate(self,net,inputs):
        '''
        激活网络
        :param net:  测试网络
        :param task: 测试任务
        :return: outputs
        '''
        # 取得输入
        inputNeurons = net.getInputNeurons()

        # 重置神经元和突触状态
        collections.foreach(net.getNeurons(),lambda n:n.reset())
        collections.foreach(net.getSynapses(),lambda s:s.reset())

        # 设置输入
        for d,v in enumerate(inputs):
            if d >= len(inputNeurons):break
            model = models.nervousModels.find(inputNeurons[d].modelConfiguration.modelid)
            model.execute(inputNeurons[d],net,value=v)

            s = net.getOutputSynapse(inputNeurons[d].id)
            if collections.isEmpty(s):continue

            collections.foreach(s,lambda x:x.getModel().execute(x,net))

        # 反复执行
        ns = net.getNeurons()
        neuronCount = net.getNeuronCount()
        iterCount = 0
        outputNeurons = net.getOutputNeurons()
        #while not collections.all(outputNeurons,lambda n:'value' in n.states.keys()) and iterCount<=neuronCount:
        while not collections.all(outputNeurons, lambda n: 'value' in n.states) and iterCount <= neuronCount:
            iterCount += 1
            #uncomputeNeurons = collections.findall(ns,lambda n:'value' not in n.states.keys())
            uncomputeNeurons = collections.findall(ns, lambda n: 'value' not in n.states)
            if collections.isEmpty(uncomputeNeurons):break
            for n in uncomputeNeurons:
                model = n.getModel()
                synapses = net.getInputSynapse(n.id)
                if collections.isEmpty(synapses):continue
                #if not collections.all(synapses,lambda s:'value' in s.states.keys()):continue
                if not collections.all(synapses, lambda s: 'value' in s.states): continue
                model.execute(n,net)

                synapses = net.getOutputSynapse(n.id)
                if collections.isEmpty(synapses):continue
                collections.foreach(synapses,lambda s:s.getModel().execute(s,net))

        # 将没结果的输出神经元的值设置为0
        #outputNeuronsWithNoResult = collections.findall(outputNeurons,lambda n:'value' not in n.states.keys())
        outputNeuronsWithNoResult = collections.findall(outputNeurons, lambda n: 'value' not in n.states)
        if not collections.isEmpty(outputNeuronsWithNoResult):
            collections.foreach(outputNeuronsWithNoResult,lambda n:exec("n['value']=0"))
        # 取得结果
        outputs = list(map(lambda n:n['value'],outputNeurons))
        if len(outputs) == 1:outputs = outputs[0]
        return outputs


# 基于tf的神经网络运行器
class TensorflowNeuralNetworkRunner:
    pass

# 简单前馈神经网络运行期
class GeneralNeuralNetworkRunner:
    def __init__(self):
        pass
    def activate(self,net,inputs,activeno,**kwargs):
        '''
        激活网络
        :param net:       NeuralNetwork  网络
        :param inputs:    Union(float,list,tuple,ndarray...)  输入
        :param activeno:  int 激活编号
        :param kwargs:    dict 激活参数
        :return:
        '''

        # 取得运行方式，缺省是按照时间运行
        #runMode = 'time' if kwargs is None or 'runMode' not in kwargs.keys() else int(kwargs['runMode'])
        runMode = 'time' if kwargs is None or 'runMode' not in kwargs else int(kwargs['runMode'])
        if runMode == 'time':
            self._activateByTime(net,inputs,activeno,kwargs)
        else:
            self._activateByEvent(net,inputs,activeno,kwargs)


    def _activateByEvent(self,net,inputs,activeno,**kwargs):
        '''
        按照时间激活
        :param net:      NeuralNetwork  网络
        :param inputs:   Union(float,list,tuple,ndarray...)  输入
        :param activeno: int 激活编号
        :param kwargs:   dict 激活参数
        :return: （网络输出,结束时间)，网络输出是一个list，其中每项是一个元组(value,activation,firerate,time)
        '''
        # 参数:最大迭代次数
        #maxIterCount = 0 if kwargs is None or 'maxIterCount' not in kwargs.keys() else int(kwargs['maxIterCount'])
        maxIterCount = 0 if kwargs is None or 'maxIterCount' not in kwargs else int(kwargs['maxIterCount'])
        iterCount = 0
        # 参数：最大时钟
        #maxclock = 0 if kwargs is None or 'maxclock' not in kwargs.keys() else float(kwargs['maxclock'])
        maxclock = 0 if kwargs is None or 'maxclock' not in kwargs else float(kwargs['maxclock'])
        lastclock,clock = 0.0,0.0

        # 重置网络运行信息
        net.reset()

        # 取得输入神经元
        inputNeurons = net.getInputNeurons()
        # 设置输入
        for i, neuron in enumerate(inputNeurons):
            if i >= len(inputs): break
            model = neuron.getModel()
            model.execute(neuron, net, value=inputs[i], no=activeno)

        # 参数：检查输出状态稳定的次数
        #outputMaxCheckCount = 1 if kwargs is None or 'outputMaxCheckCount' not in kwargs.keys() else int(
        #    kwargs['outputMaxCheckCount'])
        outputMaxCheckCount = 1 if kwargs is None or 'outputMaxCheckCount' not in kwargs else int(
            kwargs['outputMaxCheckCount'])
        outputCheckCount = 0
        lastOutputs = []

        while 1:
            # 检查是否达到最大时钟
            if maxclock > 0 and clock >= maxclock:
                return lastOutputs,clock
            # 取得所有所有神经元
            neurons = net.getNeurons()
            outputNeurons = net.getOutputNeurons()
            outputno = 0
            outputs = [None] * len(outputNeurons)
            nexttimes = []
            # 遍历所有神经元，检查下一次事件的时间
            for i, neuron in enumerate(neurons):
                if neuron in inputNeurons:
                    nexttimes.append(sys.maxsize)
                    continue
                # check函数检查下次发生状态发生变化的事件的时间
                model = neuron.getModel()
                time = model.check(neuron, net, value=inputs[i], no=activeno, time=(lastclock, clock, clock-lastclock))
                nexttimes.append(time)

            # 取得下次事件时间的最小值
            mintime = np.min(nexttimes)
            mintimeindex = np.argmin(mintime)
            lastclock,clock = clock,mintime

            # 执行将最早发生事件的那些神经元
            for index in mintimeindex:
                neuron = neurons[index]
                model = neuron.getModel()
                value, activation, firerate, time = \
                    model.execute(neuron, net, value=inputs[i], no=activeno, time=(lastclock, clock, clock-lastclock))

            # 获取输出
            outputs = []
            for j,outputNeuron in enumerate(outputNeurons):
                value,activation,firerate,time = outputNeuron.getReturnState()
                outputs.append((value,activation,firerate,time))

            # 如果设置了最大时钟，则持续执行直到最大时钟到达
            if maxclock > 0:
                continue

            # 检查输出神经元是否全部产生输出,以及输出是否还变化
            if not collections.all(outputs, lambda o: o != None and o[0] != None):  # 输出不全，继续循环
                lastOutputs = copy.deepcopy(outputs)
                continue

            # 检查输出与上次输出是否相等
            if operator.eq(lastOutputs, outputs):
                outputCheckCount += 1
                if outputCheckCount >= outputMaxCheckCount:
                    return lastOutputs,clock
            else:
                outputCheckCount = 0
                lastOutputs = copy.deepcopy(outputs)




    def _activateByTime(self,net,inputs,activeno,**kwargs):
        '''
        按照时间激活
        :param net:      NeuralNetwork  网络
        :param inputs:   Union(float,list,tuple,ndarray...)  输入
        :param activeno: int 激活编号
        :param kwargs:   dict 激活参数
        :return: tuple : （网络输出,结束时间)，网络输出是一个list，其中每项是一个元组(value,activation,firerate,time)
        '''
        # 参数:迭代时间间隔
        #ticks = 0.01 if kwargs is None or 'ticks' not in kwargs.keys() else float(kwargs['ticks'])
        ticks = 0.01 if kwargs is None or 'ticks' not in kwargs else float(kwargs['ticks'])
        # 参数：最大时钟
        #maxclock = 0 if kwargs is None or 'maxclock' not in kwargs.keys() else float(kwargs['maxclock'])
        maxclock = 0 if kwargs is None or 'maxclock' not in kwargs else float(kwargs['maxclock'])
        clock = 0.0

        # 重置网络运行信息
        net.reset()


        # 取得输入神经元
        inputNeurons = net.getInputNeurons()
        # 设置输入
        for i,neuron in enumerate(inputNeurons):
            if i >= len(inputs): break
            model = neuron.getModel()
            model.execute(neuron, net, value=inputs[i],no=activeno)

        # 参数：检查输出状态稳定的次数
        #outputMaxCheckCount = 1 if kwargs is None or 'outputMaxCheckCount' not in kwargs.keys() else int(kwargs['outputMaxCheckCount'])
        outputMaxCheckCount = 1 if kwargs is None or 'outputMaxCheckCount' not in kwargs else int(
            kwargs['outputMaxCheckCount'])
        outputCheckCount = 0
        lastOutputs = []

        # 反复检查运行结果
        while 1:
            # 设置时钟
            lastclock,clock = clock,clock + ticks
            if maxclock > 0 and clock >= maxclock:
                return lastOutputs,clock

            # 遍历所有神经元
            neurons = net.getNeurons()
            outputNeurons = net.getOutputNeurons()
            outputno = 0
            outputs = []*len(outputNeurons)
            for i,neuron in enumerate(neurons):
                if neuron in inputNeurons:
                    continue
                # execute函数检查是否各种状态是否发生变化，并计算变化；如果计算有效，若满足执行并返回结果，否则返回的是上次的结果
                model = neuron.getModel()
                value, activation, firerate, time = \
                    model.execute(neuron, net, value=inputs[i], no=activeno,time=(lastclock,clock,ticks))

                # 如果本次执行有效,且是输出神经元
                if neuron in outputNeurons:
                    if time == clock:
                        outputs[outputno] = (value,activation,firerate,time)
                    outputno += 1


            # 如果设置了最大时钟，则持续执行直到最大时钟到达
            if maxclock > 0:
                continue
            # 检查输出神经元是否全部产生输出,以及输出是否还变化
            if not collections.all(outputs,lambda o:o != None and o[0] != None): # 输出不全，继续循环
                lastOutputs = copy.deepcopy(outputs)
                continue

            # 检查输出与上次输出是否相等
            if operator.eq(lastOutputs,outputs):
                outputCheckCount += 1
                if outputCheckCount >= outputMaxCheckCount:
                    return lastOutputs,clock
            else:
                outputCheckCount = 0
                lastOutputs = copy.deepcopy(outputs)



# 运行器注册
runners = Registry()
runners.register(SimpleNeuralNetworkRunner(),'simple')
runners.register(GeneralNeuralNetworkRunner(),'general')

ExtendJsonEncoder.ignoreTypes.append(NeuralNetworkTask)
