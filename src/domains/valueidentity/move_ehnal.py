#!/usr/bin/python3
# -*- coding: UTF-8 -*-

'''
该实验研究EHNAL(Evolutionary Heterogeneous Neural Network Based on Attention Logic)如何将物体向特定位置的移动。
在一个粗糙的水平面上放置一个物体，机械臂需要将物体推动到距离物体1米以外的位置。
机械臂的传感器输入是物体距离目标的距离和当前速度
算法的输出是推力的大小

假设摩擦力恒定。

'''

import ne
import ne.senal as ehnal
import ne.callbacks as callbacks
from brain.networks import NetworkType
from brain.networks import NeuralNetwork
from brain.runner import NeuralNetworkTask
from evolution.env import Evaluator
from evolution.session import EvolutionTask
from utils.properties import Properties

# 定义适应度评估函数,让机械臂完成一次推动操作，当物体移动距离超过1米，或物体速度已经为0，输出推力是0的情况下结束。
# 1-结束时距离目标位置的距离为适应度大小
def fitness(ind,session):
    # 物体质量和摩擦力大小
    quality = 0.2  #物体质量200克
    friction = 5   #摩擦力是5N
    # 初始距离和初始速度
    distance  = 1 #米
    speed = 0     #米/秒
    inputs = [distance,speed]

    interval = 0.01 #秒

    net = ind.genome
    while 1:
        result = net.run(inputs) #由于是非监督学习，调用run获取网络运行结果
        power = result[0]        # 力的大小
        if power == 0 and speed == 0:
            break
        a = (power - friction) / quality
        s = speed * interval + 0.5 * a * interval * interval
        distance -= s
        if distance <= 0:
            break
    return 1-abs(distance)


def run():
    # 初始化算法模块
    senal.ehnal_init()

    # 定义网络
    netdef = {
        'netType': NetworkType.Perceptron,  # NetworkType，网络类型,必须
        'neuronCounts': [2, 1],  # list（初始）网络各层神经元数量,必须
        'idGenerator': 'neat',  # str 生成网络，神经元，突触id的类，参见DefauleIDGenerator,list idgenerator命令可以列出所有的id生成器对象
        'config': {
            'layered': True,  # bool 是否分层,可选
            'substrate': True,  # bool 是否使用基座,可选
            'acyclie': False,  # bool 是否允许自身连接,可选
            'recurrent': False,  # bool 是否允许同层连接,可选
            'reversed': False,  # bool 是否允许反向连接,可选
            'dimension': 2,  # int 空间坐标维度,可选
            'range': NeuralNetwork.MAX_RANGE,  # list 坐标范围，可选'
        },
        'runner': {
            'name': 'simple',  # str 网络运行器名称,必须
            'task': task,  # NeuralNetworkTask,网络运行任务,必须
        },
        'models': {  # dict 神经元计算模型的配置信息,必须
            'input': {  # str 模型配置名称（不是模型名称）
                'name': 'input',  # str,名称，与上面总是一样,可选
                'modelid': 'input',  # str，模型id，必须，用这个来找到对应的计算模型对象,因此应确保该计算模型已注册
            },
            'hidden': {
                'name': 'hidden',  # str 隐藏神经元配置名称，可选
                'modelid': 'hidden',  # str 隐藏神经元计算模型id,必须
                'activationFunction': {  # dict 可选
                    'name': 'sigmod',  # str 激活函数名称，必须
                    'a': 1.0, 'b': 1.0, 'T': 1.0  # float 激活函数参数，可选
                },
                'bias': 'uniform[-30.0:30.0]',  # str 隐藏神经元的偏置变量，均匀分布，必须，可以是uniform[begin,end]或者normal(u,sigma)
            },
            'synapse': {
                'name': 'synapse',  # str 突触计算模型配置名称,可选
                'modelid': 'synapse',  # str 突触计算模型Id，必须
                'weight': 'uniform[-30.0:30.0]'  # str 突触学习变量，均匀分布，必须
            }
        }
    }