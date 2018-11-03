#!/usr/bin/python3

from brain.networks import NetworkType
from brain.runner import NeuralNetworkTask

# 定义网络训练任务
train_x = [[0,0],[0,1],[1,0],[1,1]]
train_y = [0,1,1,0]
test_x = [[0,0],[0,1],[1,0],[1,1]]
test_y = [0,1,1,0]
task = NeuralNetworkTask(train_x,train_y,test_x,test_y)

# 定义网络
netdef = {
    'netType' : NetworkType.Perceptron,                       # 网络类型,必须
    'neuronCounts' : [2,1],                                   # （初始）网络各层神经元数量,必须
    'idGenerator' :  'neat',                                  # 生成网络，神经元，突触id的类，参见DefauleIDGenerator
    'config' : {
        'layered' : True,                                     # 是否分层
        'substrate' : True,                                   # 是否使用基座
        'acyclie' : False,                                    # 是否允许自身连接
        'recurrent':False,                                    # 是否允许同层连接
        'reversed':False,                                     # 是否允许反向连接
        'dimension':2,                                        # 空间坐标维度
    },
    'runner':{
        'name' : 'simple',                                    # 网络运行器名称
        'task' : task,                                        # 网络运行任务,必须
    },
    'models':{                                                # 神经元计算模型的配置信息,必须
        'input':{                                             # 模型配置名称（不是模型名称）
            'name' : 'input',                                 # 名称，与上面总是一样
            'modelid':'input',                                # 模型id，用这个来找到对应的计算模型对象,因此应确保该计算模型已注册
        },
        'hidden':{
            'name':'hidden',                                  # 隐藏神经元配置名称
            'modelid':'hidden',                               # 隐藏神经元计算模型id
            'activationFunction':{
                'name' : 'sigmod',                            # 激活函数名称
                'a' :1.0,'b':1.0,'T':1.0                      # 激活函数参数
            },
            'bias':'Uniform[-5.0:5.0]',                       # 隐藏神经元的偏置变量，均匀分布
        },
        'synapse':{
            'name':'synapse',                                 # 突触计算模型配置名称
            'modelid':'synapse',                              # 突触计算模型Id
            'weight':'Normal[0.3,1.5]'                        # 突触学习变量，正态分布
        }
    }


}