# -*- coding: UTF-8 -*-


__all__=['activation','elements','models','networks','runner']

def brain_init():
    pass

from utils.properties import Properties
from brain.networks import NetworkType
from brain.networks import NeuralNetwork

__default_netdef_params__ = {}
def setNetDefDefaultParam(**kwargs):
    if kwargs is None or len(kwargs)<=0:return
    for key,value in kwargs.items():
        __default_netdef_params__[key] = value
def getNetDefDefaultParam(key):
    return None if key not in __default_netdef_params__ else __default_netdef_params__[key]


def createNetDef(**kwargs):
    '''
    创建网络定义参数
    :param kwargs dict 参数字典，包括:
            task         NeuralNetworkTask 任务对象
            neuronCounts list,初始神经元数量，一般是输入神经元和输出神经元的初始数量，例如[15，5]
            netType      int  参见brain.networks.NetworkType，缺省为感知机NetworkType.Perceptron
            idGenerator  IdGenerator，神经网络中所有元素的id生成器，缺省为DefaultIdGenerator
            layered      bool 是否分层，缺省为True
            substrate    bool 是否有基座，缺省为True
            acyclie      bool 是否有自循环节点，缺省为False
            recurrent    bool 是否有递归结构，缺省为False
            reversed     bool 是否有反向连接，缺省为False
            dimension    int  坐标维度，缺省为2，0表示不设置坐标
            range        list 坐标范围，例如[(0,1000),(0,9999999),(0,99999)]，分别为三维坐标的范围，缺省为NeuralNetwork.MAX_RANGE
            runnername   str  网络运行器名称，缺省为'simple'
            models       dict 多个模型，每个模型又是一个dict，必须包括name和modelid，例如:
                                'input':{                                             # str 模型配置名称（不是模型名称）
                                    'name' : 'input',                                 # str,名称，与上面总是一样,可选
                                    'modelid':'input',                                # str，模型id，必须，用这个来找到对应的计算模型对象,因此应确保该计算模型已注册
                                },
                                'hidden':{
                                    'name':'hidden',                                  # str 隐藏神经元配置名称，可选
                                    'modelid':'hidden',                               # str 隐藏神经元计算模型id,必须
                                    'activationFunction':{                            # dict 可选
                                        'name' : 'sigmod',                            # str 激活函数名称，必须
                                        'a' :1.0,'b':1.0,'T':1.0                      # float 激活函数参数，可选
                                    },
                                    'bias':'uniform[-3.0:3.0]',                     # str 隐藏神经元的偏置变量，均匀分布，必须，可以是uniform[begin,end]或者normal(u,sigma)
                                },
                                'synapse':{
                                    'name':'synapse',                                 # str 突触计算模型配置名称,可选
                                    'modelid':'synapse',                              # str 突触计算模型Id，必须
                                    'weight':'uniform[-3.0:3.0]'                    # str 突触学习变量，均匀分布，必须
                                }

    :return: dict 网络定义参数
    '''
    if kwargs is None: kwargs = {}
    if 'neuronCounts' not in kwargs:raise RuntimeError('can not find neuronCounts in createNetDef ')
    # 定义网络
    netdef = {
        'netType' : NetworkType.Perceptron if 'netType' not in kwargs else  kwargs['netType'],                       # NetworkType，网络类型,必须
        'neuronCounts' : kwargs['neuronCounts'],                                                                     # list（初始）网络各层神经元数量,必须
        'idGenerator' :  'default' if 'idGenerator' not in kwargs else  kwargs['idGenerator'],                          # str 生成网络，神经元，突触id的类，参见DefauleIDGenerator,list idgenerator命令可以列出所有的id生成器对象
        'config' : {
            'layered' : True if 'layered' not in kwargs else  kwargs['layered'],                                     # bool 是否分层,可选
            'substrate' : True if 'substrate' not in kwargs else  kwargs['substrate'],                               # bool 是否使用基座,可选
            'acyclie' : False if 'acyclie' not in kwargs else  kwargs['acyclie'],                                    # bool 是否允许自身连接,可选
            'recurrent':False if 'recurrent' not in kwargs else  kwargs['recurrent'],                                # bool 是否允许同层连接,可选
            'reversed':False if 'reversed' not in kwargs else  kwargs['reversed'],                                   # bool 是否允许反向连接,可选
            'dimension':2  if 'dimension' not in kwargs else  kwargs['dimension'],                                   # int 空间坐标维度,可选
            'range':NeuralNetwork.MAX_RANGE if 'range' not in kwargs else  kwargs['range'],                  # list 坐标范围，可选'
        },
        'runner':{
            'name' : 'simple' if 'runnername' not in kwargs else  kwargs['runnername'] ,                             # str 网络运行器名称,必须
            'task' : None if 'task' not in kwargs else kwargs['task']                                        # NeuralNetworkTask,网络运行任务,必须
        },
        'models':kwargs['models'] if 'models' in kwargs else {                                                # dict 神经元计算模型的配置信息,必须
            'input':{                                             # str 模型配置名称（不是模型名称）
                'name' : 'input',                                 # str,名称，与上面总是一样,可选
                'modelid':'input',                                # str，模型id，必须，用这个来找到对应的计算模型对象,因此应确保该计算模型已注册
            },
            'hidden':{
                'name':'hidden',                                  # str 隐藏神经元配置名称，可选
                'modelid':'hidden',                               # str 隐藏神经元计算模型id,必须
                'activationFunction':{                            # dict 可选
                    'name' : 'sigmod',                            # str 激活函数名称，必须
                    'a' :1.0,'b':1.0,'T':1.0                      # float 激活函数参数，可选
                },
                'bias':'uniform[-3.0:3.0]',                       # str 隐藏神经元的偏置变量，均匀分布，必须，可以是uniform[begin,end]或者normal(u,sigma)
            },
            'synapse':{
                'name':'synapse',                                 # str 突触计算模型配置名称,可选
                'modelid':'synapse',                              # str 突触计算模型Id，必须
                'weight':'uniform[-3.0:3.0]'                      # str 突触学习变量，均匀分布，必须
            }
        }
    }
    return Properties(netdef)


