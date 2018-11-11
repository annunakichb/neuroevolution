#!/usr/bin/python3

import ne
import ne.neat as neat
import ne.callbacks as callbacks
from brain.networks import NetworkType
from brain.networks import NeuralNetwork
from brain.runner import NeuralNetworkTask
from evolution.env import Evaluator
from evolution.session import EvolutionTask
from utils.properties import Properties

# 定义适应度评估函数
def fitness(ind,session):
    net = ind.genome
    net.doTest()
    #accuracy = net[NeuralNetworkTask.INDICATOR_ACCURACY]
    #return accuracy
    #if accuracy >= 1.0:return 100

    return (4 - 4 * net[NeuralNetworkTask.INDICATOR_MEAN_SQUARED_ERROR]) # + 10 * accuracy

def run():
    # 初始化neat算法模块
    neat.neat_init()

    # 定义网络训练任务
    train_x = [[0,0],[0,1],[1,0],[1,1]]
    train_y = [0,1,1,0]
    test_x = [[0,0],[0,1],[1,0],[1,1]]
    test_y = [0,1,1,0]
    task = NeuralNetworkTask(train_x,train_y,test_x,test_y)

    # 定义网络
    netdef = {
        'netType' : NetworkType.Perceptron,                       # NetworkType，网络类型,必须
        'neuronCounts' : [2,1],                                   # list（初始）网络各层神经元数量,必须
        'idGenerator' :  'neat',                                  # str 生成网络，神经元，突触id的类，参见DefauleIDGenerator,list idgenerator命令可以列出所有的id生成器对象
        'config' : {
            'layered' : True,                                     # bool 是否分层,可选
            'substrate' : True,                                   # bool 是否使用基座,可选
            'acyclie' : False,                                    # bool 是否允许自身连接,可选
            'recurrent':False,                                    # bool 是否允许同层连接,可选
            'reversed':False,                                     # bool 是否允许反向连接,可选
            'dimension':2,                                        # int 空间坐标维度,可选
            'range':NeuralNetwork.MAX_RANGE,                      # list 坐标范围，可选'
        },
        'runner':{
            'name' : 'simple',                                    # str 网络运行器名称,必须
            'task' : task,                                        # NeuralNetworkTask,网络运行任务,必须
        },
        'models':{                                                # dict 神经元计算模型的配置信息,必须
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
                'bias':'uniform[-30.0:30.0]',                     # str 隐藏神经元的偏置变量，均匀分布，必须，可以是uniform[begin,end]或者normal(u,sigma)
            },
            'synapse':{
                'name':'synapse',                                 # str 突触计算模型配置名称,可选
                'modelid':'synapse',                              # str 突触计算模型Id，必须
                'weight':'uniform[-30.0:30.0]'                    # str 突触学习变量，均匀分布，必须
            }
        }
    }

    # 定义种群
    popParam = {
        'indTypeName' : 'network',                                #种群的个体基因类型名，必须，该类型的个体基因应已经注册过，参见evolution.agent,必须
        'genomeFactory':None,                                     #基因工厂，个体类型中已经提供了基因工厂对象，这里如果设置，可以替换前者，可选
        'factoryParam' :{                                         # 工厂参数，必须
           'connectionRate':1.0,                                  # 连接比率
        },
        'genomeDefinition' : netdef,                              #基因定义参数,可选
        'size':100,                                               #种群大小，必须
        'elitistSize':0.05,                                       #精英个体占比，小于1表示比例，大于等于1表示数量
        'species':{                                               #物种参数，可选
            'method':'neat_species',                              # 物种分类方法,在物种参数中必须
            'alg':'kmean',                                        # 算法名称
            'size': 5,                                            # 物种个体数量限制，0表示无限制或动态
            'iter':50,                                            # 算法迭代次数
        },
        'features':{                                              # 特征评估函数配置，必须
            'fitness' : Evaluator('fitness',[(fitness,1.0)])      # 适应度评估器,如果评估器只包含一个函数,也可以写成Evaluator('fitness',fitness)
        }
    }

    # 定于运行参数
    runParam = {
        'terminated' : {
            'maxIterCount' : 300,                                # 最大迭代次数，必须
            'maxFitness' : 3.95,                                   # 最大适应度，必须
        },
        'log':{
            'individual' : 'elite',                               # 日志中记录个体方式：记录所有个体，可以选择all,elite,maxfitness（缺省）,custom
            'debug': False                                         # 是否输出调试信息
        },
        'evalate':{
            'parallel':0,                                         # 并行执行评估的线程个数，缺省0，可选
        },
        'operations':{
            #'method' : 'neat',                                   # 已有的进化操作序列名称，与text两个只用一个
            'text' : 'neat_selection,neat_crossmate,neat_mutate'  # 进化操作序列
        },
        'mutate':{
            'propotion' : 0.15,                                   # 变异比例,有多少个个体参与变异，小于等于1表示比例，大于1表示固定数量
            'model':{
                'rate' : 0.0,                                     # 模型变异比例
                'range' : ''                                      # 可选的计算模型名称，多个用逗号分开，缺省是netdef中所有模型
            },
            'activation':{
                'rate' : 0.0,                                     # 激活函数的变异比率
                'range':'sigmod'                                  # 激活函数的
            },
            'topo' : {
                'addnode' : 0.45,                                  # 添加节点的概率
                'addconnection':0.45,                              # 添加连接的概率
                'deletenode':0.03,                                 # 删除节点的概率
                'deleteconnection':0.07                            # 删除连接的概率
            },
            'weight':{
                'epoch':5,                                         # 权重调整次数
            }
        }
    }

    evolutionTask = EvolutionTask(1,popParam,callbacks.neat_callback)
    evolutionTask.execute(runParam)

if __name__ == '__main__':
    run()