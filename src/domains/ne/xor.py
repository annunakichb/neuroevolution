#!/usr/bin/python3

from brain.networks import NetworkType
from brain.runner import NeuralNetworkTask

from evolution.env import Evaluator

def dict_getattr(self, item):
    return self[item]

def dict_setattr(self,item,value):
    self[item] = value

dict.__getattr__ = dict_getattr
dict.__setattr__ = dict_setattr


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
    'idGenerator' :  'neat',                                  # 生成网络，神经元，突触id的类，参见DefauleIDGenerator,list idgenerator命令可以列出所有的id生成器对象
    'config' : {
        'layered' : True,                                     # 是否分层
        'substrate' : True,                                   # 是否使用基座
        'acyclie' : False,                                    # 是否允许自身连接
        'recurrent':False,                                    # 是否允许同层连接
        'reversed':False,                                     # 是否允许反向连接
        'dimension':2,                                        # 空间坐标维度
    },
    'runner':{
        'name' : 'simple',                                    # 网络运行器名称,必须
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

# 定义适应度评估函数
def fitness(ind,session):
    net = ind.genome
    nettask = net.doTest()
    accuracy = nettask[NeuralNetworkTask.INDICATOR_ACCURACY]
    if accuracy >= 1.0:return 100
    return (4 - 4 * nettask[NeuralNetworkTask.INDICATOR_MEAN_ABSOLUTE_ERROR]) + 10 * accuracy


# 定义种群
popParam = {
    'indTypeName' : 'network',                                #种群的个体基因类型名，必须，该类型的个体基因应已经注册过，参见evolution.agent,必须
    'genomeFactory':None,                                     #基因工厂，个体类型中已经提供了基因工厂对象，这里如果设置，可以替换前者，可选
    'genomeDefinition' : netdef,                              #基因定义参数
    'size':100,                                               #种群大小，必须
    'elitistSize':0.2,                                        #精英个体占比，小于1表示比例，大于等于1表示数量
    'species':{                                               #物种参数，可选
        'method':'',                                          #物种分类方法
        'size':0,                                             #物种个体数量限制，0表示无限制或动态
    },
    'features':{                                              # 特征评估函数配置，必须
        'fitness' : Evaluator('fitness',{fitness,1.0})        # 适应度评估器,如果评估器只包含一个函数,也可以写成Evaluator('fitness',fitness)
    }
}

# 定于运行参数
runParam = {
    'terminated' : {
        'maxIterCount' : 1000000,                             # 最大迭代次数
        'maxFitness' : 100,                                   # 最大适应度
    },
    'log':{
        'individual' : 'all',                                 # 日志中记录个体方式：记录所有个体，可以选择all,elite,maxfitness,custom
    },
    'operations':{
        'text' : 'neat_selection,neat_crossmate,neat_mutate'  # 进化操作序列
    },
    'mutate':{
        'propotion' : 1.0,                                    # 变异比例,有多少个个体参与变异，小于等于1表示比例，大于1表示固定数量
        'items' :{
            'weight' : 0.0,                                   # 个体变异改变权重概率
            'addnode' : 0.45,                                 # 个体变异添加节点概率
            'addconnection' : 0.45,                           # 个体变异添加连接概率
            'removenode' : 0.05,                              # 个体变异删除节点概率
            'removeconnection' : 0.05                         # 个体变异删除连接概率
        }
    }
}

