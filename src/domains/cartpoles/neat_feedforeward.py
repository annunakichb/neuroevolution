# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import gc
import numpy as np
import os
import csv

from domains.cartpoles.enviornment.cartpole import SingleCartPoleEnv
import ne.callbacks as callbacks
import ne.neat as neat
from brain.networks import NetworkType
from brain.networks import NeuralNetwork
from brain.runner import NeuralNetworkTask
from evolution.env import Evaluator
from evolution.session import EvolutionTask

import domains.cartpoles.enviornment.force as force
import domains.cartpoles.enviornment.runner as runner
from brain.viewer import NetworkView
import utils.files as files



def fitness(ind,session):
    '''
    以连续不倒的次数作为适应度
    :param ind:
    :param session:
    :return:
    '''
    env = SingleCartPoleEnv()
    net = ind.getPhenome()
    reward_list, notdone_count_list = runner.do_evaluation(1,env,net.activate)

    return max(notdone_count_list)

fitness_records = []
complex_records=[]
maxfitness_records = []
env = SingleCartPoleEnv()
mode = 'noreset'
epochcount = 0
maxepochcount = 10
complexunit = 20.
modularities = []

# 记录最优个体的平衡车运行演示视频
def callback(event,monitor):
    callbacks.neat_callback(event,monitor)
    global epochcount
    global  mode
    if event == 'epoch.end':
        gc.collect()
        maxfitness = monitor.evoTask.curSession.pop.inds[0]['fitness']
        maxfitness_ind = monitor.evoTask.curSession.pop.inds[0]
        maxfitness_records.append(maxfitness)

        # 如果最大适应度达到了env.max_notdone_count（说明对当前环境已经产生适应），或者进化迭代数超过10次（当前环境下适应达到最大）
        # 则提升复杂度
        epochcount += 1
        if maxfitness >= env.max_notdone_count or epochcount >= maxepochcount:
            # 保存复杂度和对应最大适应度记录
            fitness_records.append(maxfitness)
            complex_records.append(force.force_generator.currentComplex())
            print([(f, c) for f, c in zip(complex_records, fitness_records)])
            #np.save('neat_result', (complex_records, fitness_records))

            # 保存过程中每一步的最大适应度记录
            filename = os.path.split(os.path.realpath(__file__))[0] + os.sep + 'datas_' + mode + os.sep + \
                    'neat' + os.sep + 'neat.csv'
            out = open(filename, 'a', newline='')
            csv_write = csv.writer(out, dialect='excel')
            csv_write.writerow([complex_records[-1]]+maxfitness_records)
            maxfitness_records.clear()

            # 保存最优基因网络拓扑
            filename = os.path.split(os.path.realpath(__file__))[0] + os.sep + 'datas_' + mode + os.sep + \
                    'neat' + os.sep + str(complex_records[-1]) + '_neat_ind' + str(maxfitness_ind.id) + '_' + \
                       str(maxfitness) + '.svg'
            netviewer = NetworkView()
            netviewer.drawNet(maxfitness_ind.genome, filename=filename, view=False)

            # 计算最优网络模块度
            modularity = maxfitness_ind.genome.compute_modular()
            modularities.append((complex_records[-1],modularity))
            print(modularities)
            filename = os.path.split(os.path.realpath(__file__))[0] + os.sep + 'datas_' + mode + os.sep + \
                       'neat' + os.sep + 'modulars.csv'
            out = open(filename, 'a', newline='')
            csv_write = csv.writer(out, dialect='excel')
            csv_write.writerow([complex_records[-1],modularity])

            # 提升复杂度
            changed, maxcomplex, k,w, f, sigma = force.force_generator.promptComplex(complexunit)
            if changed:
                print('环境复杂度=%.3f,k=%.2f,w=%.2f,f=%.2f,sigma=%.2f' % (maxcomplex, k,w, f, sigma))
                if mode == 'reset':
                    monitor.evoTask.curSession.runParam.terminated.maxIterCount = epochcount-1
                if maxcomplex >= 400:
                    monitor.evoTask.curSession.runParam.terminated.maxIterCount = epochcount - 1
            else:
                np.save('neat_result', complex_records, fitness_records)
                #plt.plot(complex_records, reward_list, label='reward')
                plt.plot(complex_records, fitness_records, label='times')
                plt.xlabel('complexes')
                plt.savefig('./neat_cartpole.png')
                return

            epochcount = 0
    elif event == 'session.end':
        filename = 'singlecartpole.session.'+ str(monitor.evoTask.curSession.taskxh)+'.mov'
        eliest = monitor.evoTask.curSession.pop.eliest


def run(**kwargs):
    global mode
    global maxepochcount
    global complexunit

    #mode = 'noreset' if 'mode' not in kwargs.keys() else kwargs['mode']
    #maxepochcount = 10 if 'maxepochcount' not in kwargs.keys() else int(kwargs['maxepochcount'])
    #complexunit = 20.0 if 'complexunit' not in kwargs.keys() else float(kwargs['complexunit'])
    mode = 'noreset' if 'mode' not in kwargs else kwargs['mode']
    maxepochcount = 10 if 'maxepochcount' not in kwargs else int(kwargs['maxepochcount'])
    complexunit = 20.0 if 'complexunit' not in kwargs else float(kwargs['complexunit'])

    while True:
        execute()
        if mode == 'reset':
            continue
        break


def execute():
    '''
    执行
    :return:
    '''
    # 初始化neat算法模块
    neat.neat_init()

    # 定义网络训练任务

    task = NeuralNetworkTask()

    # 定义网络
    netdef = {
        'netType' : NetworkType.Perceptron,                       # NetworkType，网络类型,必须
        'neuronCounts' : [4,1],                                   # list（初始）网络各层神经元数量,必须
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
        'size':50,                                               #种群大小，必须
        'elitistSize':0.05,                                        #精英个体占比，小于1表示比例，大于等于1表示数量
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
            'maxIterCount' : 100000,                               # 最大迭代次数，必须
            'maxFitness' : 1000000.,                                   # 最大适应度，必须
        },
        'log':{
            'individual' : 'elite',                                 # 日志中记录个体方式：记录所有个体，可以选择all,elite,maxfitness（缺省）,custom
            'debug': False,                                        # 是否输出调试信息
            'file': 'neat_cartpole.log'                            # 日志文件名
        },
        'evalate':{
            'parallel':0,                                         # 并行执行评估的线程个数，缺省0，可选
        },
        'operations':{
            #'method' : 'neat',                                   # 已有的进化操作序列名称，与text两个只用一个
            'text' : 'neat_selection,neat_crossmate,neat_mutate'  # 进化操作序列
        },
        'mutate':{
            'propotion' : 0.1,                                      # 变异比例,有多少个个体参与变异，小于等于1表示比例，大于1表示固定数量
            'parallel': 0,  # 并行执行变异的线程个数，缺省0，可选
            'model':{
                'rate' : 0.0,                                     # 模型变异比例
                'range' : ''                                      # 可选的计算模型名称，多个用逗号分开，缺省是netdef中所有模型
            },
            'activation':{
                'rate' : 0.0,                                     # 激活函数的变异比率
                'range':'sigmod'                                  # 激活函数的
            },
            'topo' : {
                'addnode' : 0.4,                                  # 添加节点的概率
                'addconnection':0.4,                              # 添加连接的概率
                'deletenode':0.1,                                 # 删除节点的概率
                'deleteconnection':0.1                            # 删除连接的概率
            },
            'weight':{
                'parallel': 0,                                    # 并行执行权重变异的线程个数，缺省0，可选
                'epoch':3,                                          # 权重调整次数
            }
        }

    }
    gc.disable()
    evolutionTask = EvolutionTask(1,popParam,callback)
    evolutionTask.execute(runParam)
    gc.enable()
if __name__ == '__main__':

    force.init()
    run(mode='noreset',maxepochcount=50,complexunit=5.)


