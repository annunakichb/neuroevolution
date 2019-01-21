from sympy import *
import  numpy as np

import brain
from brain.networks import NetworkType
from brain.networks import NeuralNetwork
import ne.ehnal as ehnal
from evolution.session import Evaluator
from evolution.session import EvolutionTask

from ne.ehnal.factory import EhnalNeuralNetworkGenomeFactory

class Env:
    g = 9.8         # 重力常数 N/kg
    C = 2.5         # 摩擦力常数
    m = 0.1         # 小球质量 kg
    l = 2.          # 绳长 m
    Tmin = -5.      # N 最小扭矩
    Tmax = 5.       # N 最大扭矩
    height = 1.     # m 垂直高度
    destnation = 3. # m 目的地坐标

# 适应度函数
def fitness(ind,session):
    '''
    适应度函数
    :param ind:      个体
    :param session:  进化任务session
    :return:         若最终小球始终没有移动，适应度为0
                     若小球没有抛出，则适应度为1
                     若小球抛出，则垂足点到目标点的距离 / 着落点到目标点的距离为适应度
    '''
    # 创建网络
    net = ind.getPhenome()
    # 运行器和运行参数
    runner = brain.runner.runners.find('ehnal')
    runParam = {'runMode': 'event', 'outputMaxCheckCount': 2}

    # 最大实验次数和初始最大适应度
    maxExperimentCount = 10
    maxfitness = 0.
    for i in range(maxExperimentCount):
        # 角度，角速度，微分方程求解t间隔，微分方程求解次数，网络激活时间
        theta, omega, step,maxclock,clock = 0., 0., 0.01,30,0.
        # 网络输出结果：是否切断，落地点坐标
        cut,distance = 0,0
        maxIterCount = 100
        iterCount = 0
        while 1:
            # 检查迭代次数
            iterCount += 1
            if iterCount >= maxIterCount:
                break
            # 设置输入
            inputs = [theta,omega]
            # 激活网络
            outputs = runner.activate(net,inputs,clock,**runParam)
            # 网络没有输出有效结果
            if outputs is None or len(outputs) <= 0:
                clock += step
                continue
            # 没有切断绳子
            elif outputs[1] is None or outputs[1][1] == False:
                T = outputs[0][0] # 网络输出的扭矩值
                T = (Env.Tmax-Env.Tmin) * (T - 1)/(exp(1.)-1) + Env.Tmin # 归一化到Tmin - Tmax之间
                theta,omega = __diff_beforethrow((theta,omega),step,(T,))
                if theta < -pi / 2:
                    theta = - pi / 2
                    omega = 0
                elif theta > pi / 2:
                    theta = pi / 2
                    omega = 0
            # 绳子被切断
            else:
                # 计算起点坐标
                x = Env.l * sin(theta)
                y = (Env.l + Env.height) - Env.l * cos(theta)
                # 计算线速度
                v = omega * Env.l
                # 计算落地时间 y = V*t*sin(theta) - g*t*t/2
                temp = 2 * v * sin(theta)
                t = temp + sqrt(temp*temp - 8 * Env.g * -1 * y)
                # 计算飞行距离
                d = v * t * cos(theta)
                # 计算落地点x坐标
                x = x + d
                distance = x
                cut = 1
                break

        # 计算适应度
        if cut == 0:
            fitness = 0.
        else: #若绳子被切断
            t = abs(distance - Env.destnation)
            fitness = exp(-1*(t*t)/(Env.destnation*Env.destnation))
        if maxfitness < fitness:
            maxfitness = fitness

        # 　网络学习过程
        reward = fitness
        o1_deviation_direction = 1 if cut == 0 else 0
        o1_deviation_value = 1 if cut == 0 else 0
        o2_deviation_direction = sign(distance - Env.destnation)
        o2_deviation_value = abs(distance - Env.destnation)
        net.feekback(reward,(o1_deviation_direction,o1_deviation_value),(o2_deviation_direction,o2_deviation_value))

    return maxfitness

def __diff_beforethrow(v,steps,params):
    '''
    小球做单摆运动的微分方程，因为T的值
    :param v: tuple x和y的值，x是theta角度(-pi/2 - pi/2),y是角速度
    :param t: float 时间
    :param params: tuple T的值 Tmin = -5N Tmax = 5N
    :return:
    '''
    theta,omega = v
    T, = params
    d_theta = omega
    d_omega = -C * omega + Env.g * np.sin(theta) / Env.l + T / (Env.m*Env.l*Env.l)

    omega = omega + steps * d_omega
    theta = theta + steps * d_theta
    return theta,omega



def run():
    # 初始化
    ehnal.ehnal_init()

    # 定义网络
    netdef = {
        'netType': NetworkType.Perceptron,  # NetworkType，网络类型,必须
        'neuronCounts': [3, 2],  # list（初始）网络各层神经元数量,必须
        'neurons':[
                    [# 第一层有三个输入
                     {'caption':'目标距离','modelName':'input','attributes':{},'params':{}},
                     {'caption':'摆动角度', 'modelName': 'input', 'attributes': {}, 'params': {}},
                     {'caption':'角速度', 'modelName': 'input', 'attributes': {}, 'params': {}},
                     {'caption':'物体位置', 'modelName': 'input', 'attributes': {}, 'params': {}}
                    ],
                    [# 最后一层有两个输入
                        {'caption': '扭矩', 'modelName': 'output', 'attributes': {}, 'params': {}},
                        {'caption': '切断点', 'modelName': 'output', 'attributes': {}, 'params': {}}
                    ]
                  ],
        'idGenerator': 'default',  # str 生成网络，神经元，突触id的类，参见DefauleIDGenerator,list idgenerator命令可以列出所有的id生成器对象
        'config': {
            'layered': False,  # bool 是否分层,可选
            'substrate': False,  # bool 是否使用基座,可选
            'acyclie': True,  # bool 是否允许自身连接,可选
            'recurrent': False,  # bool 是否允许同层连接,可选
            'reversed': True,  # bool 是否允许反向连接,可选
            'dimension': 2,  # int 空间坐标维度,可选
            'range': NeuralNetwork.MAX_RANGE,  # list 坐标范围，可选'
        },
        'runner': {
            'name': 'ehnal',  # str 网络运行器名称,必须
            'task': task,  # NeuralNetworkTask,网络运行任务,必须
        },
        'models': {  # dict 神经元计算模型的配置信息,必须
            'input': {  # str 模型配置名称（不是模型名称）
                'name': 'input',     # str,名称，与上面总是一样,可选
                'modelid': 'input',  # str，模型id，必须，用这个来找到对应的计算模型对象,因此应确保该计算模型已注册
            },
            'output':{
                'modelid': 'output'  # str，模型id，必须，用这个来找到对应的计算模型对象,因此应确保该计算模型已注册
            }
        }
    }

    # 定义种群
    popParam = {
        'indTypeName': 'network',  # 种群的个体基因类型名，必须，该类型的个体基因应已经注册过，参见evolution.agent,必须
        'genomeFactory': EhnalNeuralNetworkGenomeFactory(),  # 基因工厂，个体类型中已经提供了基因工厂对象，这里如果设置，可以替换前者，可选
        'factoryParam': {  # 工厂参数，必须
            'connectionRate': 1.0,  # 连接比率
        },
        'genomeDefinition': netdef,  # 基因定义参数,可选
        'size': 100,  # 种群大小，必须
        'elitistSize': 0.05,  # 精英个体占比，小于1表示比例，大于等于1表示数量
        'species': {  # 物种参数，可选
            'method': 'neat_species',  # 物种分类方法,在物种参数中必须
            'alg': 'kmean',  # 算法名称
            'size': 5,  # 物种个体数量限制，0表示无限制或动态
            'iter': 50,  # 算法迭代次数
        },
        'features': {  # 特征评估函数配置，必须
            'fitness': Evaluator('fitness', [(fitness, 1.0)])  # 适应度评估器,如果评估器只包含一个函数,也可以写成Evaluator('fitness',fitness)
        }
    }

    # 定于运行参数
    runParam = {
        'terminated': {
            'maxIterCount': 1000000,  # 最大迭代次数，必须
            'maxFitness': 0.98,  # 最大适应度，必须
        },
        'log': {
            'individual': 'elite',  # 日志中记录个体方式：记录所有个体，可以选择all,elite,maxfitness（缺省）,custom
            'debug': False  # 是否输出调试信息
        },
        'evalate': {
            'parallel': 0,  # 并行执行评估的线程个数，缺省0，可选
        },
        'operations': {
            # 'method' : 'neat',                                   # 已有的进化操作序列名称，与text两个只用一个
            'text': 'neat_selection,neat_crossmate,neat_mutate'  # 进化操作序列
        },
        'mutate': {
            'propotion': 0.1,  # 变异比例,有多少个个体参与变异，小于等于1表示比例，大于1表示固定数量
            'model': {
                'rate': 0.0,  # 模型变异比例
                'range': ''  # 可选的计算模型名称，多个用逗号分开，缺省是netdef中所有模型
            },
            'activation': {
                'rate': 0.0,  # 激活函数的变异比率
                'range': 'sigmod'  # 激活函数的
            },
            'topo': {
                'addnode': 0.4,  # 添加节点的概率
                'addconnection': 0.4,  # 添加连接的概率
                'deletenode': 0.1,  # 删除节点的概率
                'deleteconnection': 0.1  # 删除连接的概率
            },
            'weight': {
                'epoch': 20,  # 权重调整次数
            }
        }

    }

    evolutionTask = EvolutionTask(10, popParam, neat.callbacks.neat_callback)
    evolutionTask.execute(runParam)