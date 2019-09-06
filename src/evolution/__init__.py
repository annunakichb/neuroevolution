# -*- coding: UTF-8 -*-

from utils.properties import Properties
from evolution.env import Evaluator
from evolution.agent import IndividualType
__all__ = ['agent','env','montior','session','createPopParam','createRunParam']

def evolution_init():
    '''
    进化初始化，必须首先调用初始化函数
    :return:
    '''
    dictIndType = IndividualType('dict', dict, None, dict)
    agent.individualTypes.register(dictIndType, 'dict')

def createPopParam(**kwargs):
    if kwargs is None : kwargs = {}
    if 'size' not in kwargs: raise RuntimeError('lost size in createPopParam')
    if 'elitistSize' not in kwargs: raise RuntimeError('lost elitistSize in createPopParam')
    # 定义种群
    popParam = {
        'indTypeName': 'dict'  if 'indTypeName' not in kwargs else kwargs['indTypeName'],  # 种群的个体基因类型名，必须，该类型的个体基因应已经注册过，参见evolution.agent,必须
        'genomeFactory': None if 'genomeFactory' not in kwargs else kwargs['genomeFactory'],  # 基因工厂，个体类型中已经提供了基因工厂对象，这里如果设置，可以替换前者，可选
        'factoryParam': {'connectionRate':1.0} if 'factoryParam' not in kwargs else kwargs['factoryParam'],
        'genomeDefinition': None if 'genomeDefinition' not in kwargs else kwargs['genomeDefinition'],  # 基因定义参数,可选
        'size': kwargs['size'],  # 种群大小，必须
        'elitistSize': 0.1 if 'elitistSize' not in kwargs else kwargs['elitistSize'],  # 精英个体占比，小于1表示比例，大于等于1表示数量
        'species': {  # 物种参数，可选
            'method': '' if 'speciesMethod' not in kwargs else kwargs['speciesMethod'],  # 物种分类方法,在物种参数中必须
            'params': {} if 'speciesParams' not in kwargs else kwargs['speciesParams']
        }
    }
    evaluators = None if 'evaluators' not in kwargs else kwargs['evaluators']
    if evaluators is None:
        return Properties(popParam)
    if callable(evaluators):
        popParam['features'] =  {  # 特征评估函数配置，必须
            'fitness': Evaluator('fitness', [(evaluators, 1.0)]) if 'evaluator' not in kwargs else kwargs['evaluator']# 适应度评估器,如果评估器只包含一个函数,也可以写成Evaluator('fitness',fitness)
        }
    elif isinstance(evaluators,list):
        for evaluator in evaluators:
            popParam['features'][evaluator.key] = evaluator

    return Properties(popParam)


def createRunParam(maxIterCount=10000,maxFitness=0,operations='neat_selection,neat_crossmate,neat_mutate',**kwargs):
    if kwargs is None:kwargs = {}
    # 定于运行参数
    runParam = {
        'terminated': {
            'maxIterCount': maxIterCount,  # 最大迭代次数，必须
            'maxFitness': maxFitness,  # 最大适应度，必须
        },
        'log': {
            'individual': 'elite' if 'logindividual' not in kwargs else kwargs['logindividual'],  # 日志中记录个体方式：记录所有个体，可以选择all,elite,maxfitness（缺省）,custom
            'debug': False if 'debug' not in kwargs else kwargs['debug'],  # 是否输出调试信息
            'file': 'neat_cartpole.log' if 'logfile' not in kwargs else kwargs['logfile']  # 日志文件名
        },
        'evalate': {
            'parallel': 0  ,  # 并行执行评估的线程个数，缺省0，可选
        },
        'operations': {
            # 'method' : 'neat',                                   # 已有的进化操作序列名称，与text两个只用一个
            'text': operations  # 进化操作序列
        },
        'mutate': {
            'propotion': 0.1 if 'mutate_propotion' not in kwargs else kwargs['mutate_propotion'],  # 变异比例,有多少个个体参与变异，小于等于1表示比例，大于1表示固定数量
            'parallel': 0 if 'mutate_parallel' not in kwargs else kwargs['mutate_parallel'],  # 并行执行变异的线程个数，缺省0，可选
            'model': {
                'rate': 0.0 if 'mutate_model_rate' not in kwargs else kwargs['mutate_model_rate'],  # 模型变异比例
                'range': '' if 'mutate_model_range' not in kwargs else kwargs['mutate_model_range'] # 可选的计算模型名称，多个用逗号分开，缺省是netdef中所有模型
            },
            'activation': {
                'rate': 0.0 if 'mutate_activation_rate' not in kwargs else kwargs['mutate_activation_rate'],  # 激活函数的变异比率
                'range': 'sigmod' if 'mutate_activation_range' not in kwargs else kwargs['mutate_activation_range']  # 激活函数的
            },
            'topo': {
                'addnode': 0.4  if 'mutate_addnode' not in kwargs else kwargs['mutate_addnode'],  # 添加节点的概率
                'addconnection': 0.4 if 'mutate_addconnection' not in kwargs else kwargs['mutate_addconnection'],  # 添加连接的概率
                'deletenode': 0.1  if 'mutate_deletenode' not in kwargs else kwargs['mutate_deletenode'],  # 删除节点的概率
                'deleteconnection': 0.1  if 'mutate_deleteconnection' not in kwargs else kwargs['mutate_deleteconnection']  # 删除连接的概率
            },
            'weight': {
                'method':'nes',
                'parallel': 0 if 'weight_parallel' not in kwargs else kwargs['weight_parallel'],  # 并行执行权重变异的线程个数，缺省0，可选
                'epoch': 1 if 'weight_epoch' not in kwargs else kwargs['weight_epoch'],  # 权重调整次数
            }
        }

    }
    return Properties(runParam)
