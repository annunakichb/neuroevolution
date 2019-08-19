from brain.networks import  *
from utils.properties import Properties
from evolution.env import Evaluator

def createNetDef(task,**kwargs):
    '''
    执行
    :return:
    '''

    if kwargs is None: kwargs = {}
    if 'neuronCounts' not in kwargs:raise RuntimeError('can not find neuronCounts in createNetDef ')
    # 定义网络
    netdef = {
        'netType' : NetworkType.Perceptron if 'netType' not in kwargs else  kwargs['netType'],                       # NetworkType，网络类型,必须
        'neuronCounts' : kwargs['neuronCounts'],                                                                     # list（初始）网络各层神经元数量,必须
        'idGenerator' :  'neat' if 'idGenerator' not in kwargs else  kwargs['idGenerator'],                          # str 生成网络，神经元，突触id的类，参见DefauleIDGenerator,list idgenerator命令可以列出所有的id生成器对象
        'config' : {
            'layered' : True if 'layered' not in kwargs else  kwargs['layered'],                                     # bool 是否分层,可选
            'substrate' : True if 'substrate' not in kwargs else  kwargs['substrate'],                               # bool 是否使用基座,可选
            'acyclie' : False if 'acyclie' not in kwargs else  kwargs['acyclie'],                                    # bool 是否允许自身连接,可选
            'recurrent':False if 'recurrent' not in kwargs else  kwargs['recurrent'],                                # bool 是否允许同层连接,可选
            'reversed':False if 'reversed' not in kwargs else  kwargs['reversed'],                                   # bool 是否允许反向连接,可选
            'dimension':2  if 'dimension' not in kwargs else  kwargs['dimension'],                                   # int 空间坐标维度,可选
            'range':NeuralNetwork.MAX_RANGE if 'dimension' not in kwargs else  kwargs['dimension'],                  # list 坐标范围，可选'
        },
        'runner':{
            'name' : 'simple' if 'runnername' not in kwargs else  kwargs['runnername'] ,                             # str 网络运行器名称,必须
            'task' : task,                                        # NeuralNetworkTask,网络运行任务,必须
        },
        'models':kwargs['models'] if 'models' not in kwargs else {                                                # dict 神经元计算模型的配置信息,必须
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
    return Properties(netdef)

def createPopParam(netdef,fitness,**kwargs):
    if kwargs is None : kwargs = {}
    if 'size' not in kwargs: raise RuntimeError('lost size in createPopParam')
    if 'elitistSize' not in kwargs: raise RuntimeError('lost elitistSize in createPopParam')
    # 定义种群
    popParam = {
        'indTypeName': 'network'  if 'indTypeName' not in kwargs else kwargs['indTypeName'],  # 种群的个体基因类型名，必须，该类型的个体基因应已经注册过，参见evolution.agent,必须
        'genomeFactory': None if 'genomeFactory' not in kwargs else kwargs['genomeFactory'],  # 基因工厂，个体类型中已经提供了基因工厂对象，这里如果设置，可以替换前者，可选
        'factoryParam': {  # 工厂参数，必须
            'connectionRate': 1.0 if 'connectionRate' not in kwargs else kwargs['connectionRate'],  # 连接比率
        },
        'genomeDefinition': netdef,  # 基因定义参数,可选
        'size': kwargs['size'],  # 种群大小，必须
        'elitistSize': kwargs['elitistSize'],  # 精英个体占比，小于1表示比例，大于等于1表示数量
        'species': {  # 物种参数，可选
            'method': 'neat_species',  # 物种分类方法,在物种参数中必须
            'alg': 'kmean',  # 算法名称
            'size': 5,  # 物种个体数量限制，0表示无限制或动态
            'iter': 50,  # 算法迭代次数
        },
        'features': {  # 特征评估函数配置，必须
            'fitness': Evaluator('fitness', [(fitness, 1.0)]) if 'evaluator' not in kwargs else kwargs['evaluator']# 适应度评估器,如果评估器只包含一个函数,也可以写成Evaluator('fitness',fitness)
        }
    }
    return Properties(popParam)


def createRunParam(maxIterCount=10000,maxFitness=10000,**kwargs):
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
            'text': 'neat_selection,neat_crossmate,neat_mutate' if 'operations' not in kwargs else kwargs['operations']  # 进化操作序列
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
                'parallel': 0 if 'weight_parallel' not in kwargs else kwargs['weight_parallel'],  # 并行执行权重变异的线程个数，缺省0，可选
                'epoch': 1 if 'weight_epoch' not in kwargs else kwargs['weight_epoch'],  # 权重调整次数
            }
        }

    }
    return Properties(runParam)
