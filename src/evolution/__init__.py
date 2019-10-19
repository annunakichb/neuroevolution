# -*- coding: UTF-8 -*-


from utils.properties import Properties
from evolution.env import Evaluator
from evolution.agent import IndividualType
from utils.properties import PropertyInfo
__all__ = ['agent','env','montior','session','createPopParam','createRunParam']

def evolution_init():
    '''
    进化初始化，必须首先调用初始化函数
    :return:
    '''
    dictIndType = IndividualType('dict', dict, None, dict)
    agent.individualTypes.register(dictIndType, 'dict')

pop_param_info = {
    'size':PropertyInfo(1, 'size', int, alias='popparam.size', required='necessary', default=100,range='(0,Nan)', desc='种群规模'),
    'elitistSize':PropertyInfo(2, 'elitistSize', float, alias='popparam.elitistSize', required='necessary', default=0.2, desc='种群规模'),
    'indTypeName':PropertyInfo(3, 'indTypeName', str, alias='popparam.indTypeName', required='optional', default='dict', desc='种群的个体基因类型名，该类型的个体基因应已经注册过，参见evolution.agent'),
    'genomeFactory':PropertyInfo(4, 'genomeFactory', object, alias='popparam.genomeFactory', required='optional', desc=' 基因工厂，个体类型中已经提供了基因工厂对象，这里如果设置，可以替换前者'),
    'connectionRate':PropertyInfo(5, 'connectionRate', float, alias='popparam.factoryParam.connectionRate', required='optional', default=1.0, desc='初始连接比率'),
    'genomeDefinition':PropertyInfo(6, 'genomeDefinition', object, alias='popparam.factoryParam.genomeDefinition', required='optional',default=None, desc='基因定义参数'),
    'species_method':PropertyInfo(7, 'species_method', str, alias='popparam.species.method', required='optional',default='', desc='物种分类方法'),
    'species_alg':PropertyInfo(8, 'species_alg', str, alias='popparam.species.alg', required='optional',default='', desc='物种分类算法'),
    'species_size':PropertyInfo(9, 'species_size', int, alias='popparam.species.size', required='optional',default=6, desc='物种最大数量'),
    'species_iter':PropertyInfo(10, 'species_iter', int, alias='popparam.species.iter', required='optional',default=50, desc='物种算法迭代次数'),
    'evaluators':PropertyInfo(11,'evaluators', object, alias='popparam.evaluators', required='optional',default=None, desc='适应度评估器,如果评估器只包含一个函数,也可以写成Evaluator(''fitness'',fitness)'),
}
def print_popparam():
    for name,pi in pop_param_info.items():
        print(str(pi))
def set_default_popparam(name,defaultvalue):
    for pname,pi in pop_param_info.items():
        if pi.nameInfo.name == name or pi.alias == name:
            pi.default = defaultvalue
            return


def createPopParam(**kwargs):
    if kwargs is None : kwargs = {}
    if 'size' not in kwargs: raise RuntimeError('lost size in createPopParam')
    # 定义种群
    popParam = {
        'indTypeName': pop_param_info['indTypeName'].default  if 'indTypeName' not in kwargs else kwargs['indTypeName'],  # 种群的个体基因类型名，必须，该类型的个体基因应已经注册过，参见evolution.agent,必须
        'genomeFactory': pop_param_info['genomeFactory'].default if 'genomeFactory' not in kwargs else kwargs['genomeFactory'],  # 基因工厂，个体类型中已经提供了基因工厂对象，这里如果设置，可以替换前者，可选
        'factoryParam': {'connectionRate':pop_param_info['connectionRate'].default} if 'factoryParam' not in kwargs else kwargs['factoryParam'],
        'genomeDefinition': pop_param_info['genomeDefinition'].default if 'genomeDefinition' not in kwargs else kwargs['genomeDefinition'],  # 基因定义参数,可选
        'size': kwargs['size'],  # 种群大小，必须
        'elitistSize': pop_param_info['elitistSize'].default if 'elitistSize' not in kwargs else kwargs['elitistSize'],  # 精英个体占比，小于1表示比例，大于等于1表示数量
        'species': {
            'species_method':pop_param_info['species_method'].default if 'species_method' not in kwargs else kwargs['species_method'],
            'species_alg': pop_param_info['species_alg'].default if 'species_alg' not in kwargs else kwargs['species_alg'],
            'species_size': pop_param_info['species_size'].default if 'species_size' not in kwargs else kwargs['species_size'],
            'species_iter': pop_param_info['species_iter'].default if 'species_iter' not in kwargs else kwargs['species_iter']

        }
    }
    evaluators = None if 'evaluators' not in kwargs else kwargs['evaluators']
    if evaluators is None:
        return Properties(popParam)
    if callable(evaluators):
        popParam['features'] =  {  # 特征评估函数配置，必须
            'fitness': Evaluator('fitness', [(evaluators, 1.0)]) if 'evaluator' not in kwargs else kwargs['evaluator']# 适应度评估器,如果评估器只包含一个函数,也可以写成Evaluator('fitness',fitness)
        }
    elif isinstance(evaluators, list):
        popParam['features'] = {}
        for evaluator in evaluators:
            popParam['features'][evaluator[2]] = Evaluator(evaluator[2],[(evaluator[0],evaluator[1])])
    elif isinstance(evaluators,tuple):
        popParam['features'] = {}
        for key,evaluator in evaluators.items():
            popParam['features'][key] = Evaluator('key',[(evaluator[0],evaluator[1])])

    return Properties(popParam)

# 运行参数信息
run_params_info = [
        PropertyInfo(1,'maxIterCount',int,alias='runparam.maxIterCount',required='necessary',default=100,range='(0,Nan)',desc='最大迭代次数'),
        PropertyInfo(2, 'maxFitness', int,alias='runparam.maxFitness',required='necessary',default=1.,range='[0,Nan)', desc='最大适应度'),
        PropertyInfo(3, 'operations', str,alias='runparam.operations',required='optional',default='', desc='进化操作名称,多个用逗号分开'),
        PropertyInfo(4, 'logindividual',bool,alias='runparam.log.individual',required='optional',default=False, desc='是否记录个体信息到日志(这会使日志很大)'),
        PropertyInfo(5, 'logfile', str, alias='runparam.log.file', required='optional',default='evolution.log', desc='是否记录个体信息到日志(这会使日志很大)'),
        PropertyInfo(6, 'parallel', bool, alias='runparam.evalate.parallel', required='optional', default=False,desc='启动并行评估'),
        PropertyInfo(7, 'mutate_propotion', float, alias='runparam.operation.mutate.propotion', required='optional', default=0.01,desc='变异比例'),
        PropertyInfo(8, 'mutate_parallel',bool, alias='runparam.operation.mutate.parallel', required='optional',default=False, desc='启动并行变异过程'),
        PropertyInfo(9, 'mutate_model_rate', float, alias='runparam.operation.mutate.model.rate', required='optional',default=0.0, desc='计算模型也参与变异的概率'),
        PropertyInfo(10, 'mutate_model_range',str , alias='runparam.operation.mutate.model.range', required='optional',default='', desc='可选的计算模型名称，多个用逗号分开，缺省是netdef中所有模型'),
        PropertyInfo(11, 'mutate_activation_rate', float, alias='runparam.operation.mutate.activation.rate', required='optional',default=0.0, desc='激活函数的变异比率'),
        PropertyInfo(12, 'mutate_activation_range', str, alias='runparam.operation.mutate.activation.range', required='optional',default='sigmod', desc='参与变异的激活函数名称'),
        PropertyInfo(13, 'mutate_addnode', float, alias='runparam.operation.mutate.topo.addnode', required='optional',default=0.4, desc='添加节点的概率'),
        PropertyInfo(14, 'mutate_addconnection', float, alias='runparam.operation.mutate.topo.addconnection', required='optional',default=0.4, desc='添加连接的概率'),
        PropertyInfo(14, 'mutate_deletenode', float, alias='runparam.operation.mutate.topo.deletenode', required='optional',default=0.1, desc='删除节点的概率'),
        PropertyInfo(15, 'mutate_deleteconnection', float, alias='runparam.operation.mutate.topo.deleteconnection', required='optional',default=0.1, desc='删除连接的概率'),
        PropertyInfo(16, 'weight_method', str, alias='runparam.operation.mutate.weight.method', required='optional',default='nes', desc='修正权重的方法'),
        PropertyInfo(17, 'weight_parallel', bool, alias='runparam.operation.mutate.weight.parallel', required='optional',default=False, desc='是否并行修正权重'),
        PropertyInfo(18, 'weight_epoch', int, alias='runparam.operation.mutate.weight.epoch', required='optional',default=3, desc='权重修改算法执行轮次')
]

def printRunParam():
    '''
    打印缺省运行参数
    :return:
    '''
    for pi in run_params_info:
        print(pi.alias,' required=',pi.required,' type=',str(pi.type),' default=',str(pi.default))

def set_default_runparam(xh,name,defaultvalue):
    for pi in run_params_info:
        if xh == pi.xh or pi.nameInfo.name == name or pi.alias == name:
            pi.default = defaultvalue
            return

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
            'file': 'evolution.log' if 'logfile' not in kwargs else kwargs['logfile']  # 日志文件名
        },
        'evalate': {
            'parallel': 0 if 'parallel' not in kwargs else kwargs['parallel'],  # 并行执行评估的线程个数，缺省0，可选
        },
        'operations': {
            # 'method' : 'neat',                                   # 已有的进化操作序列名称，与text两个只用一个
            'text': operations  # 进化操作序列
        },
        'selection': kwargs['selection'] if 'selection' in kwargs else {},
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

    for key,value in kwargs.items():
        if key in [pi.nameInfo.name for pi in run_params_info]:continue
        runParam[key] = value

    return Properties(runParam)
