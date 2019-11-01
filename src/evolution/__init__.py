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
    # 注册字典为个体类型
    dictIndType = IndividualType('dict', dict, None, dict)
    agent.individualTypes.register(dictIndType, 'dict')

#region 种群参数元信息管理
# 种群参数元信息
pop_param_info = {
    'size':PropertyInfo(1, 'size', int, catalog='popparam.size', required='necessary', default=100,range='(0,Nan)', desc='种群规模'),
    'elitistSize':PropertyInfo(2, 'elitistSize', float, catalog='popparam.elitistSize', required='necessary', default=0.2, desc='种群规模'),
    'indTypeName':PropertyInfo(3, 'indTypeName', str, catalog='popparam.indTypeName', required='optional', default='dict', desc='种群的个体基因类型名，该类型的个体基因应已经注册过，参见evolution.agent'),
    'genomeFactory':PropertyInfo(4, 'genomeFactory', object, catalog='popparam.genomeFactory', required='optional', desc=' 基因工厂，个体类型中已经提供了基因工厂对象，这里如果设置，可以替换前者'),
    'connectionRate':PropertyInfo(5, 'connectionRate', float, catalog='popparam.factoryParam.connectionRate', required='optional', default=1.0, desc='初始连接比率'),
    'genomeDefinition':PropertyInfo(6, 'genomeDefinition', object, catalog='popparam.genomeDefinition', required='optional',default=None, desc='基因定义参数'),
    'species_method':PropertyInfo(7, 'species_method', str, catalog='popparam.species.method', required='optional',default='', desc='物种分类方法'),
    'species_alg':PropertyInfo(8, 'species_alg', str, catalog='popparam.species.alg', required='optional',default='', desc='物种分类算法'),
    'species_size':PropertyInfo(9, 'species_size', int, catalog='popparam.species.size', required='optional',default=6, desc='物种最大数量'),
    'species_iter':PropertyInfo(10, 'species_iter', int, catalog='popparam.species.iter', required='optional',default=50, desc='物种算法迭代次数'),
    'features':PropertyInfo(11,'features', dict, catalog='popparam.features', required='optional',default={}, desc='适应度评估器,如果评估器只包含一个函数,也可以写成Evaluator(''fitness'',fitness)'),
}
pop_param_info = Properties(pop_param_info)

def print_popparam():
    '''
    打印所有种群参数元信息
    :return: None
    '''
    for name,pi in pop_param_info.items():
        print(str(pi))

def set_default_popparam(name,defaultvalue):
    '''
    设置种群参数缺省值
    :param name:         str  种群参数名或者别名
    :param defaultvalue: Any  缺省值
    :return:
    '''
    for pname,pi in pop_param_info.items():
        if pi.nameInfo.name == name or pi.alias == name:
            pi.default = defaultvalue
            return


def createPopParam(**kwargs):
    '''
    创建种群参数
    :param kwargs:  dict  参见pop_param_info
    :return: Properties   种群参数字典对象
    '''
    return Properties.create_from_dict(pop_param_info,**kwargs)


# 运行参数信息
run_params_info = {
        'maxIterCount':PropertyInfo(1,'maxIterCount',int,catalog='runparam.terminated.maxIterCount',required='necessary',default=100,range='(0,Nan)',desc='最大迭代次数'),
        'maxFitness':PropertyInfo(2, 'maxFitness', int,catalog='runparam.terminated.maxFitness',required='necessary',default=1.,range='[0,Nan)', desc='最大适应度'),
        'operations':PropertyInfo(3, 'operations', str,catalog='runparam.operations',required='optional',default='', desc='进化操作名称,多个用逗号分开'),
        'logindividual':PropertyInfo(4, 'logindividual',str,catalog='runparam.log.individual',required='optional',default='elite', desc='是否记录个体信息到日志(这会使日志很大)'),
        'debug':PropertyInfo(5, 'debug',bool,catalog='runparam.log.debug',required='optional',default=False, desc='是否记录调试信息'),
        'logfile':PropertyInfo(6, 'logfile', str, catalog='runparam.log.file', required='optional',default='evolution.log', desc='是否记录个体信息到日志(这会使日志很大)'),
        'parallel':PropertyInfo(7, 'parallel', bool, catalog='runparam.evaulate.parallel', required='optional', default=False,desc='启动并行评估'),
        'mutate_propotion':PropertyInfo(8, 'mutate_propotion', float, catalog='runparam.operation.mutate.propotion', required='optional', default=0.01,desc='变异比例'),
        'mutate_parallel':PropertyInfo(9, 'mutate_parallel',bool, catalog='runparam.mutate.parallel', required='optional',default=False, desc='启动并行变异过程'),
        'mutate_model_rate':PropertyInfo(10, 'mutate_model_rate', float, catalog='runparam.mutate.model.rate', required='optional',default=0.0, desc='计算模型也参与变异的概率'),
        'mutate_model_range':PropertyInfo(11, 'mutate_model_range',str , catalog='runparam.mutate.model.range', required='optional',default='', desc='可选的计算模型名称，多个用逗号分开，缺省是netdef中所有模型'),
        'mutate_activation_rate':PropertyInfo(12, 'mutate_activation_rate', float, catalog='runparam.mutate.activation.rate', required='optional',default=0.0, desc='激活函数的变异比率'),
        'mutate_activation_range':PropertyInfo(13, 'mutate_activation_range', str, catalog='runparam.mutate.activation.range', required='optional',default='sigmod', desc='参与变异的激活函数名称'),
        'mutate_addnode':PropertyInfo(14, 'mutate_addnode', float, catalog='runparam.mutate.topo.addnode', required='optional',default=0.4, desc='添加节点的概率'),
        'mutate_addconnection':PropertyInfo(15, 'mutate_addconnection', float, catalog='runparam.mutate.topo.addconnection', required='optional',default=0.4, desc='添加连接的概率'),
        'mutate_deletenode':PropertyInfo(16, 'mutate_deletenode', float, catalog='runparam.mutate.topo.deletenode', required='optional',default=0.1, desc='删除节点的概率'),
        'mutate_deleteconnection':PropertyInfo(17, 'mutate_deleteconnection', float, catalog='runparam.mutate.topo.deleteconnection', required='optional',default=0.1, desc='删除连接的概率'),
        'weight_method':PropertyInfo(18, 'weight_method', str, catalog='runparam.mutate.weight.method', required='optional',default='nes', desc='修正权重的方法'),
        'weight_parallel':PropertyInfo(19, 'weight_parallel', bool, catalog='runparam.mutate.weight.parallel', required='optional',default=False, desc='是否并行修正权重'),
        'weight_epoch':PropertyInfo(20, 'weight_epoch', int, catalog='runparam.mutate.weight.epoch', required='optional',default=3, desc='权重修改算法执行轮次')
}
run_params_info = Properties(run_params_info)

def print_runparam():
    '''
    打印所有种群参数元信息
    :return: None
    '''
    for name,pi in run_params_info.items():
        print(str(pi))

def set_default_runparam(name,defaultvalue):
    '''
    设置种群参数缺省值
    :param name:         str  种群参数名或者别名
    :param defaultvalue: Any  缺省值
    :return:
    '''
    for pname,pi in run_params_info.items():
        if pi.nameInfo.name == name or pi.alias == name:
            pi.default = defaultvalue
            return

def createRunParam(maxIterCount,maxFitness,**kwargs):
    '''
    创建运行参数
    :param maxIterCount:  int    最大迭代次数
    :param maxFitness:    float  最大适应度
    :param kwargs:        dict   其他参数，参见run_params_info
    :return:
    '''
    kwargs['maxIterCount'] = maxIterCount
    kwargs['maxFitness'] = maxFitness
    return Properties.create_from_dict(run_params_info,**kwargs)
