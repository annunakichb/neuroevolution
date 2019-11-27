# -*- coding: UTF-8 -*-
from utils.properties import PropertyInfo
from brain.runner import NeuralNetworkTask

__all__=['activation','elements','models','networks','runner']

def brain_init():
    pass

from utils.properties import Properties
from brain.networks import NetworkType
from brain.networks import NeuralNetwork

netdef_info = {
    'netType':PropertyInfo(1, 'netType', networks.NetworkType, catalog='netdef.netType', required='optional', default=networks.NetworkType.Perceptron,desc='网络类型'),
    'idGenerator':PropertyInfo(2, 'idGenerator', object, catalog='netdef.idGenerator', required='optional', default=networks.DefaultIdGenerator(), desc='ID生成器'),
    'layered':PropertyInfo(3, 'layered', bool, catalog='netdef.config.layered', required='optional', default=True, desc='分层'),
    'substrate':PropertyInfo(4, 'substrate', bool, catalog='netdef.config.substrate', required='optional',default=False, desc='基座'),
    'acycle':PropertyInfo(5, 'acycle', bool, catalog='netdef.config.acycle', required='optional', default=False, desc='允许自连接'),
    'recurrent':PropertyInfo(6, 'recurrent', bool, catalog='netdef.config.recurrent', required='optional',default=False, desc='允许循环'),
    'reversed':PropertyInfo(7, 'reversed', str, catalog='netdef.config.reversed', required='optional',default=False, desc='允许反向'),
    'dimension':PropertyInfo(8, 'dimension', str, catalog='netdef.config.dimension', required='optional',default=0, desc='坐标维度'),
    'range':PropertyInfo(9, 'range', list, catalog='netdef.config.range', required='optional',default=[], desc='有效坐标范围'),
    'runnername':PropertyInfo(10, 'runnername', str, catalog='netdef.config.runnername', required='optional',default='', desc='运行器名称'),
    'models':PropertyInfo(11,'models', dict, catalog='netdef.models', required='optional',default={}, desc='计算模型'),
    'task':PropertyInfo(12,'tasks', NeuralNetworkTask, catalog='netdef.task', required='optional',default=None, desc='计算任务')
}
def print_netdef_info():
    '''
    打印所有种群参数元信息
    :return: None
    '''
    for name,pi in netdef_info.items():
        print(str(pi))

def set_default_netdef_info(name,defaultvalue):
    '''
    设置种群参数缺省值
    :param name:         str  种群参数名或者别名
    :param defaultvalue: Any  缺省值
    :return:
    '''
    for pname,pi in netdef_info.items():
        if pi.nameInfo.name == name or pi.alias == name:
            pi.default = defaultvalue
            return

def createNetDef(**kwargs):
    return Properties.create_from_dict(netdef_info, **kwargs)
