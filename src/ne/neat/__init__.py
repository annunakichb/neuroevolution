

__all__ = ['crossmate','idgenerator','mutate','selection','neat_init']

import brain.networks as networks
import evolution.session as session

import utils.collections as collections
from .idgenerator import NeatIdGenerator
from .selection import NeatSelection
from .mutate import NeatMutate
from .crossmate import NeatCrossmateOperation

def neat_init():
    # 设置字典扩展，这样可以将字典的key看成字典对象的成员来操作
    dict.__getattr__ = collections.dict_getattr
    dict.__setattr__ = collections.dict_setattr

    # 注册id生成器对象
    networks.idGenerators.register(NeatIdGenerator(),'neat')
    # 注册选择操作对象
    session.operationRegistry.register(NeatSelection(),'neat_selection')
    # 注册交叉操作对象
    session.operationRegistry.register(NeatCrossmateOperation(),'neat_crossmate')
    # 注册变异操作对象
    session.operationRegistry.register(NeatMutate(),'neat_mutate')

    # 注册操作顺序图
    session.operationGraphis.register('neat_selection,neat_crossmate,neat_mutate','neat')
