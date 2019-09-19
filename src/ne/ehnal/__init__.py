import ne
import evolution.session as session
import evolution.agent as agent

from ne.ehnal import EhnalNeuralNetworkGenomeFactory



__all__ = ['EhnalNeuralNetworkGenomeFactory','idgenerator','mutate','selection','ehnal_init']

###############################################################################################
#############################基于关注逻辑的特征神经元网络############################################
####################Feature neuron network based on attention logic############################
###############################################################################################


def ehnal_init():
    # 神经元进化模块初始化
    ne.neuroevolution_init()

    # 注册物种划分对象
    agent.speciesType.register(NeatSpeciesMethod(), 'neat_species')
    # 注册选择操作对象
    session.operationRegistry.register(NeatSelection(), 'neat_selection')
    # 注册交叉操作对象
    session.operationRegistry.register(NeatCrossmateOperation(), 'neat_crossmate')
    # 注册变异操作对象
    session.operationRegistry.register(NeatMutate(), 'neat_mutate')

    # 注册操作顺序图
    session.operationGraphis.register('neat_selection,neat_crossmate,neat_mutate', 'neat')
