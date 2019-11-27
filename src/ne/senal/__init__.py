
import  evolution
import evolution.session as session
import evolution.agent as agent
from evolution.session import Evaluator
from evolution.agent import IndividualType
from ne.senal.network import SENetwork
from ne.senal.factory import SENALGenomeFactory
from ne.senal.network import SENetworkGenome
from ne.senal.network import SENetworkDecoder
from ne.senal.operations import SelectionOperation
from ne.senal.operations import MutateOperation
import ne.senal.activity
import ne as ne
import brain.networks
from ne.senal.network import SenalIdGenerator

#__all__ = ['senal_init']

###############################################################################################
#############################基于关注逻辑的特征神经元网络############################################
####################Feature neuron network based on attention logic############################
###############################################################################################

indTypeName = 'senal_network'
def senal_init():
    # 神经元进化模块初始化
    ne.neuroevolution_init()

    # 创建缺省的id生成器
    idGenerator = SenalIdGenerator()
    brain.networks.idGenerators.register(idGenerator, 'senal')
    brain.set_default_netdef_info('idGenerator','senal')

    # 注册个体类型
    indType = IndividualType('senal_network', SENetworkGenome, SENALGenomeFactory(), SENetwork,SENetworkDecoder())
    neuralNetworkIndType = agent.individualTypes.register(indType, indTypeName)
    evolution.set_default_popparam(name='indTypeName',defaultvalue=indTypeName)

    # 注册进化操作算子
    session.operationRegistry.register(SelectionOperation(), 'senal_selection')
    session.operationRegistry.register(MutateOperation(), 'senal_mutate')
    evolution.set_default_runparam('operations','senal_selection,senal_mutate')

    # 注册缺省活动过程评估
    evaluator = Evaluator('fitness',[(ne.senal.activity.do_activity,1.0)])
    evolution.set_default_popparam('features',{'fitness':evaluator})
