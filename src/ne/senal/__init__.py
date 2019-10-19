import ne
import  evolution
import evolution.session as session
import evolution.agent as agent
from evolution.agent import IndividualType
from ne.senal.network import SENetwork
from ne.senal.factory import SENALGenomeFactory
from ne.senal.network import SENetworkGenome
from ne.senal.network import SENetworkDecoder
__all__ = ['senal_init']

###############################################################################################
#############################基于关注逻辑的特征神经元网络############################################
####################Feature neuron network based on attention logic############################
###############################################################################################

indTypeName = 'senal_network'
def senal_init():
    # 神经元进化模块初始化
    ne.neuroevolution_init()

    # 注册个体类型
    indType = IndividualType('senal_network', SENetworkGenome, SENALGenomeFactory(), SENetwork,SENetworkDecoder())
    neuralNetworkIndType = agent.individualTypes.register(indType, indTypeName)
    evolution.set_default_popparam(name='indTypeName',defaultvalue=indTypeName)


