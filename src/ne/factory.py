


from brain.networks import NeuralNetwork

import evolution.agent as agent
from evolution.agent import IndividualType



class DefaultNeuralNetworkGenomeFactory:
    def create(self,popParam):
        pass

# 注册ne个体类型
neuralNetworkIndType = IndividualType('network',NeuralNetwork,DefaultNeuralNetworkGenomeFactory(),NeuralNetwork)
agent.individualTypes.register(neuralNetworkIndType,'network')

