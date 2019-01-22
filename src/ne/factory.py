
import  logging
import numpy as np
from utils.coords import Coordination
import utils.collections as collections

import brain.networks as networks
from brain.networks import NeuralNetwork
from brain.elements import Neuron


from brain.elements import Synapse

import evolution.agent as agent
from evolution.agent import IndividualType



class DefaultNeuralNetworkGenomeFactory:
    def create(self,popParam):

        indType = agent.individualTypes.find(popParam.indTypeName)
        netdef = popParam.genomeDefinition
        idGenerator = networks.idGenerators.find(netdef.idGenerator)

        genomeFactory = indType.genomeFactory
        if popParam.genomeFactory is not None: genomeFactory = popParam.genomeFactory
        factoryParam = popParam.factoryParam

        size = popParam.size

        genomes = []
        for i in range(size):
            netid = idGenerator.getNetworkId()
            net = NeuralNetwork(netid,netdef)

            # 创建输入神经元
            self.__createLayerNeuron(net,netdef,0,netdef.neuronCounts[0],netdef.models.input)

            # 创建输出神经元
            self.__createLayerNeuron(net,netdef,len(netdef.neuronCounts)-1,netdef.neuronCounts[-1], netdef.models.hidden)

            # 创建中间神经元（如果有）
            hiddenCount = netdef.neuronCounts[1:-1]
            if not collections.isEmpty(hiddenCount):
                for i in range(len(hiddenCount)):
                    self.__createLayerNeuron(net, netdef, i+1, netdef.neuronCounts[i+1], netdef.models.input)

            genomes.append(net)
            # 初始化连接
            if popParam.factoryParam.connectionRate <= 0:         #初始无连接
                continue
            elif popParam.factoryParam.connectionRate >= 1:       #初始全连接
                for i in range(len(net.neurons)-1):
                    for j,n1 in enumerate(net.neurons[i]):       # 取第i层神经元
                        for k,n2 in enumerate(net.neurons[i+1]): # 取第i+1层神经元
                            synapseid = idGenerator.getSynapseId(net,n1.id,n2.id)
                            synapse = Synapse(synapseid,0,n1.id,n2.id,netdef.models.synapse)
                            net.synapses.append(synapse)
            else:                                                # 以一定概率进行连接
                allsynapse = []
                for i in range(len(net.neurons)-1):
                    for j,n1 in enumerate(net.neurons[i]):       # 取第i层神经元
                        for k,n2 in enumerate(net.neurons[i+1]): # 取第i+1层神经元
                            allsynapse.append((n1,n2))
                synapsecount = int(len(allsynapse)*popParam.factoryParam.connectionRate)
                if synapsecount > 0 :
                    indexes = np.random.uniform(0,len(allsynapse)-1,synapsecount)
                    for i in indexes:
                        n1 = allsynapse[i][0]
                        n2 = allsynapse[i][1]
                        synapseid = idGenerator.getSynapseId(net, n1.id, n2.id)
                        synapse = Synapse(synapseid, 0, n1.id, n2.id, netdef.models.synapse)
                        net.synapses.append(synapse)


            logging.debug('# neat工厂创建网络:'+str(net))

        return genomes


    def __createLayerNeuron(self,net,netdef,layerIndex,count,modelConfig):
        if count <= 0:return None

        # 计算实际层编号
        layerInterval = int((netdef.config.range[0][1] - netdef.config.range[0][0])/(len(netdef.neuronCounts)-1))
        layer = 0
        if layerIndex == 0:layer = 0
        elif layerIndex == len(netdef.neuronCounts)-1:layer = netdef.config.range[0][1]
        else: layer = layerIndex * layerInterval

        # 生成神经元坐标
        idGenerator = networks.idGenerators.find(netdef.idGenerator)
        while len(net.neurons) <= layerIndex:
            net.neurons.insert(layerIndex,[])
        coords = self.__createCoord(netdef, layer, count)

        # 创建神经元
        for i in range(count):
            coord = coords[i]
            neuronid = idGenerator.getNeuronId(net, coord, None)
            neuron = Neuron(neuronid, layer, 0, modelConfig, coord)
            net.neurons[layerIndex].append(neuron)

        return net.neurons[layerIndex]



    def __createCoord(self,netdef,layer,count):
        '''
        创建坐标
        :param netdef:  dict 网络定义
        :param layer:   int  层号
        :param count:   int  神经元数量
        :return: list of Coordination
        '''
        width = netdef.config.range[1][1] - netdef.config.range[1][0]
        interval = int(width / (count + 1))

        coords = []
        for i in range(count):
            values = [layer]
            if netdef.config.dimension >= 2:
                values.append((i+1)*interval)
            if netdef.config.dimension >= 3:
                values.append(int((netdef.config.range[2][1] - netdef.config.range[2][0])/2))
            coord = Coordination(*values)
            coords.append(coord)
        return coords
