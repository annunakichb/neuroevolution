
import evolution.agent as agent

import brain.networks as networks
from brain.elements import  Neuron
from brain.elements import  Module
from brain.networks import NeuralNetwork
from ne import  DefaultNeuralNetworkGenomeFactory

class EhnalNeuralNetworkGenomeFactory(DefaultNeuralNetworkGenomeFactory):
    def create(self,popParam):
        # 取得个体类型，神经网络
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
            for i in range(netdef.neuronCounts[0]):
                # 创建输入神经元
                neuron = Neuron(idGenerator.getNeuronId(),0,0,netdef.models.input)
                # 创建输入神经元的模块
                module_sensor = Module(idGenerator.getModuleId(),0,[],[],[neuron])
                module_sensor.attributes['attention'] = 'sensor'
                # 创建输入神经元的值分布模块
                module_sensor_distribution = Module(idGenerator.getModuleId(),0,[module_sensor],[],[])
                module_sensor_distribution.attributes['attentation'] =
                net.putneuron(neuron)
                net.putmodule(module_sensor)

            self.__createLayerNeuron(net,netdef,0,netdef.neuronCounts[0],netdef.models.input)


            # 创建输出神经元
            self.__createLayerNeuron(net,netdef,len(netdef.neuronCounts)-1,netdef.neuronCounts[-1], netdef.models.output)

            # 创建输入神经元的值分布模块


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

    def _createModule(self,modelId):