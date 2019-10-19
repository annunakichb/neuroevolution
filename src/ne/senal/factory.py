
import  numpy as np
import logging
import evolution.agent as agent

import brain.networks as networks
from brain.elements import  Neuron
from brain.elements import  Module
from brain.networks import NeuralNetwork
from ne import  DefaultNeuralNetworkGenomeFactory
from ne.senal.box import BoxGene
from ne.senal.box import Box
from ne.senal.network import SENetwork
from ne.senal.network import SENetworkGenome

class SENALGenomeFactory(DefaultNeuralNetworkGenomeFactory):
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
            genome = SENetworkGenome()

            for index,inputbox in enumerate(netdef.inputboxs):
                id = idGenerator.getModuleId()
                boxGene = BoxGene(id,inputbox.expression,inputbox.initsize,[],'sensor',inputbox.group,inputbox.attributes)
                genome.box_genes.append(boxGene)

            for index,outputbox in enumerate(netdef.outputboxs):
                id = idGenerator.getModuleId()
                boxGene = BoxGene(id, inputbox.expression, inputbox.initsize, [], 'effector', inputbox.group,
                                  inputbox.attributes)
                genome.box_genes.append(boxGene)

            # 生成一个中间盒子
            ## 随机选择一个输入盒子
            np.random.choice(a=netdef.inputboxs,size=1)
            id = idGenerator.getModuleId()

            hiddleBoxGene = BoxGene(id)
            logging.debug('# SENAL工厂创建网络:'+str(net))

        return genomes
