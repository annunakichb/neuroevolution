
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
from ne.senal.box import BoxActionConnectionGene
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
            genome = SENetworkGenome(popParam.genomeDefinition)
            genome.id = idGenerator.getNetworkId()

            for index,inputbox in enumerate(netdef.inputboxs):
                boxGene = BoxGene(index,'sensor',**inputbox)
                genome.sensor_box_genes.append(boxGene)
            oid = len(netdef.inputboxs)+1
            for index,outputbox in enumerate(netdef.outputboxs):
                boxGene = BoxGene(index+oid, 'effector',**outputbox)
                genome.receptor_box_genes.append(boxGene)

                # 对每一个输出盒子，随机选择一个输入盒子作为连接，优先选择同组输入
                receptor_box_id = index
                activation_box_ids  = [g.id for g in genome.select_group_sensor(boxGene)]
                activation_box_id = np.random.choice(a=np.array(activation_box_ids), size=1)
                attention_box_ids = [activation_box_id]
                actionConnectionGene = BoxActionConnectionGene(None,attention_box_ids,receptor_box_id)
                genome.action_connection_genes.append(actionConnectionGene)
            genomes.append(genome)

        return genomes
