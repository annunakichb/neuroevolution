import numpy as np
from scipy.cluster.vq import vq,kmeans,whiten

import utils.collections as collections
from utils.cluster import KMean

import brain.networks as networks
from evolution.agent import Specie

class NeatSpeciesMethod:
    def __init__(self):
        pass

    def execute(self,session):
        n_clusters = session.popParam.species.size
        method = session.popParam.species.method
        iter = session.popParam.species.iter
        alg = session.popParam.species.get('alg','kmean')

        # 取得所有个体的特征向量
        idgenerator = networks.idGenerators.find(session.popParam.genomeDefinition.idGenerator)
        inds = session.pop.inds
        indVecs = list(map(lambda ind:self.__getIndVector(ind,session),inds))
        indArray = whiten(np.array(indVecs))
        if alg == 'kmean':
            centroids,distortion = kmeans(obs=indArray,k_or_guess=n_clusters,iter=iter)
            labels = vq(indArray, centroids)
            species = []
            for index,center in enumerate(centroids):
                species_inds = [ind for i,ind in enumerate(inds) if i in np.argwhere(labels[0]==index)]
                if collections.isEmpty(species_inds):continue
                specieId = idgenerator.getSpeciesid(list(map(lambda ind:ind.getPhenome(),species_inds)))
                for ind in species_inds:
                    ind.speciedId = specieId
                sp = Specie(id=specieId,inds = species_inds,pop = session.pop)
                species.append(sp)
            session.pop.species = species

            return session.pop.species
        else:
            raise RuntimeError('物种分类算法名称无效(popParam.species.alg):'+alg)


    def __getIndVector(self,ind,session):
        '''
        对每个个体，生成个体特征向量
        :param ind:
        :param session:
        :return:
        '''
        # 取得所有生成过的神经元id和网络id
        net = ind.getPhenome()
        inputNeurons = net.getInputNeurons()
        idgenerator = networks.idGenerators.find(session.popParam.genomeDefinition.idGenerator)
        nids = [id for id in idgenerator.getAllCacheNeuronIds() if net.getNeuron(id) not in inputNeurons]
        sids = idgenerator.getAllCacheSynasesIds()

        # 取得每个神经元的偏置和每个突触的权重
        nbiases = list(map(lambda nid:0 if net.getNeuron(nid) is None else net.getNeuron(nid).getVariableValue('bias',np.array([0.0])).tolist()[0],nids))
        sweights = list(map(lambda sid:0 if net.getSynapse(id=sid) is None else net.getSynapse(id=sid)['weight'].tolist()[0],sids))
        return nbiases + sweights



    def _compute_distance(self, ind1,ind2, session):
        # 距离计算，这段代码参考了https://github.com/CodeReclaimers/neat-python

        # 取得距离计算参数
        disjoint_coefficient = session.popParam.species.disjoint_coefficient
        weight_coefficient = session.popParam.species.weight_coefficient

        # 取得个体神经网络和所有神经元以及所有突触
        net1 = ind1.getPhenome()
        net2 = ind2.getPhenome()

        ns1 = net1.getNeurons()
        ns2 = net2.getNeurons()
        if collections.isEmpty(ns1):ns1 = []
        if collections.isEmpty(ns2):ns2 = []

        sps1 = net1.getSynapses()
        sps2 = net2.getSynapses()
        if collections.isEmpty(sps1): sps1 = []
        if collections.isEmpty(sps2): sps2 = []

        # 计算节点差异
        disjoint_nodes = 0
        node_distance = 0.0
        for n1 in ns1:
            if net2.getNeuron(n1.id) is None:
                disjoint_nodes += 1

        for n2 in ns2:
            n1 = net1.getNeuron(n2.id)
            if n1 is None:
                disjoint_nodes += 1
            else:
                d = abs(n1.getVariableValue('bias',0.0) - n2.getVariableValue('bias',0.0)) + abs(n1['value'] - n2['value'])
                if n1['activation'] != n2['activation']:d += 1.0

                d = d * weight_coefficient
                node_distance += d


        max_nodes = max(len(ns1), len(ns2))
        node_distance = (node_distance + disjoint_coefficient * disjoint_nodes)/max_nodes


        # 计算突触差异
        disjoint_connections = 0
        connection_distance = 0.0
        for s1 in sps1:
            if net2.getSynapse(id=s1.id) is None:
                disjoint_connections += 1
        for s2 in sps2:
            s1 = net1.getSynapse(id = s2.id)
            if s1 is None:
                disjoint_connections += 1
            else:
                d = abs(s1['weight'] - s2['weight'])
                d = d * weight_coefficient
                connection_distance += d


        max_conn = max(len(sps1), len(sps2))
        connection_distance = (connection_distance + disjoint_coefficient * disjoint_connections)/max_conn if max_conn>0 else 0.0

        # 计算总距离
        distance = node_distance + connection_distance
        return distance