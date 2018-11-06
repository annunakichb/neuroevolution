
__all__ = ['NeatIdGenerator']
#无重复的id生成器，
class NeatIdGenerator:
    def __init__(self):
        '''
        无重复的id生成器，网络id按序，同样坐标的神经元id相同，同样连接的突触id相同
        '''
        self.netid = 0
        self.neuronId = 0
        self.synapseId = 0
        self.speciesid = 0
        self.synapseIdcaches = {}
        self.neuronIdcaches = {}

    def getNetworkId(self):
        '''
        取得网络id
        :return:
        '''
        self.netid += 1
        return self.netid

    def getNeuronId(self,net,coord,synapse):
        '''
        取得神经元id
        :param net:
        :param coord:
        :return:
        '''
        # 首先判断是否已有
        for (pos,sid),id in  self.neuronIdcaches.items():
            if pos is not None and coord is not None and pos == coord:return id
            if synapse is not None and sid is not None and synapse.id == sid:return id
        sid = None if synapse is None else synapse.id
        self.neuronId += 1
        self.neuronIdcaches[(coord,sid)] = self.neuronId
        return self.neuronId

    def getSynapseId(self,net,fromId,toId):
        '''
        取得突触id
        :param net:
        :param fromId:
        :param toId:
        :return:
        '''
        if fromId in self.synapseIdcaches.keys():
            t = self.synapseIdcaches[fromId]
            if toId in t.keys():
                return t[toId]

        self.synapseId += 1
        if not fromId in self.synapseIdcaches.keys():
            self.synapseIdcaches[fromId] = {}
        self.synapseIdcaches[fromId][toId] = self.synapseId
        return self.synapseId

    def getSpeciesid(self):
        self.speciesid += 1
        return self.speciesid