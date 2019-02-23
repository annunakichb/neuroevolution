import  utils.collections as collections
from functools import reduce

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
        self.synapseIdCounts = {}
        self.neuronIdcaches = {}
        self.neuronIdCounts = {}
        self.speciesIdcaches = {}

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
        :param net:    NeuralNetwork 所属网络
        :param coord:  Coordination 坐标
        :param synapse: Synapse 突触，表示该神经元是从哪个突触上分裂出来的
        :return: 不管哪个网络，coord相同的神经元id也一定相同，如果coord无效，则synapse.id相同的神经元id也一定相同
        '''

        # 判断是有已有
        if coord is not None:
            #if str(coord) in self.neuronIdcaches.keys():
            if str(coord) in self.neuronIdcaches:
                id = self.neuronIdcaches[str(coord)]
                self.neuronIdCounts[id] = self.neuronIdCounts[id] + 1
                return id
        if synapse is not None:
            #if synapse.id in self.neuronIdcaches.keys():
            if synapse.id in self.neuronIdcaches:
                id = self.neuronIdcaches[synapse.id]
                self.neuronIdCounts[id] = self.neuronIdCounts[id] + 1
                return id

        self.neuronId += 1

        if coord is not None:
            self.neuronIdcaches[str(coord)] = self.neuronId
            self.neuronIdCounts[self.neuronId] = 1
        elif synapse is not None:
            self.neuronIdcaches[synapse.id] = self.neuronId
            self.neuronIdCounts[self.neuronId] = 1
        else:
            raise RuntimeError("无法生成神经元id(NeatIdGenerator):坐标和突触id必须至少有一个")

        return self.neuronId

    def removeNeuronId(self,nid):
        #if nid not in self.neuronIdCounts.keys():return
        if nid not in self.neuronIdCounts: return
        self.neuronIdCounts[nid] = self.neuronIdCounts[nid] - 1
        if self.neuronIdCounts[nid] <= 0:
            #self.neuronIdcaches = {k: self.neuronIdcaches[k] for k in self.neuronIdcaches.keys() if self.neuronIdcaches[k] != nid}
            self.neuronIdcaches = {k: self.neuronIdcaches[k] for k in self.neuronIdcaches if
                                   self.neuronIdcaches[k] != nid}
            del self.neuronIdCounts[nid]

    def getAllCacheNeuronIds(self):
        '''
        取得所有缓存的神经元id
        :return: list 按照从小到大排序的神经元id
        '''
        nids = [] if collections.isEmpty(self.neuronIdcaches) else list(self.neuronIdcaches.values())
        nids.sort()
        return nids

    def getSynapseId(self,net,fromId,toId):
        '''
        取得突触id
        :param net:    NeuralNetwork  所属网络
        :param fromId: Union(int,str) 输入神经元id
        :param toId:   Union(int,str) 输出神经元id
        :return:       无论是哪个网络，输入和输出神经元都相同的突触id也相同
        '''
        key = str(fromId) + "-" + str(toId)
        #if key in self.synapseIdcaches.keys():
        if key in self.synapseIdcaches:
            id = self.synapseIdcaches[key]
            self.synapseIdCounts[id] = self.synapseIdCounts[id]+1
            return id

        self.synapseId += 1
        self.synapseIdcaches[key] = self.synapseId
        self.synapseIdCounts[self.synapseId] = 1
        return self.synapseId


    def removeSynapse(self,sid):
        #if sid not in self.synapseIdCounts.keys():return
        if sid not in self.synapseIdCounts: return
        self.synapseIdCounts[sid] = self.synapseIdCounts[sid] - 1
        if self.synapseIdCounts[sid] <= 0:
            #self.synapseIdcaches = {k: self.synapseIdcaches[k] for k in self.synapseIdcaches.keys() if
            #                       self.synapseIdcaches[k] != sid}
            self.synapseIdcaches = {k: self.synapseIdcaches[k] for k in self.synapseIdcaches if
                                    self.synapseIdcaches[k] != sid}
            del self.synapseIdCounts[sid]

    def getAllCacheSynasesIds(self,net=None):
        '''
        取得所有缓存的突触id
        :param net: 所属网络,如果为None，则为所有网络的合并
        :return: list 按照从小到大排序的神经元id
        '''
        sids = [] if collections.isEmpty(self.synapseIdcaches) else list(self.synapseIdcaches.values())
        sids.sort()
        return sids


    def getSpeciesid(self,netlist):
        netids = list(set(sorted([net.id for net in netlist])))
        key = reduce(lambda x,y:str(x)+","+str(y),netids)
        #if key in self.speciesIdcaches.keys():
        if key in self.speciesIdcaches:
            return self.speciesIdcaches[key]

        self.speciesid += 1
        self.speciesIdcaches[key] = self.speciesid
        return self.speciesid