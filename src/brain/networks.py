# -*- coding: UTF-8 -*-

from enum import Enum
import  numpy as np
import networkx as nx
import community
import threading
from functools import reduce
from utils import collections as collections
from utils.coords import Coordination
from utils.properties import Registry
from brain.elements import Synapse
from brain.elements import Neuron
import brain.runner as runner
from brain.runner import NeuralNetworkTask



__all__ = ['NetworkType','DefaultIdGenerator','idGenerators','NeuralNetwork']
# 网络类型
class NetworkType(Enum):
    # 感知机
    Perceptron = 1
    # BP
    BP = 2
    # 带时延的网络
    TPNN = 3
    # 脉冲
    Spiking = 4

#region 网络ld管理

class DefaultIdGenerator:
    def __init__(self):
        '''
        缺省id产生器，只能按序生成网络，神经元和突触的id，不支持并发
        '''
        self.netid = 0
        self.neuronId = 0
        self.synapseId = 0
        self.moduleId = 0
        self.ids = {}
    def getId(self,type,**kwargs):
        if type == 'net':
            return self.getNetworkId()
        elif type == 'neuron':
            return self.getNeuronId(kwargs['net'],kwargs['coord'],kwargs['synapse'])
        elif type == 'synapse':
            return self.getSynapseId(kwargs['net'],kwargs['fromId'],kwargs['toId'])
        elif type == 'module':
            return self.getModuleId(kwargs['net'])
        else:
            if type not in self.ids:
                self.ids[type] = 0
            r = self.ids[type]
            self.ids[type] += 1
            return r


    def getNetworkId(self):
        self.netid += 1
        return self.netid
    def getNeuronId(self,net,coord,synapse):
        self.neuronId += 1
        return self.neuronId
    def getSynapseId(self,net,fromId,toId):
        self.synapseId += 1
        return self.synapseId
    def getModuleId(self,net):
        self.moduleId += 1
        return self.moduleId

idGenerators = Registry()
idGenerators.register(DefaultIdGenerator(),'default')



#endregion

# 神经网络
class NeuralNetwork:

    #region 基本信息
    MAX_LAYER = 99999                                       #最大层，第一层是0层，因此总层数是该值+1
    MAX_RANGE = [(0,MAX_LAYER),(0,9999999),(0,99999)]      #最大坐标范围



    def __init__(self,id,definition):
        '''
        神经网络
        :param id:              id
        :param definition:      神经网络定义，参见xor.py例子
        '''
        self.id = id
        self.definition = definition
        self.neurons = []
        self.synapses = []
        self.attributes = {}
        self.taskstat = {}
        self.taskTestResult = []
        self.modules = []



    def __str__(self):
        ns = self.getNeurons()
        sps = self.getSynapses()
        return "net"+str(self.id) + \
                 "(" + reduce(lambda x,y:x+","+y,map(lambda n:str(n),ns))+")"+\
                 "[" + reduce(lambda x,y:x+","+y,map(lambda s:str(s),sps))+"]"

    def merge(self,net,newnetid,birth):
        '''
        合并两个网络，两个网络本身并不
        :param net:
        :param newnetid:
        :return:
        '''
        if net is None:return
        newnet = NeuralNetwork(newnetid,self.definition)

        for ns in self.neurons:
            for n in ns:
                if newnet.getNeuron(n.id) is not None:continue
                newnet.put(birth, n.coord, n.layer, n.modelConfiguration,n.id)
                #newnet.putneuron(n)
        for ns in net.neurons:
            for n in ns:
                if newnet.getNeuron(n.id) is not None: continue
                newnet.put(birth, n.coord, n.layer, n.modelConfiguration,n.id)
                #newnet.putneuron(n)

        for synapse in self.synapses:
            if newnet.getSynapse(id=synapse.id) is not None:continue
            newnet.connect(synapse.fromId, synapse.toId, birth, synapse.modelConfiguration)
            #newnet._connectSynapse(synapse)

        for synapse in net.synapses:
            if newnet.getSynapse(id=synapse.id) is not None:continue
            newnet.connect(synapse.fromId, synapse.toId, birth, synapse.modelConfiguration)
            #newnet._connectSynapse(synapse)

        return newnet

    def createAllNeurons(self,model_config_function=None):
        '''
        创建所有神经元
        :param model_config_function:　　对每层，每个神经元分配计算模型配置信息的函数
        :return:　None
        '''
        if model_config_function is None:
            model_config_function = self.__default_model_config_funtion
        for index,count in enumerate(self.definition.neuronCounts):
            modelconfig = model_config_function(index,-1)
            self.createLayer(index,modelconfig)

    def __default_model_config_funtion(self,layer,xh):
        '''
        缺省模型分配函数
        :param layer: 　神经元层
        :param xh: 　　神经元序号
        :return: 　　　模型配置
        '''
        if layer == 0:
            return self.definition.models.input
        elif layer == len(self.definition.neuronCounts)-1:
            if 'output' in self.definition.models:
                return self.definition.models.output
        return self.definition.models.hidden
    def createLayer(self,layerIndex,modelConfig):
        if layerIndex < 0 or layerIndex >= len(self.definition.neuronCounts):
            raise RuntimeError('createLayer failed: invaild layerIndex:'+ str(layerIndex))
        count = self.definition.neuronCounts[layerIndex]
        # 计算实际层编号
        layerInterval = int((self.definition.config.range[0][1] - self.definition.config.range[0][0])/(len(self.definition.neuronCounts)-1))
        layer = 0
        if layerIndex == 0:layer = self.definition.config.range[0][0]
        elif layerIndex == len(self.definition.neuronCounts)-1:layer = self.definition.config.range[0][1]
        else: layer = self.definition.config.range[0][0]+layerIndex * layerInterval

        while len(self.neurons) <= layerIndex:
            self.neurons.insert(layerIndex, [])
        # 生成神经元坐标
        idGenerator = idGenerators.find(self.definition.idGenerator)
        coords = self.__createCoord(layer, count)

        # 创建神经元
        for i in range(count):
            coord = coords[i]
            neuronid = idGenerator.getNeuronId(self, coord, None)
            neuron = Neuron(neuronid, layer, 0, modelConfig, coord)
            self.neurons[layerIndex].append(neuron)

        return self.neurons[layerIndex]

    def __createCoord(self,layer,count):
        '''
        创建坐标
        :param layer:   int  层号
        :param count:   int  神经元数量
        :return: list of Coordination
        '''
        netdef = self.definition
        width = netdef.config.range[1][1] - netdef.config.range[1][0]
        interval = int((width+1) / count)

        coords = []
        for i in range(count):
            values = [layer]
            if netdef.config.dimension >= 2:
                values.append(netdef.config.range[1][0] + i*interval)
            if netdef.config.dimension >= 3:
                values.append(int((netdef.config.range[2][1] - netdef.config.range[2][0])/2))
            coord = Coordination(*values)
            coords.append(coord)
        return coords
    #endregion

    #region 神经元素数量
    def getInputNeuronCount(self):
        '''
        输入神经元数量
        :return:
        '''
        return len(self.getInputNeurons())

    def getOutputNeuronCount(self):
        '''
        输出神经元数量
        :return:
        '''
        return len(self.getOutputNeurons())

    def getHiddenNeuronCount(self):
        '''
        隐藏神经元数量
        :return:
        '''
        return  len(self.getHiddenNeurons())

    def getNeuronCount(self):
        '''
        神经元总数量
        :return:
        '''
        return len(self.getNeurons())

    def getNeuronLayerCount(self):
        '''
        每层神经元数量
        :return: dict 每层的神经元数量，key是层id，value是数量
        '''
        r = {}
        for index, ns in enumerate(self.neurons):
            if collections.isEmpty(ns):continue
            r[ns[0].layer] = r.get(ns[0].layer,0)+1
        return r

    #endregion

    #region 查找神经元集合

    def getInputNeurons(self):
        '''
        取得所有输入神经元
        :return:
        '''
        if len(self.neurons)<=0:return []
        return self.neurons[0]

    def getOutputNeurons(self):
        '''
        取得所有输出神经元
        :return:
        '''
        if len(self.neurons)<=1:return []
        return self.neurons[-1]

    def getHiddenNeurons(self):
        '''
        取得所有隐藏层神经元
        :return:
        '''
        if len(self.neurons) <= 2: return []
        r = []
        nl = self.neurons[1:-1]
        if len(nl) <= 0:return r
        for ns in nl:
            r.extend(ns)
        return r
        #return reduce(lambda x,y:x.extend(y),self.neurons[1:-1])

    def getNeurons(self,layer = -1,activation=None):
        '''
        取得特定层，且满足激活状态的神经元
        :param layer:        int 层，-1表示所有层
        :param activation:   bool 激活状态，None表示所有状态
        :return:
        '''
        if len(self.neurons) <= 0: return []

        r = []
        collections.foreach(self.neurons, lambda ns: r.extend(ns))
        if layer < 0 and activation is None:
            return r

        return collections.findall(r,lambda n:(layer<0 or n.layer == layer) and (r['activation']==activation))


    def getHasVarNeurons(self,varname):
        '''
        取得拥有某个名称变量的所有神经元
        :param varname:
        :return:
        '''
        neurons = self.getNeurons()
        return collections.findall(neurons,lambda n:n.getVariable(varname) is not None)

    def getLayerNeurons(self):
        '''取得分层神经元集合'''
        return self.neurons

    def getNeuronLayerIndex(self,layer):
        '''
        取得某层在神经元集合中的下标
        :param layer:
        :return:
        '''
        for index,ns in enumerate(self.neurons):
            if len(ns)<=0:continue
            if ns[0].layer == layer:return index
        return -1

    def getPrevNeuron(self,neuronId):
        '''
        取得某神经元的前序连接神经元
        :param neuronId:
        :return:
        '''
        synapses = self.getInputSynapse(neuronId)
        if collections.isEmpty(synapses):return []
        return list(map(lambda s:self.getNeuron(id=s.fromId),synapses))

    def getNextNeuron(self,neuronId):
        '''
        取得某神经元的后序连接神经元
        :param neuronId:
        :return:
        '''
        synapses = self.getOutputSynapse(neuronId)
        if collections.isEmpty(synapses): return []
        return list(map(lambda s: self.getNeuron(id=s.toId), synapses))



    # endregion

    # region 查找特定神经元
    def getNeuron(self, id=-1, layer=-1, xhInLayer=-1,coord=None):
        '''
        查找满足特定条件神经元（先按id，再按坐标，在按层和层内序号)
        :param id:       神经元id，有效则会优先查找
        :param layer:    所在层
        :param xhInLayer:  层内序号
        :param coord:    坐标
        :return:
        '''
        if id > 0: #优先按照id查找
            ns = self.getNeurons()
            return collections.first(ns,lambda n:n.id == id)

        if coord is not None: #其次按照特定坐标寻找
            ns = self.getNeurons()
            return collections.first(ns, lambda n: n.coord == coord)

        #查找特定层中某个序号的神经元
        ns = self.getNeurons(layer =layer)
        if not collections.isEmpty(ns):return None
        if xhInLayer >= len(ns):return None
        return ns[xhInLayer]
    #endregion

    #region 查询突触
    def getSynapses(self,fromId = -1,toId=-1):
        '''所有突触'''
        if fromId == -1 and toId == -1:
            return self.synapses
        elif fromId == -1:
            return self.getInputSynapse(toId)
        elif toId == -1:
            return self.getOutputSynapse(fromId)
        else:
            s = self.getSynapse(fromId=fromId,toId=toId)
            return [] if s is None else [s]

    def getInputSynapse(self,neuronId):
        '''取得指定神经元的输入突触'''
        return collections.findall(self.synapses,lambda s:s.toId == neuronId)

    def getOutputSynapse(self,neuronId):
        '''取得指定神经元的输出突触'''
        return collections.findall(self.synapses, lambda s: s.fromId == neuronId)

    def getSynapse(self,id=-1,fromId=-1,toId=-1):
        '''
        取得特定突触
        :param id:          str  or int 突触id
        :param fromId:      str or int 输入神经元
        :param toId:        str or int 输出神经元
        :return:
        '''
        if id != -1:
            return collections.first(self.synapses,lambda s:s.id == id)
        return collections.first(self.synapses,lambda s:s.fromId == fromId and s.toId == toId)

    def getConnectionMarix(self,returntype='tuple',valuetype='01'):
        '''
        @:param returntype str
                           tuple 返回元组 tuple[fromId][toId] = synapse
                           list  返回二维列表,值根据valuetype
        ::param valuetype str   01  有连接为1
                                synapse 值为突触对象
                                weight  值为权重
        得到连接矩阵
        :return: dict key是两个神经元的id
        '''


        if returntype == 'tuple':
            marix = {}
            synapses = self.getSynapses()
            def func(s):
                marix[s.fromId][s.toId] = s
            collections.foreach(synapses,func)
            return marix
        else:
            marix = []
            allneurons = self.getNeurons()
            for i in range(len(allneurons)):
                srcNeuron = allneurons[i]
                row = []
                for j in range(len(allneurons)):
                    destNeuron = allneurons[j]
                    synapse = self.getSynapse(fromId= srcNeuron.id,toId=destNeuron.id)
                    if valuetype == '01':
                        row.append(0 if synapse is None else 1.)
                    elif valuetype == 'synapse':
                        row.append(synapse)
                    else:
                        row.append(synapse['weight'])
                marix.append(row)
            return marix
    #endregion

    #region 神经元和突触数量
    def getNeuronsCount(self,layer = -1):
        '''
        查询特定层的神经元数量
        :param layer: -1表示所有层
        :return:
        '''
        return collections.count(self.getNeurons(),lambda n:layer == -1 or n.layer == layer)

    def getLayerCount(self):
        '''
        得到每层的神经元数量
        :return:  dict key是layerId，value是数量
        '''
        ls = {}
        ns = self.getNeurons()

        collections.foreach(ns,lambda s : exec('ls[s.layer] = ls.get(s.layer,0)+1'))
        return ls
    #endregion

    #region 修改神经元或者突触
    def _connectSynapse(self,synapse):
        '''
        添加突触
        :param synapse: 突触对象，如果缺少id，会为其添加一个，如果该连接已经存在，会覆盖原来的
        :return: 网络对象
        '''
        if synapse is None:return self

        # 检查id
        if synapse.id <= 0:
            idGenerator = self.definition.get('idGenerator', 'default')
            if idGenerator is None: raise RuntimeError("连接神经元失败(NeuralNetwork.connect(synapse)):idGenerator无效")
            synapse.id = idGenerator.getSynapseId(self,synapse.fromId,synapse.toId)

        # 检查是否已经存在
        s = self.getSynapse(fromId=synapse.fromId,toId=synapse.toId)
        if s is None:
            self.synapses.append(synapse)
            return self

        index = self.synapses.index(s)
        self.synapses[index] = synapse
        return self

    def connect(self,srcid,destid,birth,synapseModelConfig):
        '''
        连接两个神经元
        :param srcid:  int or list 输入神经元id
        :param destid: int or list 输输出神经元id
        :param birth:  int or float 连接时间
        :param synapseModelConfig: dict 突触计算模型配置
        :return:
        '''
        idGenerator = idGenerators.find(self.definition.get('idGenerator','default'))
        if idGenerator is None:raise RuntimeError("连接神经元失败(NeuralNetwork.connect(srcid,destid)):idGenerator无效")
        if synapseModelConfig is None: raise RuntimeError(
            "连接神经元失败(NeuralNetwork.connect(srcid,destid)):synapseModelConfig效")

        if srcid is not list and destid is not list:
            srcNeuron = self.getNeuron(id=srcid)
            destNeuron = self.getNeuron(id=destid)
            if srcNeuron is None:raise  RuntimeError("连接神经元失败(NeuralNetwork.connect(srcid,destid)):接入神经元无效："+srcid)
            if destNeuron is None: raise RuntimeError("连接神经元失败(NeuralNetwork.connect(srcid,destid)):接入神经元无效："+destid)
            if srcNeuron.layer >= destNeuron.layer:
                raise RuntimeError('连接神经元失败(NeuralNetwork.connect(srcid,destid)):接入神经元的层大于等于接出神经元的层:src='+str(srcNeuron)+',dest='+str(destNeuron))
            synapse = Synapse(idGenerator.getSynapseId(self,srcid,destid),birth,srcid,destid,synapseModelConfig)
            self._connectSynapse(synapse)
            return self
        if srcid is list:
            for sid in srcid:
                self.connect(sid,destid,birth,synapseModelConfig)
        if destid is list:
            for did in destid:
                self.connect(srcid,did,birth,synapseModelConfig)
        return self

    def putneuron(self,neuron,inids=None,outinds=None,synapseModelConfig=None):
        '''
        添加神经元,会检查神经元id，是否重复（重复会删除旧的）
        :param neuron:   Neuron 待添加神经元
        :param inids:    int or list 输入神经元id
        :param outinds:  int or list 输出神经元id
        :param synapseModelConfig: 突触计算模型
        :return:
        '''
        if neuron is None:return self

        # 检查神经元id
        if neuron.id <= 0:
            idGenerator = idGenerators.find(self.definition.get('idGenerator', 'default'))
            if idGenerator is None: raise RuntimeError("连接神经元失败(NeuralNetwork.connect(srcid,destid)):idGenerator无效")
            neuron.id = idGenerator.getNeuronId(self,neuron.coord)

        # 检查神经元是否已经存在
        n = self.getNeuron(id = neuron.id)
        if n is not None:
            self.remove(n)

        # 第一次添加神经元
        layerindex = -1
        if len(self.neurons)<=0:
            self.neurons.append([])
            layerindex = 0
        else:
        # 根据神经元所在层查找对应位置
            for i,ns in enumerate(self.neurons):
                if collections.isEmpty(ns):
                    del self.neurons[i]
                    i -= 1
                    continue
                if ns[0].layer == neuron.layer:
                    layerindex = i
                elif ns[0].layer > neuron.layer:
                    layerindex = i
                    self.neurons.insert(i,[])
                if layerindex >= 0:break
                continue

        # 添加神经元
        if layerindex < 0: self.neurons.append([neuron])
        else: self.neurons[layerindex].append(neuron)

        # 连接
        if inids is not None and synapseModelConfig is not None:
            self.connect(inids,neuron.id,neuron.birth,synapseModelConfig)
        if outinds is not None and synapseModelConfig is not None:
            self.connect(neuron.id,outinds,neuron.birth,synapseModelConfig)

        return self

    def put(self, birth, coord, layer, neuronModelConfig, id = None,inids=None, outinds=None, synapseModelConfig=None):
        '''
        添加神经元
        :param birth:                int or float 添加时间，必须
        :param coord:                Coordinate   坐标，可选
        :param layer:                int          层，必须
        :param neuronModelConfig:    dict 神经计算模型配置，必须
        :param id                    int  神经元id,可选,如果无效,会尝试生成一个,但是要求coord有效
        :param inids:                int or list，输入神经元id
        :param outinds:              int or list，输入神经元id
        :param synapseModelConfig:   dict，突触计算模型配置
        :return:
        '''
        if birth < 0:raise  RuntimeError('添加神经元失败(NeuralNetwork.put):birth无效')
        #if layer < 0:raise  RuntimeError('添加神经元失败(NeuralNetwork.put):layer无效')
        if neuronModelConfig is None:raise  RuntimeError('添加神经元失败(NeuralNetwork.put):neuronModelConfig无效')
        idGenerator = idGenerators.find(self.definition.get('idGenerator', 'default'))
        if idGenerator is None: raise RuntimeError("连接神经元失败(NeuralNetwork.connect(srcid,destid)):idGenerator无效")

        if id is None:
            id = idGenerator.getNeuronId(self,coord,None)
        n = Neuron(id,layer,birth,neuronModelConfig,coord)
        return self.putneuron(n,inids,outinds,synapseModelConfig)


    def remove(self,ele):
        if ele is None:return
        if isinstance(ele,Synapse):
            self.synapses.remove(ele)
            return
        if isinstance(ele,Neuron):
            input_sypanses = self.getInputSynapse(ele.id)
            output_sypanses = self.getOutputSynapse(ele.id)
            synapses = output_sypanses + input_sypanses
            collections.foreach(synapses,lambda s:self.remove(s))
            for i,ns in enumerate(self.neurons):
                for j,n in enumerate(ns):
                    if n == ele:
                        ns.remove(n)
                        if len(ns)<=0:
                            self.neurons.remove(ns)
                        return


    #endregion

    #region 执行相关
    def reset(self):
        '''
        重置执行状态
        :return: None
        '''
        ns = self.getNeurons()
        collections.foreach(ns,lambda n:n.reset())
        synapses = self.getSynapses()
        collections.foreach(synapses,lambda s:s.reset())

    def getRunner(self):
        name = self.definition.runner.name
        return runner.runners.find(name)

    def doLearn(self):
        runner = self.getRunner()
        task = self.definition.runner.task
        runner.doLearn(self,task)
        return task

    def activate(self,inpputs):
        runner = self.getRunner()
        return runner.activate(self,inpputs)

    def doTest(self):
        runner = self.getRunner()
        task = self.definition.runner.task
        self.taskTestResult = []
        self.taskstat = {}
        runner.doTest(self, task)
        self.doTestStat()
        return self.taskTestResult

    def setTestResult(self,i,test_result,doStat=False):
        '''
        记录测试结果
        :param i 样本序号
        :param test_result:  测试结果
        :return: None
        '''
        if self.taskTestResult is None:self.taskTestResult = []
        while i > len(self.taskTestResult):
            self.taskTestResult.append(0.0)
        self.taskTestResult.append(test_result)

        if doStat:
            self.doTestStat()

    def doTestStat(self):
        '''
        对测试结果进行统计
        :return: None
        '''

        task = self.definition.runner.task
        if task.test_y is None or len(task.test_y)<=0:
            return

        testcount = correctcount = 0
        mae = mse = 0.0
        for index,result in enumerate(self.taskTestResult):
            if task.test_y is None or len(task.test_y) <= index:
                continue
            if collections.equals(self.taskTestResult[index],task.test_y[index]):
                correctcount += 1
            else:
                diff = abs(self.taskTestResult[index]-task.test_y[index])
                if not task.kwargs['multilabel'] and diff <= task.kwargs['deviation']:
                    correctcount += 1
                elif task.kwargs['multilabel']:
                    if isinstance(task.kwargs['deviation'],float):
                        if np.average(diff) <= task.kwargs['deviation']:
                            correctcount += 1
                    elif isinstance(task.kwargs['deviation'],list):
                        if collections.all(diff - task.kwargs['deviation'],lambda t : t <0):
                            correctcount += 1

            mae += abs(self.taskTestResult[index]-task.test_y[index])
            mse += pow(self.taskTestResult[index]-task.test_y[index],2)
            testcount += 1


        self.taskstat[NeuralNetworkTask.INDICATOR_TEST_COUNT] = testcount
        self.taskstat[NeuralNetworkTask.INDICATOR_CORRECT_COUNT] = correctcount
        self.taskstat[NeuralNetworkTask.INDICATOR_ACCURACY] = correctcount/testcount
        self.taskstat[NeuralNetworkTask.INDICATOR_MEAN_ABSOLUTE_ERROR] = mae / testcount
        self.taskstat[NeuralNetworkTask.INDICATOR_MEAN_SQUARED_ERROR] = mse / testcount

    def __setitem__(self, key, value):
        self.taskstat[key] = value

    def __getitem__(self, item):
        #if item in self.taskstat.keys():return self.taskstat[item]
        if item in self.taskstat: return self.taskstat[item]
        return None
        #return super(NeuralNetwork,self).__getitem__(item)

    def compute_modular(self):
        '''

        :return: 模块度的取值范围为：[−1/2,1)[−1/2,1)，有可能得到负值;
                   论文表示当Q值在0.3~0.7之间时，说明聚类的效果很好
        '''


        G = nx.Graph()
        allneurons = self.getNeurons()
        nids = [n.id for n in allneurons]
        G.add_nodes_from(nids)

        synapses = self.getSynapses()
        for s in synapses:
            G.add_edge(s.fromId,s.toId)

        part = community.best_partition(G)
        return (community.modularity(part, G),part)


        # 生成点与点之间的距离矩阵,这里用的欧氏距离:
        '''
        points = self.getConnectionMarix(returntype=list, valuetype='01')
        disMat = sch.distance.pdist(points, 'euclidean')
        # 进行层次聚类:
        Z = sch.linkage(disMat, method='average')
        # 根据linkage matrix Z得到聚类结果:
        cluster = sch.fcluster(Z,t=0,criterion='inconsistent')
        print('网络'+str(self.id)+'的模块划分:'+str(cluster))
        # 计算模块度,基于"Neural Modularity Helps Organisms Evolve to
        # Learn New Skills without Forgetting Old Skills"
        '''

        #1.生成社区矩阵
        '''
        m = len(set(cluster))  # 聚类个数
        if m <= 1:
            return 0.
        n = len(points)
        points = np.array(points)
        lin = list(map(np.sum, points))
        col = list(map(np.sum, zip(*points)))
        sum = 0.
        for i in range(n):
            for j in range(n):
                if cluster[i] != cluster[j]:
                    continue
                ki_in = col[i]
                kj_out = lin[j]
                sum += points[i][j] - ki_in * kj_out / (2 * m)

        Q = sum / (2 * m)
        return Q
        '''

        '''
        marix = []
        r = np.arange(1,m+1,1)
        for moduleid1 in r:
            row = []
            for moduleid2 in r:
                index1 = np.where(cluster==moduleid1)
                index2 = np.where(cluster==moduleid2)
                count = 0
                for i in index1[0]:
                    for j in index2[0]:
                        if points[i][j] > 0:
                            count += 1
                row.append(count)
            marix.append(row)

        n = len(marix)
        marix = np.array(marix)
        lin = list(map(np.sum, marix))
        col = list(map(np.sum, zip(*marix)))
        sum = 0.
        for i in range(n):
            for j in range(n):
                if marix[i][j] == 0:
                    continue
                ki_in = col[i]
                kj_out = lin[j]
                sum += marix[i][j] - ki_in*kj_out / (2*m)

        Q = sum / (2*m)
        return Q
        '''
        #endregion






