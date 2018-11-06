from random import choice, random, shuffle
from .selection import NeatSelection
import brain.networks as networks
from brain.elements import Neuron
from brain.elements import Synapse
from brain.runner import NeuralNetworkTask
from utils.properties import Range
import utils.collections as collections


__all__ = ['NeatMutate']

class NeatMutate:
    name = 'neat_mutate'
    def __init__(self):
        self.name = NeatMutate.name

    def execute(self,session):
        # 取得选择操作的执行结果
        select_result = session.monitor.getOperationResult(NeatSelection.name)
        if select_result is None:
            return False, 'Neat变异操作无法执行，选择操作结果无效', None
        if not select_result[0]:
            return False, 'Neat变异操作无法执行，选择操作结果失败', None
            mutateinds = select_result[2][1]
        if mutateinds is None:
            return False, 'Neat变异操作无法执行，选择结果中变异个体集合无效', None

        if len(mutateinds) <= 0:
            return True, '没有需要变异操作的个体', None

        # 生成变异各种操作概率
        topoopertions = ['addnode','addconnection','deletenode','deleteconnection']
        topoopertionrates = [session.runParam.mutate.topo.addnode,session.runParam.mutate.topo.addconnection,
                             session.runParam.mutate.topo.deletenode,session.runParam.mutate.topo.deleteconnection]
        # 对候选个体进行变异
        for mutateid in mutateinds:
            if session.runParam.mutate.model.rate >= 0:
                raise RuntimeError('对个体计算模型的变异还没有实现')
            if session.runParam.mutate.activation.rate >= 0:
                raise RuntimeError('对个体激活函数的变异还没有实现')

            np.random.seed(0)
            p = np.array(topoopertionrates)
            index = np.random.choice(topoopertions, p=p.ravel())
            mutateoperation = topoopertions[index]
            self.__dometate(session.pop[mutateid],mutateoperation,session)

        # 对每个个体的权重进行随机变化
        for ind in session.pop.inds:
            self.__doWeightTrain(ind,session)

        return True,'',None

    def __dometate(self,ind,operation,session):
        if operation == 'addnode': return self.__do_mutate_addnode(ind,session)
        elif operation == 'addconnection':return self.__do_mutate_addconnection(ind,session)
        elif operation == 'deletenode':return self.__do_mutate_deletenode(ind,session)
        elif operation == 'deleteconnection':return self.__do_mutate_deleteconnection(ind,session)
        return False,'无法识别的变异操作',None

    def __do_mutate_addnode(self,ind,session):
        net = ind.getPhenome()
        synapses = net.getSynapse()
        synapse = None
        if len(synapses)<=0:
            r,msg,synapse = self.____do_mutate_addconnection(ind)
        else:
            synapse = choice(synapses)

        fromneuron = net.getNeuron(session.fromId)
        toneuron = net.getNeuron(session.toId)
        neuronModel = net.definition.models.hidden
        synapseModel = net.definition.models.synapse
        newNeuronLayer = int((fromneuron.layer - toneuron.layer)/2)
        idgenerator = networks.idGenerators.find(net.definition.idGenerator)
        newNeuronid = idgenerator.getNeuronId(net,None,synapse)
        newNeuron = Neuron(newNeuronid,newNeuronLayer,session.curTime,neuronModel)
        net.put(newNeuron,fromneuron.id,toneuron.id,synapseModel)
        net.remove(synapse)
        session.monitor.recordDebug(NeatMutate.name,'添加新节点:'+str(newNeuron))

        return True,'',newNeuron

    def ____do_mutate_addconnection(self,ind,session):
        # 随机选择两个神经元
        net = ind.getPhenome()
        ns = net.getNeurons()
        if len(ns) < 2:
            return False,'添加连接失败：只有一个神经元',None

        # 寻找尚未连接的神经元对
        un_conn_neuron_pair = []
        for n1 in ns:
            for n2 in ns:
                if n1 == n2:
                    continue
                if n1.layer >= n2.layer:
                    continue
                if net.getSynapse(n1.id,n2.id) is None:
                    un_conn_neuron_pair.append((n1,n2))
        if len(un_conn_neuron_pair)<=0:
            return False,'添加连接失败：目前任意两个神经元之间都有连接，无法再添加',None
        # 随机选择
        index = random.sample(range(len(un_conn_neuron_pair)))
        n1,n2 = un_conn_neuron_pair[index]


        idgenerator = networks.idGenerators.find(net.definition.idGenerator)
        synapseid = idgenerator.getSynapseId(net,n1.id,n2.id)

        synapse = Synapse(synapseid,session.curTime,n1.id,n2.id,net.definition.models.synapse)
        net.connect(synapse)
        session.monitor.recordDebug(NeatMutate.name, '添加新连接:' + str(synapse))
        return True,'',synapse

    def __do_mutate_deletenode(self,ind,session):
        net = ind.getPhenome()
        ns = net.getHiddenNeurons()

        # 随机选择
        index = random.sample(range(len(ns)))
        neuron = ns[index]
        net.remove(neuron)
        session.monitor.recordDebug(NeatMutate.name, '删除神经元:' + str(neuron.id))
        return True,'',neuron

    def __do_mutate_deleteconnection(self,ind,session):
        net = ind.getPhenome();
        synapses = net.getSynapses()
        # 随机选择
        index = random.sample(range(len(synapses)))
        s = synapses[index]
        net.remove(s)
        session.monitor.recordDebug(NeatMutate.name, '删除连接:' + str(s.id))
        return True, '', s

    def __doWeightTrain(self,ind,session):
        net = ind.getPhenome();
        synapses = net.getSynapses()
        neurons = net.getHiddenNeurons()

        range1 = Range(net.definition.models.synapse.weight)
        range2 = Range(net.definition.models.hidden.bias)
        epoch = session.runParam.mutate.weight.epoch
        origin_mae = None
        last_mae = None
        for i in range(epoch):
            # 对权重改变前的网络进行测试
            task = net.doTest()
            old_mae = task[NeuralNetworkTask.INDICATOR_MEAN_ABSOLUTE_ERROR]
            if origin_mae is None:origin_mae = old_mae
            # 先记住原有的权重
            old_weights = [(s,s['weight']) for s in synapses]
            # 根据权重分布进行随机采样
            for s in synapses:
                s['weight'] = range1.sample()
            # 记住原有的偏置
            old_bias = [(n,n['bias']) for n in neurons]
            # 对神经元偏置进行随机采样:
            for n in neurons:
                n['bias'] = range2.sample()

            #权重改变后再次对网络进行任务测试
            task = net.doTest()
            new_mae = task[NeuralNetworkTask.INDICATOR_MEAN_ABSOLUTE_ERROR]
            if i == epoch-1:last_mae = new_mae
            # 改变后测试误差更小
            if new_mae <= old_mae:
                continue

            # f否则恢复原来的权重
            for i,s in enumerate(synapses):
                s['weight'] = old_weights[i]
            for i,n in enumerate(neurons):
                n['bias'] = old_bias[i]

        session.monitor.recordDebug(NeatMutate.name,'权重修正前后误差变化:old='+str(origin_mae)+",new="+str(last_mae)+",diff="+str(origin_mae - last_mae))
