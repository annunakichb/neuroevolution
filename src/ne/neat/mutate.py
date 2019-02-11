from random import choice, random, shuffle
from functools import reduce
import  numpy as np
from .selection import NeatSelection
import brain.networks as networks
from brain.elements import Neuron
from brain.elements import Synapse
from brain.runner import NeuralNetworkTask
from utils.properties import Range
import utils.collections as collections
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor,as_completed,wait,ALL_COMPLETED,FIRST_COMPLETED

__all__ = ['NeatMutate']

def parallel_doWeightTrain_wrapper(neatNutate,ind,session):
    neatNutate.__doWeightTrain(ind,session)
    return True

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


        # 对候选个体进行变异
        mutateStat = {}
        for mutateid in mutateinds:
            if session.runParam.mutate.model.rate > 0:
                raise RuntimeError('对个体计算模型的变异还没有实现')
            #if session.runParam.mutate.activation.rate > 0:
            #    raise RuntimeError('对个体激活函数的变异还没有实现')
            succ,msg,oper,obj = self.__domutate(session.pop[mutateid], session)
            if succ:
                if oper not in mutateStat.keys():mutateStat[oper] = 0
                mutateStat[oper] = mutateStat[oper] + 1

        # 对每个个体的权重进行随机变化
        parallel = session.runParam.mutate.weight.parallel
        if parallel is not None and parallel > 0:
            pool = ThreadPoolExecutor(max_workers=parallel)
            all_task = []
            for ind in session.pop.inds:
                all_task.append(pool.submit(self.__doWeightTrain, ind, session))
            wait(all_task, return_when=ALL_COMPLETED)
        else:
            for ind in session.pop.inds:
                self.__doWeightTrain(ind,session)

        resultMsg = reduce(lambda x,y:x+","+y,map(lambda key:key + "数量=" + str(mutateStat[key]),mutateStat)) if len(mutateStat)>0 else '无有效变异'
        return True,resultMsg,None

    def __domutate(self, ind, session):
        # 生成变异各种操作概率
        topoopertions = ['addnode', 'addconnection', 'deletenode', 'deleteconnection','activation']
        topoopertionrates = [session.runParam.mutate.topo.addnode, session.runParam.mutate.topo.addconnection,
                             session.runParam.mutate.topo.deletenode, session.runParam.mutate.topo.deleteconnection,
                             session.runParam.mutate.activation.rate]
        operationfuncs = [self.__do_mutate_addnode,self.__do_mutate_addconnection,self.__do_mutate_deletenode,self.__do_mutate_deleteconnection,self.__do_mutate_activationFunction]

        #np.random.seed(0)
        retryCount = 0
        while 1:
            p = np.array(topoopertionrates)
            mutateoperation = np.random.choice(np.array(topoopertions), p=p.ravel())
            index = topoopertions.index(mutateoperation)
            if index <0:return False,'无法识别的变异操作',None

            r = operationfuncs[index](ind,session)
            if r[0]:
                session.monitor.recordDebug(NeatMutate.name, 'ind' + str(ind.id) + '变异操作=' + mutateoperation,r[1] + '' if r[3] is None else r[1]+str(r[3]))
                return r
            else:
                v = topoopertionrates[index]  / (len(topoopertionrates)-1)
                for i,rate in enumerate(topoopertionrates):
                    topoopertionrates[i] = 0.0 if i == index else topoopertionrates[i] + v
                retryCount += 1
            if retryCount >= len(topoopertions):
                return False,'ind' + str(ind.id) + '所有变异操作都失败','',None

    def __do_mutate_addnode(self,ind,session):
        net = ind.genome
        synapses = net.getSynapses()
        synapse = None
        if len(synapses)<=0:
            r,msg,synapse = self.____do_mutate_addconnection(ind)
        else:
            synapse = choice(synapses)

        fromneuron = net.getNeuron(synapse.fromId)
        toneuron = net.getNeuron(synapse.toId)
        neuronModel = net.definition.models.hidden
        synapseModel = net.definition.models.synapse
        newNeuronLayer = int((toneuron.layer + fromneuron.layer)/2)
        idgenerator = networks.idGenerators.find(net.definition.idGenerator)
        newNeuronid = idgenerator.getNeuronId(net,None,synapse)
        newNeuron = Neuron(newNeuronid,newNeuronLayer,session.curTime,neuronModel)
        net.putneuron(newNeuron,fromneuron.id,toneuron.id,synapseModel)
        net.remove(synapse)

        return True,'','addnode',newNeuron

    def __do_mutate_addconnection(self,ind,session):
        # 随机选择两个神经元
        net = ind.genome
        ns = net.getNeurons()
        if len(ns) < 2:
            return False,'ind'+str(ind.id)+'添加连接失败：只有一个神经元','addconnection',None

        # 寻找尚未连接的神经元对
        un_conn_neuron_pair = []
        for n1 in ns:
            for n2 in ns:
                if n1 == n2:
                    continue
                if n1.layer >= n2.layer:
                    continue
                if net.getSynapse(fromId=n1.id,toId=n2.id) is None:
                    un_conn_neuron_pair.append((n1,n2))
        if len(un_conn_neuron_pair)<=0:
            return False,'ind'+str(ind.id)+'添加连接失败：目前任意两个神经元之间都有连接，无法再添加','addconnection',None
        # 随机选择
        n1,n2 = choice(un_conn_neuron_pair)

        idgenerator = networks.idGenerators.find(net.definition.idGenerator)
        synapseid = idgenerator.getSynapseId(net,n1.id,n2.id)

        synapse = Synapse(synapseid,session.curTime,n1.id,n2.id,net.definition.models.synapse)
        net._connectSynapse(synapse)
        session.monitor.recordDebug(NeatMutate.name, 'ind'+str(ind.id)+'添加新连接' , str(synapse))
        return True,'变异操作完成','addconnection',synapse

    def __do_mutate_deletenode(self,ind,session):
        net = ind.genome
        ns = net.getHiddenNeurons()
        if collections.isEmpty(ns):
            return False,'','deletenode',None
        # 随机选择
        neuron = choice(ns)
        net.remove(neuron)
        return True,'','deletenode',neuron

    def __do_mutate_activationFunction(self,ind,session):
        net = ind.genome
        ns = net.getHiddenNeurons()
        if collections.isEmpty(ns):
            return False, '', 'modifyactivationFunction', None
        # 随机选择
        neuron = choice(ns)
        activationFunction  = neuron._doSelectActiovationFunction()
        return True,'','modifyactivationFunction',str(neuron)+':'+activationFunction.nameInfo.name

    def __do_mutate_deleteconnection(self,ind,session):
        net = ind.genome
        synapses = net.getSynapses()
        if collections.isEmpty(synapses):
            return False,'','deleteconnection',None
        # 随机选择
        s = choice(synapses)
        net.remove(s)
        session.monitor.recordDebug(NeatMutate.name, 'ind'+str(ind.id)+'删除连接',str(s.id))
        return True, '','deleteconnection', s



    def __doWeightTrain(self,ind,session):
        net = ind.genome
        synapses = net.getSynapses()
        neurons = net.getHasVarNeurons('bias')

        #print('权重修正前:',str(ind))
        range1 = Range(net.definition.models.synapse.weight)
        range2 = Range(net.definition.models.hidden.bias)
        epoch = session.runParam.mutate.weight.epoch

        evoluator = session.pop.params.features['fitness']
        if 'fitness' not in ind.features.keys(): #新产生的个体没有计算过适应度
            value = evoluator.calacute(ind, session)
            ind['fitness'] = value
        origin_fitness = ind['fitness']
        last_fitness = origin_fitness
        old_fitness = origin_fitness
        for i in range(epoch):
            # 先记住原有的权重
            old_weights = [s['weight'] for s in synapses]
            # 根据权重分布进行随机采样
            for s in synapses:
                s['weight'] = range1.sample()
            # 记住原有的偏置
            old_bias = [n['bias'] for n in neurons]
            # 对神经元偏置进行随机采样:
            for n in neurons:
                n['bias'] = range2.sample()

            #权重改变后再次对网络计算适应度
            new_fitness = evoluator.calacute(ind, session)
            # 改变后测试误差更小
            if new_fitness >= old_fitness:
                old_fitness = new_fitness
                last_fitness = new_fitness
                ind['fitness'] = new_fitness
                continue

            # f否则恢复原来的权重
            for index,s in enumerate(synapses):
                s['weight'] = old_weights[index]
            for index,n in enumerate(neurons):
                n['bias'] = old_bias[index]

        #print('权重修正后:', str(ind))
        session.monitor.recordDebug(NeatMutate.name,'ind'+str(ind.id)+'权重修正前后适应度变化','old='+str(origin_fitness)+",new="+str(last_fitness)+",diff="+str(last_fitness - origin_fitness))