import brain.networks as networks
from evolution.agent import Individual
from .selection import NeatSelection

__all__ = ['NeatCrossmateOperation']

class NeatCrossmateOperation:
    name = 'neat_corssmate'
    def __init__(self):
        pass

    def execute(self, session):
        # 取得选择操作的结果
        select_result = session.monitor.getOperationResult(NeatSelection.name)
        if select_result is None:
            return False,'交叉操作无法执行，选择操作结果无效',None
        if not select_result[0]:
            return False,'交叉操作无法执行，选择操作结果失败',None
        corssmateinds = select_result[2][0]
        if corssmateinds is None:
            return False,'交叉操作无法执行，选择结果中交叉个体无效',None

        if len(corssmateinds)<=0:
            return True,'没有需要交叉操作的个体',None


        # 遍历并进行交叉操作
        newinds = []
        count = 0
        for indpair in corssmateinds:
            net1 = session.pop[indpair[0]].getPhenome()
            net2 = session.pop[indpair[1]].getPhenome()
            idgenerator = networks.idGenerators.find(net1.definition.idGenerator)
            netid = idgenerator.getNetworkId()
            net = net1.merge(net2,netid)

            ind = Individual(net.id, session.curTime, net,session.pop.params.indTypeName,list(indpair),session.pop[indpair[0]].speciedId)
            session.pop.inds.append(ind)
            session.pop.getSpecie(ind.speciedId).putIndId(ind.id)
            session.monitor.recordDebug(NeatCrossmateOperation.name,'新增个体',str(ind))
            count += 1

        return  True,'交叉操作产生'+str(count)+'新个体',newinds






