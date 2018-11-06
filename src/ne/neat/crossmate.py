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
        for indpair in corssmateinds:
            net1 = indpair[0].getPhenome()
            net2 = indpair[1].getPhenome()
            net = net1.merge(net2)
            idgenerator = networks.idGenerators.find(net.definition.idGenerator)
            net.id = idgenerator.getNetworkId()
            ind = Individual(net.id, 0, session.curTime, session.pop.param.indTypeName)
            ind.speciedId = indpair[0].speciedId
            session.pop.inds.append(ind)
            session.monitor.recordDebug(NeatCrossmateOperation.name,'新增个体:'+str(ind))

        return  True,'',newinds






