import numpy as np
from ne.senal.box import BoxGene
from ne.senal.box import Box
import ne


class SelectionOperation:
    name = 'senal_selection'
    def execute(self,session):
        inds = session.pop.inds
        np.sort(inds,order = 'fitness')
        inds = inds[:len(inds)*session.runParam.mutate.propotion]
        inds = [ind.id for ind in inds]
        return True,'',None,inds

class MutateOperation:
    name = 'senal_mutate'
    def execute(self,session):
        '''
        变异操作
        :param session:
        :return:
        '''
        select_result = session.monitor.getOperationResult(SelectionOperation.name)
        mutateinds = select_result[2][1]
        if mutateinds is None:
            return False, 'Neat变异操作无法执行，选择结果中变异个体集合无效', None
        if len(mutateinds) <= 0:
            return True, '没有需要变异操作的个体', None

        # 对候选个体进行变异
        mutateStat = {}
        for mutateid in mutateinds:
            ind = session.pop[mutateid]
            net = ind.getPhenome()
            cognition_box = net.get_cognition_box()
            size = 1
            while size<=len(cognition_box):
                box = np.random.choice(cognition_box,size=size)
                if not isinstance(box,list):box = [box]
                attented_box = net.get_attented_box(box)
                operations = list(set([box.getExpressionOperation() for box in attented_box]))
                operations = ne.senal.attention.attention_operations_name - operations
                if len(operations)<=0:
                    size += 1
                    continue
                operation = np.random.choice(operations, size=1)
                boxGene = BoxGene(id=1,type=BoxGene.type_attention,expression=operation,initsize=1)
                newbox = Box(boxGene,net)
                net.put_box(newbox)
                break




