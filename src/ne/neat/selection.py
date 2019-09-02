# -*- coding: UTF-8 -*-

from functools import reduce
import random
import logging
import numpy as np
import utils.collections as collections
import utils.strs as strs


__all__ = ['NeatSelection']

class NeatSelection():
    name = 'neat_selection'
    def __init__(self):
        self.name = NeatSelection.name

    def execute(self,session):
        #region 第一步：规划每个物种中应有的个体数量
        # 取得物种集合,并按平均适应度排序
        species = session.pop.getSpecies()
        if collections.isEmpty(species):
            raise RuntimeError('NEAT选择操作失败：物种集合为空')
        species.sort(key=lambda s:s['fitness']['average'],reverse=True)

        # 根据物种的平均适应度在所有物种中占的比重，计算每个物种的目标个体数量
        specie_total_fitness = sum(list(map(lambda sp:sp['fitness']['average'],species)))

        totalSize = 0
        for i in range(len(species)):
            specie = species[i]
            # 根据物种适应度计算目标个体数量
            speicesFitness = specie['fitness']['average']
            specie.targetSize = int((speicesFitness / specie_total_fitness) * len(session.pop.inds))
            totalSize += specie.targetSize

        # 如果所有物种的目标个体数量之和仍小于种群个体数量，将不足的部分加到适应度最高的物种上（按照上面计算，不会出现大于的情况）
        if totalSize < len(session.pop.inds):
            species[0].targetSize += len(session.pop.inds) - totalSize
        totalSize = len(session.pop.inds)

        session.monitor.recordDebug('neat_selection','物种的目标个体数量',reduce(lambda x,y:x+","+y,map(lambda s:str(s.id)+"="+str(s.targetSize),species)))

        #endregion

        #region 第二步:遍历每个物种，如果物种中实际个体数量大于前面计算的每个物种的目标个体数量，则将适应度差的个体淘汰
        removeIndids = []
        for i in range(len(species)):
            specie = species[i]
            # 将物种中个体按照适应度由高到低排序，其中精英个体尽管排前面
            specie.indids.sort(key=lambda indid:session.pop[indid]['fitness'] + 0.000001 if session.pop[indid] in session.pop.eliest else 0,reverse = True)
            # 实际个体数量不多于目标个体数量，不需要淘汰
            if len(specie.indids) <= specie.targetSize:
                continue

            # 删除适应度最小的个体，直到实际个体数量与目标个体数量相等(这样的删除方法，有可能会导致精英个体也被删除)
            while len(specie.indids) > specie.targetSize:
                removeIndid = specie.indids[-1]
                removeInd = session.pop[removeIndid]
                removeIndids.append(removeIndid)
                del specie.indids[-1]               # 从物种记录中删除
                session.pop.inds.remove(removeInd)     # 从种群记录中删除
        session.monitor.recordDebug('neat_selection','删除的个体',collections.mapreduce(removeIndids,reducefunc=lambda i,j:str(i)+','+str(j)))

        # 遍历所有物种，如果有物种个体数量为0，则将该物种删除
        species = [s for s in species if len(s.indids)>0]
        #endregion

        #region 第三步：对每个物种，随机选择需要交叉操作的个体
        corssmeateInds = []
        for specie in species:
            if len(specie.indids) >= specie.targetSize:
                continue
            for i in range(specie.targetSize - len(specie.indids)):
                if len(specie.indids) == 1:
                    corssmeateInds.append((specie.indids[0],specie.indids[0]))
                elif len(specie.indids) == 2:
                    corssmeateInds.append((specie.indids[0], specie.indids[1]))
                else:
                    indexpair = random.sample(range(len(specie.indids)), 2)
                    corssmeateInds.append((specie.indids[indexpair[0]], specie.indids[indexpair[1]]))

        # 有错误:session.monitor.recordDebug('neat_selection', '交叉的个体',  reduce(lambda i, j: str(list(i)[0])+"-"+str(list(i)[1]) + ',' + str(list(j)[0])+"-"+str(list(j)[1]), corssmeateInds))
        # reduce(lambda i,j:str(i[0])+'-'+str(i[1])+','+str(j[0])+'-'+str(j[1]),[(0,1),(2,3)])  --->  0-1,2-3
        # reduce(lambda i,j:str(i[0])+'-'+str(i[1])+','+str(j[0])+'-'+str(j[1]),[(0,1),(2,3),(4,5)]) ---> 0--,4-5
        sdebug = ''
        for cross in corssmeateInds:
            if strs.isVaild(sdebug):sdebug += ','
            sdebug += str(cross[0]) + '-' + str(cross[1])
        session.monitor.recordDebug('neat_selection', '交叉的个体',sdebug)

        #region 第四步：对所有个体按照适应度从高到低排序，随机选择其中一部分作为变异个体
        metateinds = []
        # 计算变异个体数量
        mutateCount = int(session.runParam.mutate.propotion) if session.runParam.mutate.propotion >=1 else int(totalSize * session.runParam.mutate.propotion)
        if mutateCount <= 0:
            return True,'',(corssmeateInds,metateinds)

        # 对所有个体按照适应度从高到低排序
        session.pop.inds.sort(key=lambda x:x['fitness']+0.000001 if x in session.pop.eliest else 0,reverse = True)
        # 选择候选变异个体（精英个体将被排除）
        candidateInds = collections.findall(session.pop.inds,lambda ind:ind not in session.pop.eliest)
        # 为每个个体计算一个选择概率（适应度越低的被选择的概率就高）
        if len(candidateInds) <=0:
            max, avg, min, stdev = 0.,0.,0.,0.
            print('变异个体数量无效,'+str(session.pop.eliest))
            return True, '选择操作完成,其中淘汰个体数量=' + str(len(removeIndids)) + ',交叉个体数量=' + str(
                len(corssmeateInds)) + ',变异个体数量=0', (
                   corssmeateInds, [])

        else:
            max,avg,min,stdev = collections.rangefeature(list(map(lambda ind:ind['fitness'],candidateInds)))
        #fitnesssum = sum(list(map(lambda ind:ind['fitness'],candidateInds)))
        mutateSelProb = [1-((ind['fitness']-min)/((max-min) if max != min else 1)) for index,ind in enumerate(candidateInds)]
        mutateSelProb = np.array(mutateSelProb)
        p = mutateSelProb / mutateSelProb.sum()
        np.random.seed(0)
        #p = np.array(mutateSelProb)
        mutateinds = np.random.choice(candidateInds, size=mutateCount,p=p.ravel())
        session.monitor.recordDebug('neat_selection', '变异的个体',
                                    reduce(lambda i, j: i + ',' + j, map(lambda ind:str(ind.id),mutateinds)))

        return True, '选择操作完成,其中淘汰个体数量='+str(len(removeIndids))+',交叉个体数量='+str(len(corssmeateInds))+',变异个体数量='+str(len(mutateinds)), (corssmeateInds, list(map(lambda ind : ind.id,mutateinds)))






