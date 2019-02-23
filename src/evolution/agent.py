from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor,as_completed,wait,ALL_COMPLETED,FIRST_COMPLETED
from evolution.env import EvaluationValue

from utils.properties import Registry
import utils.collections as collections
import utils.strs as strs

import  numpy as np

__all__ = ['IndividualType','individualTypes','speciesType','Individual','Population','Specie']

#region 个体信息

#个体类型
class IndividualType:
    def __init__(self,name,genomeType,genomeFactory,phenomeType,genomeDecoder=None):
        '''
        个体元信息
        :param name:                       str 个体类型名
        :param genomeType:                 type 基因类型
        :param genomeFactory:              any  缺省基因工厂，它包含create函数
        :param phenomeType:                type 表现型类型
        :param genomeDecoder:              any  解码器
        '''
        self.name = name
        self.genomeType = genomeType
        self.genomeFactory = genomeFactory
        self.phenomeType = phenomeType
        self.genomeDecoder = genomeDecoder

#个体类型注册表
individualTypes = Registry()
#种群类型注册表
speciesType = Registry()




#个体类
class Individual:
    def __init__(self,id,birth,genome,indTypeName,parentIds=[],speciedId = 0):
        '''
        :param id:                 int         缺省与基因id相同,构造方法中会检查
        :param birth:              int或float  创建时间
        :param genome:             any         基因型
        :param indTypeName:        str         个体类型名
        :param parentIds:          []          父个体id
        :param speciedId:          int         所属物种id
        '''
        self.id = id
        self.birth = birth
        self.genome = genome
        self._cachedPhenome = None
        self.indTypeName = indTypeName
        self.speciedId = speciedId
        self.features = {}
        self.parentIds = parentIds

    def compareTo(self,ind,indicator='fitness'):
        '''
        比较两个个体的特征指标，缺省是'fitness'
        :param ind:
        :return:
        '''
        return self.features[indicator] - ind.features[indicator]

    def reset(self):
        phenome = self.getPhenome()
        if phenome is not None:phenome.reset()

    def getPhenome(self):
        '''取得表现型'''
        indMeta = individualTypes.find(self.indTypeName)
        if indMeta is None:raise RuntimeError('个体类型没有注册:'+self.indTypeName)

        if indMeta.genomeDecoder is None:
            return self.genome

        self._cachedPhenome =  indMeta.genomeDecoder.decode(self)
        return self._cachedPhenome

    def getCachedPhenome(self):
        return self._cachedPhenome

    def __getitem__(self, item):
        #if item in self.features.keys():
        if item in self.features:
            return self.features[item].value
        return super.__getitem__(item)

    def __setitem__(self, key, value):
        #if key not in self.features.keys():
        if key not in self.features:
            self.features[key] = EvaluationValue()
        self.features[key].append(value)

    def getFeature(self,name):
        return self.features.get(name)

    def __str__(self):
        return self._getFeatureStr() + ":" + str(self.genome) + ""

    def _getFeatureStr(self):
        r = collections.dicttostr(self.features)
        return '(' + r + ')' if strs.isVaild(r) else ''
        #if collections.isEmpty(self.features.keys()): return ''
        #elif len(self.features.keys()) == 1: return self.features.keys()
        #return map(lambda k: self.features[k], self.features.keys())
#endregion

#region 种群信息
def doIndEvaulate_warpper(pop,ind,key,evoluator,session):
    print('doIndEvaulate_warpper of ' + str(ind.id))
    return pop.__doIndEvaulate(ind,key,evoluator,session)


class Population:
    __slots__ = ['params','inds','features','eliest','species']
    def __init__(self,params,initInds):
        self.params = params
        self.inds = initInds
        self.features = {}
        self.eliest = []
        self.species = []

    @classmethod
    def create(cls,popParam):
        '''
        创建种群
        :return: evolution.agent.Population 种群对象
        '''
        # 根据种群参数找到种群所属的个体类型
        indType =  individualTypes.find(popParam.indTypeName)
        if indType is None:raise RuntimeError('创建种群失败:种群类型名称找不到:'+popParam.indTypeName)
        # 找到基因工厂
        genomeFactory = indType.genomeFactory
        if popParam.genomeFactory is not None:
            genomeFactory = popParam.genomeFactory
        # 创建个体集合和种群
        genomes = genomeFactory.create(popParam)
        inds = list(map(lambda genome:Individual(genome.id,0,genome,popParam.indTypeName),genomes))
        return Population(popParam,inds)

    def getInd(self,id):
        return collections.first(self.inds,lambda ind:ind.id == id)

    def getSpecie(self,specieId):
        return collections.first(self.species,lambda s:s.id == specieId)

    def getSpecies(self):
        return self.species

    def evaulate(self,session):
        '''
        对种群进行评估
        :param session:
        :return:
        '''
        # 遍历每一个评估项
        for key,evoluator in self.params.features.items():
            # 对每一个个体计算评估值
            parallel = session.runParam.evalate.parallel
            if parallel is not None and parallel>0:
                pool = ThreadPoolExecutor(max_workers=parallel)
                all_task = []
                for ind in self.inds:
                    all_task.append(pool.submit(self.__doIndEvaulate,ind, key, evoluator, session))
                wait(all_task,return_when=ALL_COMPLETED)
            else:
                for ind in self.inds:
                    value = evoluator.calacute(ind,session)
                    ind[key] = value

            # 计算所有个体评估值的平均，最大和最小
            max,avg,min,stdev = collections.rangefeature(list(map(lambda i:i[key],self.inds)))
            self[key] = {}
            self[key]['max'] = max
            self[key]['average'] = avg
            self[key]['min'] = min
            self[key]['stdev'] = stdev

        # 按照适应度值排序
        self.inds.sort(key=lambda ind:ind['fitness'],reverse=True)

        # 记录精英个体id
        eliestCount = int(session.popParam.elitistSize) if session.popParam.elitistSize >= 1 else int(session.popParam.elitistSize * session.popParam.size)
        self.eliest = self.inds[0:eliestCount] if eliestCount>0 else []


    def __doIndEvaulate(self,ind,key,evoluator,session):
        value = evoluator.calacute(ind, session)
        ind[key] = value
        return value

    def __getitem__(self, item):
        '''
        如果item是个体id，则返回个体对象；如果是feature名称，则返回fature值
        :param item:
        :return:
        '''
        ind = self.getInd(item)
        if ind is not None: return ind

        #if item in self.features.keys():
        if item in self.features:
            if self.features[item] is EvaluationValue:
                return self.features[item].value
            else:
                return self.features[item]

        return super(Specie, self).__getitem__(item)

    def __setitem__(self, key, value):
        '''
        索引器
        :param key:   str key
        :param value: Union(EvaluationValue,dict) 值
        :return: None
        '''
        #if key not in self.features.keys():
        if key not in self.features:
            self.features[key] = value

        if isinstance(value,float):
            self.features[key] = EvaluationValue()
            self.features[key].append(value)
        else:
            self.features[key] = value

    def getEvaluationObject(self, key, name):
        #if key not in self.features.keys():
        if key not in self.features:
            return None

        if self.features[key] is EvaluationValue:
            return self.features[key]
        if self.features[key] is dict:
            return self.features[key][name]



# 物种
class Specie:
    __slots__ = ['id','indids','pop','features','targetSize']
    def __init__(self,id,inds,pop):
        '''
        物种
        :param id:    int 物种id
        :param inds:  list  of int or list of Individuals 属于该种群的个体
        :param pop:   物种
        '''
        self.id = id
        self.pop = pop
        self.targetSize = 0
        self.indids = []
        if not collections.isEmpty(inds):
            self.indids = list(map(lambda ind:ind.id if isinstance(ind,Individual) else ind,inds))
        self.features = {}
        self.__doEvaulate()

    def putIndId(self,indid):
        if indid not in self.indids:
            self.indids.append(indid)

    def __doEvaulate(self):
        '''
        计算物种的适应度等指标
        :return:
        '''
        for key, evoluator in self.pop.params.features.items():
            max, avg, min, stdev = collections.rangefeature(list(map(lambda id:self.pop.getInd(id)[key], self.indids)))
            self[key] = {}
            self[key]['max'] = max
            self[key]['average'] = avg
            self[key]['min'] = min
            self[key]['stdev'] = stdev

        # 按照适应度值排序
        self.indids.sort(key=lambda id: self.pop.getInd(id)['fitness'], reverse=True)

    def __getitem__(self, item):
        #if item in self.features.keys():
        if item in self.features:
            if self.features[item] is EvaluationValue:
                return self.features[item].value
            else:
                return self.features[item]

        return super(Specie,self).__getitem__(item)

    def __setitem__(self, key, value):
        #if key not in self.features.keys():
        if key not in self.features:
            self.features[key] = value

        if value is float:
            self.features[key] = EvaluationValue()
            self.features[key].append(value)
        else:
            self.features[key] = value

    def __str__(self):
        return '('+collections.listtostr(self.indids) + "):[" + collections.dicttostr(self.features) + "]"


#endregion