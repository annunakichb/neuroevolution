
from utils.properties import Registry
import utils.collections as collections

#region 个体信息

#个体类型
class IndividualType:
    def __init__(self,name,genomeType,genomeFactory,phenomeType,genomeDecoder):
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

        return indMeta.genomeDecoder.create(self)

    def __getitem__(self, item):
        if item in self.features.keys():
            return self.features[item].value
        return super.__getitem__(item)

    def __setitem__(self, key, value):
        self.features[key] = value

#endregion

#region 种群信息
class Population:
    def __init__(self,params,initInds):
        self.params = params
        self.inds = initInds
        self.features = {}
        self.eliest = []

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
        inds = genomeFactory.create(popParam)
        return Population(popParam,inds)

    def evaulate(self,session):
        '''
        对种群进行评估
        :param session:
        :return:
        '''
        # 遍历每一个评估项
        for key,evoluator in self.params.features.items():
            # 对每一个个体计算评估值
            for ind in self.inds:
                value = evoluator.calacute(ind,session)
                ind[key] = value
            # 计算所有个体评估值的平均，最大和最小
            max,avg,min = collections.rangefeature(map(lambda i:i[key],self.inds))
            self[key] = {}
            self[key]['max'] = max
            self[key]['average'] = avg
            self[key]['min'] = min
        # 按照适应度值排序
        self.inds.sort(key=lambda ind:ind['fitness'],reverse=True)

        # 记录精英个体id
        eliestCount = int(session.popParam.elitistSize) if session.popParam.elitistSize >= 1 else session.popParam.elitistSize * session.popParam.size
        self.eliest = self.inds[0:eliestCount] if eliestCount>0 else []

    def __getitem__(self, item):
        if item in self.features.keys():
            return self.features[item].value
        return super.__getitem__(item)

    def __setitem__(self, key, value):
        self.features[key] = value

#endregion