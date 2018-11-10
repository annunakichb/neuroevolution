import logging
import time, threading

from utils.properties import Registry
from utils.properties import Properties
import evolution.agent as agent
from evolution.agent import IndividualType
from evolution.agent import Individual
from evolution.agent import Population
from evolution.montior import  Monitor
from evolution.env import EvaluationValue
from evolution.env import Evaluator

__all__ = ['operationRegistry','operationGraphis','Session','EvolutionTask']


# 操作注册表
operationRegistry = Registry()
# 操作顺序图注册表
operationGraphis = Registry()

# 进化session
class Session:
    #region 初始化
    def __init__(self,popParam,runParam,evolutionTask,taskxh,monitor):
        '''
        进化session
        :param evolutionTask:  任务，任务是指session的多次运行
        :param taskxh:         任务序号
        :param popParam:       种群参数
        :param runParam:       运行参数
        '''
        self.curTime = 0
        self.popParam = popParam
        self.runParam = runParam

        self.pop = None

        self.evolutionTask = evolutionTask
        self.taskxh = taskxh
        self.thread = None
        self.monitor = monitor

        self.popRecords = []
        self.operationRecords = {}
        self.exitCode = 0
        self.exitMsg = ''


    def reset(self):
        '''
        状态重置
        :return:
        '''
        self.curTime = 0
        self.monitor.reset()
        self.operationRecords = {}
        self.popRecords = []

    #endregion


    #region 运行启动和终止

    def isTerminted(self):
        if self.curTime >= self.runParam.terminated.maxIterCount:
            return True,'超过最大迭代次数'+str(self.runParam.terminated.maxIterCount)

        if self.pop['fitness']['max'] >= self.runParam.terminated.maxFitness:
            return True,'达到最大适应度('+ str(self.pop['fitness']['max']) + '>=' + str(self.runParam.terminated.maxFitness)

        return False,''

    def run(self):
        # 启动
        self.reset()
        self.monitor.recordSessionBegin()

        # 显示参数信息
        # self.monitor.recordParam('基因参数',self.popParam.genomeDefinition,'task')
        self.monitor.recordParam('种群参数',self.popParam,['task'])
        self.monitor.recordParam('运行参数',self.runParam)

        # 创建种群
        try:
            self.pop = Population.create(self.popParam)
            self.monitor.recordPopulationCreate(self.pop)
        except RuntimeError as e:
            self.monitor.recordPopulationCreate(self.pop,e)

        # 计算种群特征
        self.pop.evaulate(self)
        self.monitor.recordPopulationFeatures()
        self.monitor.recordIndividuals()
        self.popRecords.append(self.__createPopRecord())

        # 进行种群划分
        speciesMethod = agent.speciesType.find(self.popParam.species.method)
        if speciesMethod is not None:
            species = speciesMethod.execute(self)
            # self.monitor.recordSpecies(species)

        # 启动进化线程
        #self.thread = threading.Thread(target=self.loop, name='session'+self.taskxh)
        #self.thread.start()
        self.loop()

    #endregion

    #region 进化过程
    def loop(self,arg=None):
        # 循环直到满足终止条件
        while 1:
            termited,reason = self.isTerminted()
            if termited:
                # session结束
                self.monitor.recordSessionEnd(reason)
                return

            # 初始化epoch
            self.curTime += 1
            self.operationRecords = {}
            self.monitor.recordEpochBegin()

            # 遍历每一个操作并执行，目前不支持并发和分布式
            operationNames = self.getOperationNames()
            for operationName in operationNames:
                operation = operationRegistry.find(operationName)
                if operation is None:
                    self.monitor.recordError('运行失败，进化操作没有注册:'+operationName)
                    self.exitCode = -1
                    self.exitMsg = '运行失败，进化操作没有注册:'+operationName
                    return;
                try:
                    result = operation.execute(self)
                    self.operationRecords[operationName] = result
                    self.monitor.recordOperation(operation,result)
                except RuntimeError as e:
                    self.monitor.recordOperation(operation,result,e)
                    self.exitCode = -2
                    self.exitMsg = '运行失败，进化操作执行过程发生异常:' + operationName + ":" + str(e)
                    return;

            # 计算种群特征
            self.pop.evaulate(self)
            self.monitor.recordPopulationFeatures()
            self.monitor.recordIndividuals()

            # 进行种群划分
            speciesMethod = agent.speciesType.find(self.popParam.species.method)
            if speciesMethod is not None:
                species = speciesMethod.execute(self)
                self.monitor.recordSpecies(species)

            # 一次迭代结束
            self.monitor.recordEpochEnd()



    def getOperationNames(self):
        '''
        取得操作序列
        :return:
        '''
        return self.runParam.operations.text.split(',')

    def __createPopRecord(self):
        r = {}
        for featureKey in self.pop.params.features.keys():
            r[featureKey] = {}
            r[featureKey]['max'] = self.pop[featureKey]['max']
            r[featureKey]['average'] = self.pop[featureKey]['average']
            r[featureKey]['min'] = self.pop[featureKey]['min']
        return r
    #endregion

# 进化任务
class EvolutionTask:
    def __init__(self,count,popParam,callback):
        '''
        进化任务，一个进化任务是将多次执行进化，每次进化为一个session
        :param count:     int 运行次数
        :param popParam:  dict 种群参数
        '''
        self.count = count
        self.popParam = popParam

        self.runParam = None
        self.sessions = []
        self.curSession = None

        self.monitor = None
        self.callback = callback

        self.__verifyParam()

    def execute(self,runParam):
        '''
        任务执行  任务执行过程中会显示命令行创建和执行进度界面，可以输入命令来显示更多内容,输入help显示所有命令
        :param runParam: 运行参数
        :return:
        '''
        self.runParam = runParam
        self.monitor = Monitor(self,callback = self.callback)

        self.monitor.recordTaskBegin()
        for i in range(self.count):
            session = Session(self.popParam,self.runParam,self,i,self.monitor)
            self.curSession = session

            monitor = session.run()
            monitor.command()

            self.sessions.append(session)

        self.monitor.recordTaskEnd()

    def __verifyParam(self):
        if self.popParam is None:
            raise  RuntimeError('进化任务对象的popParam不能为空');
        if not isinstance(self.popParam,Properties):
            raise RuntimeError('进化任务对象的popParam类型必须为Properties');




