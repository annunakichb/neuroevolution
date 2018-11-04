import logging
import time, threading

from utils.properties import Registry
import evolution.agent as agent
from evolution.agent import IndividualType
from evolution.agent import Individual
from evolution.agent import Population
from evolution.montior import  Monitor

# 操作注册表
operationRegistry = Registry()

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

    #endregion


    #region 运行启动和终止

    def isTerminted(self):
        if self.curTime >= self.runParam.terminated.maxIterCount:
            return True,'超过最大迭代次数'+str(self.runParam.terminated.maxIterCount)

        if self.pop['feature']['max'] >= self.runParam.terminated.maxFitness:
            return True,'达到最大适应度('+ str(self.pop['feature']['max']) + '>=' + str(self.runParam.terminated.maxFitness)

        return False,''

    def run(self):
        # 启动
        self.reset()
        self.monitor.recordSessionBegin()

        # 显示参数信息
        self.monitor.recordParam('基因参数',self.popParam.genomeParam)
        self.monitor.recordParam('种群参数',self.popParam,'genomeParam')
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

        # 进行种群划分
        speciesMethod = agent.speciesType.find(self.popParam.species.method)
        if speciesMethod is not None:
            species = speciesMethod.execute(self)
            self.monitor.recordSpecies(species)

        # 启动进化线程
        self.thread = threading.Thread(target=self.loop, name='session'+self.taskxh)
        self.thread.start()

    #endregion

    #region 进化过程
    def loop(self,arg):
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
        return self.runParam.operations.text.spilt(',')
    #endregion

# 进化任务
class EvolutionTask:
    def __init__(self,count,popParam):
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


    def execute(self,runParam):
        '''
        任务执行  任务执行过程中会显示命令行创建和执行进度界面，可以输入命令来显示更多内容,输入help显示所有命令
        :param runParam: 运行参数
        :return:
        '''
        self.runParam = runParam
        self.monitor = Monitor(self)

        self.monitor.recordTaskBegin()
        for i in range(self.count):
            session = Session(self.popParam,self.runParam,self,i,monitor)

            monitor = session.run()
            monitor.command()

            self.sessions.append(session)

        self.monitor.recordTaskEnd()



