import json
import time
from prompt_toolkit.shortcuts import get_input

import utils.collections as collections
import utils.files as files

class Monitor:

    #region 初始化
    def __init__(self,evoTask):
        self.evoTask = evoTask

    def reset(self):
        pass

    #endregion

    #region 记录日志
    def getLogFileName(self):
        return files.getFullFileName('task.log','/log')

    def __recordSection(self,stageName,**kwdatas):
        contents = ['阶段名称:'+stageName,
                    '进化时间:'+str(int(self.evoTask.curSession.curTime)),
                    '实际时间:'+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))]
        keys = [] if collections.isEmpty(kwdatas) else kwdatas.keys()
        for key in keys:
            contents.append(key + ':' + kwdatas[key])
        files.appendLines(self.getLogFileName(),contents)


    def recordSessionBegin(self):
        self.__recordSection('Session开始',编号=str(self.evoTask.curSession.taskxh))

    def recordParam(self,name,dict,ignoreKeys):
        encode_json = json.dumps(obj=dict,skipkeys=ignoreKeys)
        self.__recordSection('记录参数',name=encode_json)

    def recordPopulationCreate(self,pop,exception=None):
        self.__recordSection('种群创建',完成情况='成功' if exception is None else str(exception),个体数量=len(pop.inds))

    def recordPopulationFeatures(self):
        kwdatas = {}
        keys = self.evoTask.curSession.pop.features.keys()
        for key in keys:
            kwdatas[key] = 'max='+ self.evoTask.curSession.pop[key]['max'] +\
                           ",avg="+self.evoTask.curSession.pop[key]['average']+\
                           ',min='+self.evoTask.curSession.pop[key]['min']

        self.__recordSection('种群特征',**kwdatas)

    def recordIndividuals(self):
        logInds = self.__getLogIndividuals()
        if collections.isEmpty(logInds):return
        kwdatas = {}
        for ind in logInds:
            kwdatas['ind'+ind.id] = str(ind.genome)
        self.__recordSection('重要个体',**kwdatas)


    def __getLogIndividuals(self):
        logMode = self.evoTask.curSession.runParam.log.individual
        if logMode.lower() == 'all':
            return self.evoTask.curSession.pop.inds
        elif logMode.lower() == 'elite':
            return self.evoTask.curSession.pop.eliest
        elif logMode.lower() == 'maxfitness':
            return self.evoTask.curSession.pop.inds[0]
        elif logMode.lower() == 'custom':
            return None  # 没有实现


    def recordSpecies(self,species):
        return #raise RuntimeError('没有实现')

    def recordEpochBegin(self):
        # 写迭代开始标记
        s1 = "##########################################################################################"
        s = '第'+str(int(self.evoTask.curSession.curTime))+'次迭代'
        s2 = '#'*((len(s1)-len(s))/2) + s
        while len(s2) < len(s1):
            s2 += "#"
        s = [s1,s2,s1]
        files.appendLines(self.getLogFileName(),s)

    def recordOperation(self,operation,result):
        self.__recordSection('执行操作='+operation.name,result=str(result.msg))


    def recordEpochEnd(self):
        self.__recordSection('迭代结束')

    def recordSessionEnd(self,msg):
        self.__recordSection('session结束',结束原因=msg)

    def recordTaskBegin(self):
        self.__recordSection('任务开始')

    def recordTaskEnd(self):
        self.__recordSection('任务结束')

    #endregion

    #region 接收命令
    def command(self):
        while 1:
            answer = get_input('>')
            if answer.lower() in ['quit','exit']:
                pass


    #endregion
