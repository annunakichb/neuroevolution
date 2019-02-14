import json
import time
import  logging
from prompt_toolkit import Application
from prompt_toolkit.completion import Completer
from pygments.lexers import sql

import utils.collections as collections
import utils.files as files
import utils.strs as strs
from utils.collections import ExtendJsonEncoder


__all__ = ['Monitor']


class Monitor:

    #region 初始化
    def __init__(self,evoTask,callback=None):
        self.evoTask = evoTask
        self.operationResult = {}
        self.logfilename = self.evoTask.runParam.log.file
        if self.logfilename is None or self.logfilename == '':
            self.logfilename = 'evolution.log'

        self.logger = self.__createLogger()
        self.callback = callback
        self.debugEnabled = self.evoTask.runParam.log.debug

    def reset(self):
        pass

    def getOperationResult(self,name):
        return self.operationResult[name]

    def __createLogger(self):
        LOG_FORMAT = "%(message)s"
        logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT)


        logger = logging.getLogger('monitor')
        logger.setLevel(logging.DEBUG)

        f_handler = logging.FileHandler(self.logfilename)
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(f_handler)

        #s_Handler = logging.StreamHandler()
        #s_Handler.setLevel(logging.DEBUG)
        #s_Handler.setFormatter(logging.Formatter(LOG_FORMAT))
        #logger.addHandler(s_Handler)

        return logger


    #endregion

    #region 记录日志
    def getLogFileName(self):
        return files.getFullFileName('task.log','/log')

    def __recordSection(self,stageName,**kwdatas):
        contents = '事件 = '+stageName + \
                    ',年代 = '+('' if self.evoTask.curSession is None else str(int(self.evoTask.curSession.curTime))) + \
                    ',时间 = '+time.strftime('%H:%M:%S',time.localtime(time.time()))
        self.logger.info("")
        self.logger.info(contents)

        keys = [] if collections.isEmpty(kwdatas) else kwdatas.keys()
        for key in keys:
            self.logger.info(key + ':' + str(kwdatas[key]))



    def recordDebug(self,module,debugName,debugInfo):
        if not self.debugEnabled:return
        self.logger.debug('#调试信息('+str(module)+"):"+str(debugName)+":"+str(debugInfo))
        #self.logger.info("")

    def recordSessionBegin(self):
        self.__recordSection('Session开始',Session编号=str(self.evoTask.curSession.taskxh))
        if self.callback is not None:self.callback('session.begin',self)


    def recordParam(self,name,dict,ignoreKeys=[]):
        encode_json = json.dumps(obj=dict,skipkeys=ignoreKeys,cls=ExtendJsonEncoder,indent=True)
        kwdata = {name:encode_json}
        self.__recordSection('记录参数',**kwdata)
        if self.callback is not None: self.callback('param.record', self)

    def recordPopulationCreate(self,pop,exception=None):
        self.__recordSection('种群创建',完成情况='成功' if exception is None else str(exception),个体数量=len(pop.inds))
        if self.callback is not None: self.callback('pop.create', self)

    def recordPopulationFeatures(self):
        kwdatas = {}
        keys = self.evoTask.curSession.pop.features.keys()
        for key in keys:
            kwdatas[key] = 'max='+ str(self.evoTask.curSession.pop[key]['max']) +\
                           ",avg="+ str(self.evoTask.curSession.pop[key]['average'])+\
                           ',min='+ str(self.evoTask.curSession.pop[key]['min'])

        self.__recordSection('种群特征',**kwdatas)
        if self.callback is not None: self.callback('feature.record', self)

    def recordIndividuals(self):
        logInds = self.__getLogIndividuals()
        if collections.isEmpty(logInds):return
        kwdatas = {}
        for ind in logInds:
            kwdatas['ind'+str(ind.id)] = str(ind)
        self.__recordSection('重要个体',**kwdatas)
        if self.callback is not None: self.callback('inds.record', self)


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
        if species is None:species = []
        kwdatas = {'物种数量': len(species)}

        for sp in species:
            kwdatas['specie' + str(sp.id)] = str(sp)


        self.__recordSection('物种特征', **kwdatas)
        if self.callback is not None: self.callback('species.record', self)

    def recordEpochBegin(self):
        # 写迭代开始标记
        s1 = "##########################################################################################"
        s = '第'+str(int(self.evoTask.curSession.curTime))+'次迭代'
        s2 = '#'*int(((len(s1)-len(s))/2)) + s
        while len(s2) < len(s1):
            s2 += "#"
        s = [s1,s2,s1]
        self.logger.info("")
        self.logger.info(s1)
        self.logger.info(s2)
        self.logger.info(s1)
        if self.callback is not None: self.callback('epoch.begin', self)

    def recordOperation(self,operation,result,exeception=None):
        self.__recordSection(operation.name,result=str(result[1]))
        self.operationResult[operation.name] = result
        if self.callback is not None: self.callback('operation.completed', self)


    def recordEpochEnd(self):
        self.__recordSection('本次迭代结束')
        if self.callback is not None: self.callback('epoch.end', self)

    def recordSessionEnd(self,msg):
        self.__recordSection('session结束',结束原因=msg)
        if self.callback is not None: self.callback('session.end', self)

    def recordTaskBegin(self):
        self.__recordSection('任务开始')
        if self.callback is not None: self.callback('task.begin', self)

    def recordTaskEnd(self):
        self.__recordSection('任务结束')
        if self.callback is not None: self.callback('task.end', self)

    #endregion

