
from enum import Enum
from utils import collections
from utils.properties import Registry
from utils.strs import ExtendJsonEncoder

import brain.models as models

__all__ = ['EndCondition','NeuralNetworkTask','SimpleNeuralNetworkRunner','runners']
# 训练终止条件类型
class EndCondition(Enum):
    MAXEPOCH = 1,
    MINDICATOR = 2,
    MAXDICATOR = 4,
    MINERROR = 8,
    MAXERRORUNCHANGED = 16

# 神经网络训练任务
class NeuralNetworkTask:

    #region 测试指标名称
    #测试指标：测试样本数
    INDICATOR_TEST_COUNT = 'testCount'
    #测试指标：测试正确数
    INDICATOR_CORRECT_COUNT = 'correctCount'
    # 测试指标：准确率
    INDICATOR_ACCURACY = 'accuracy'
    # 测试指标：MAE
    INDICATOR_MEAN_ABSOLUTE_ERROR = 'MAE'
    # 测试指标：MSE
    INDICATOR_MEAN_SQUARED_ERROR = 'MSE'
    #endregion

    #region 初始化

    def __init__(self,train_x=[],train_y=[],test_x=[],test_y=[]):
        '''
        神经网训练任务
        :param train_x:   训练样本
        :param train_y:   训练标签
        :param test_x:    测试样本
        :param test_y:    测试标签
        '''
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.test_result = []*len(test_y)
        self.test_stat = {}

    # 记录测试结果
    def setTestResult(self,i,test_result,doStat=False):
        '''
        记录测试结果
        :param i 样本序号
        :param test_result:  测试结果
        :return: None
        '''
        self.test_result[i] = test_result
        if doStat:
            self.__doTestStat__()
    #endregion

    #region 统计
    def __doTestStat__(self):
        '''
        对测试结果进行统计
        :return: None
        '''
        testcount = correctcount = 0
        mae = mse = 0.0
        for index,result in enumerate(self.test_result):
            if collections.equals(self.test_result[index],self.test_y[index]):
                correctcount += 1
            mae += abs(self.test_result[index]-self.test_y[index])
            mse += pow(self.test_result[index]-self.test_y[index],2)
            testcount += 1

        self.test_stat[NeuralNetworkTask.INDICATOR_TEST_COUNT] = testcount
        self.test_stat[NeuralNetworkTask.INDICATOR_CORRECT_COUNT] = correctcount
        self.test_stat[NeuralNetworkTask.INDICATOR_ACCURACY] = testcount/correctcount
        self.test_stat[NeuralNetworkTask.INDICATOR_MEAN_ABSOLUTE_ERROR] = mae / testcount
        self.test_stat[NeuralNetworkTask.INDICATOR_MEAN_SQUARED_ERROR] = mse / testcount

    def __setitem__(self, key, value):
        self.test_stat[key] = value

    def __getitem__(self, item):
        if item in self.test_stat.keys():return self.test_stat[item]
        return super.__getitem__(item)
    #endregion


# 简单前馈神经网络运行期
class SimpleNeuralNetworkRunner:
    def __init__(self):
        '''
        简单前馈神经网络运行器（要求神经网络无自连接，无循环，全部都是前馈连接）
        '''
        pass

    def doLearn(self,net,task):
        '''
        未实现
        :param net:
        :param task:
        :return:
        '''
        pass

    def doTest(self,net,task):
        '''
        执行测试
        :param net:  测试网络
        :param task: 测试任务
        :return: None
        '''
        # 取得输入
        inputNeurons = net.getInputNeurons()
        #对每一个输入样本
        for index,value in enumerate(task.test_x):
            # 重置神经元和突触状态
            collections.foreach(net.getNeurons(),lambda n:n.reset())
            collections.foreach(net.getSynapses(),lambda s:s.reset())

            # 设置输入
            for d,v in enumerate(value):
                if d >= len(inputNeurons):break
                model = models.nervousModels.find(inputNeurons[d].modelConfiguration.modelid)
                model.execute(inputNeurons[d],net,value=v)

                s = net.getOutputNeurons(inputNeurons[d].id)
                if collections.isEmpty(s):continue

                model = models.nervousModels.find(s.modelConfiguration.modelid)
                model.execute(s,net)

            # 反复执行
            ns = net.getNeurons()
            neuronCount = net.getNeuronCount()
            iterCount = 0
            outputNeurons = net.getOutputNeurons()
            while not collections.all(outputNeurons,lambda n:'value' in outputNeurons.states.keys()) and iterCount<=neuronCount:
                iterCount += 1
                uncomputeNeurons = collections.findall(ns,lambda n:'value' not in n.state.keys())
                if collections.isEmpty(uncomputeNeurons):break
                for n in uncomputeNeurons:
                    model = n.getModel()
                    synapses = net.getSynapses(toId = n.id)
                    if collections.isEmpty(synapses):continue
                    if not collections.all(synapses,lambda s:'value' in s.state.keys):continue
                    model.execute(n,net)

                    synapses = net.getSynapses(fromId = n.id)
                    if collections.isEmpty(synapses):continue
                    collections.foreach(synapses,lambda s:s.getModel().execute(s,net))

            # 将没结果的输出神经元的值设置为0
            collections.foreach(outputNeurons,lambda n:exec("n['value']=0"))
            # 取得结果
            outputs = list(map(lambda n:n['value'],outputNeurons))
            task.setTestResult(index,outputs)

        task.__doTestStat__()

# 基于tf的神经网络运行器
class TensorflowNeuralNetworkRunner:
    pass



# 运行器注册
runners = Registry()
runners.register(SimpleNeuralNetworkRunner(),'simple')

ExtendJsonEncoder.ignoreTypes.append(NeuralNetworkTask)
