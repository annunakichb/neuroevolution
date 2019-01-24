from domains.ne.cartpoles.cartpole import SingleCartPoleEnv
import domains.ne.cartpoles.cartpole as cartpole

import ne.callbacks as callbacks
import ne.neat as neat
from brain.networks import NetworkType
from brain.networks import NeuralNetwork
from brain.runner import NeuralNetworkTask
from evolution.env import Evaluator
from evolution.session import EvolutionTask
from utils.properties import Properties

from domains.ne.cartpoles.force import  *

'''
force_inc_generator = ForceIncGenerator(0,1,force_step = 1)
force_normal_generator = ForceNormalGenerator(9,2,force_init_sigma = 0.01,force_sigma_step = 0.05)
linearGenerator = LinearGenerator(0,3)
cycleGenerator = CycleGenerator(0,4,time_step=0.02,wind_range=24.0)
complex_generation = 0
'''


force_generator = ForceGenerator(15,0.5,0.0,0.01)
# 适应度计算函数:以累计奖励作为适应度
def fitness(ind,session):
    net = ind.getPhenome()

    fitnesses = []
    runs_per_net = 10
    env = SingleCartPoleEnv()

    episode_reward = 0  # 累计奖励
    observation = env.reset()

    global complex_generation
    for runs in range(runs_per_net):
        # 网络执行
        net.definition.runner.task.test_x = [observation]
        action = net.doTest()
        action = 1 if action[0] > 0.5 else 0
        env.wind = force_generator.next(runs*0.02)
        observation_, reward, done, info = env.step(action, runs)

        # x是车的水平位移，theta是杆离垂直的角度
        x, x_dot, theta, theta_dot = observation_

        # 计算奖励
        reward = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        episode_reward += reward


    return episode_reward / runs_per_net


def fitness2(ind,session):
    '''
    以连续不倒的次数作为适应度
    :param ind:
    :param session:
    :return:
    '''
    env = SingleCartPoleEnv()
    net = ind.getPhenome()

    done_count = 0  # 表示连续维持成功(done)的最大次数
    cur_max_done = 100  # 当前最大连续不倒次数

    episode_reward = 0  # 累计奖励
    observation = env.reset()

    global complex_generation

    step = 0
    while True:
        # 网络执行
        net.definition.runner.task.test_x = [observation]
        env.wind = force_generator.next(step * 0.02)
        action = net.doTest()
        action = 1 if action[0] > 0.5 else 0
        observation_, reward, done, info = env.step(action, step)

        # x是车的水平位移，theta是杆离垂直的角度
        x, x_dot, theta, theta_dot = observation_

        # 计算奖励
        reward = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        episode_reward += reward

        if done:
            return done_count
        done_count += 1

        if done_count >= cur_max_done: # 连续不倒下次数最大是30
            return done_count

        step += 1

# 记录最优个体的平衡车运行演示视频
def callback(event,monitor):
    callbacks.neat_callback(event,monitor)
    global complex_generation

    if event == 'epoch.end':
        maxfitness = monitor.evoTask.curSession.pop.inds[0]['fitness']
        if maxfitness >= 100:
            changed, maxcomplex, k,w, f, sigma = force_generator.promptComplex()
            if changed:
                print('环境复杂度=%.3f,k=%.2f,w=%.2f,f=%.2f,sigma=%.2f' % (maxcomplex, k,w, f, sigma))
            else:
                print('环境复杂度没有变化')

    elif event == 'session.end':
        filename = 'singlecartpole.session.'+ str(monitor.evoTask.curSession.taskxh)+'.mov'
        eliest = monitor.evoTask.curSession.pop.eliest
        cartpole.make_movie(eliest[0].getPhenome(),filename)

def run():
    # 初始化neat算法模块
    neat.neat_init()

    # 定义网络训练任务

    task = NeuralNetworkTask()

    # 定义网络
    netdef = {
        'netType' : NetworkType.Perceptron,                       # NetworkType，网络类型,必须
        'neuronCounts' : [4,1],                                   # list（初始）网络各层神经元数量,必须
        'idGenerator' :  'neat',                                  # str 生成网络，神经元，突触id的类，参见DefauleIDGenerator,list idgenerator命令可以列出所有的id生成器对象
        'config' : {
            'layered' : True,                                     # bool 是否分层,可选
            'substrate' : True,                                   # bool 是否使用基座,可选
            'acyclie' : False,                                    # bool 是否允许自身连接,可选
            'recurrent':False,                                    # bool 是否允许同层连接,可选
            'reversed':False,                                     # bool 是否允许反向连接,可选
            'dimension':2,                                        # int 空间坐标维度,可选
            'range':NeuralNetwork.MAX_RANGE,                      # list 坐标范围，可选'
        },
        'runner':{
            'name' : 'simple',                                    # str 网络运行器名称,必须
            'task' : task,                                        # NeuralNetworkTask,网络运行任务,必须
        },
        'models':{                                                # dict 神经元计算模型的配置信息,必须
            'input':{                                             # str 模型配置名称（不是模型名称）
                'name' : 'input',                                 # str,名称，与上面总是一样,可选
                'modelid':'input',                                # str，模型id，必须，用这个来找到对应的计算模型对象,因此应确保该计算模型已注册
            },
            'hidden':{
                'name':'hidden',                                  # str 隐藏神经元配置名称，可选
                'modelid':'hidden',                               # str 隐藏神经元计算模型id,必须
                'activationFunction':{                            # dict 可选
                    'name' : 'sigmod',                            # str 激活函数名称，必须
                    'a' :1.0,'b':1.0,'T':1.0                      # float 激活函数参数，可选
                },
                'bias':'uniform[-30.0:30.0]',                     # str 隐藏神经元的偏置变量，均匀分布，必须，可以是uniform[begin,end]或者normal(u,sigma)
            },
            'synapse':{
                'name':'synapse',                                 # str 突触计算模型配置名称,可选
                'modelid':'synapse',                              # str 突触计算模型Id，必须
                'weight':'uniform[-30.0:30.0]'                    # str 突触学习变量，均匀分布，必须
            }
        }
    }


    # 定义种群
    popParam = {
        'indTypeName' : 'network',                                #种群的个体基因类型名，必须，该类型的个体基因应已经注册过，参见evolution.agent,必须
        'genomeFactory':None,                                     #基因工厂，个体类型中已经提供了基因工厂对象，这里如果设置，可以替换前者，可选
        'factoryParam' :{                                         # 工厂参数，必须
           'connectionRate':1.0,                                  # 连接比率
        },
        'genomeDefinition' : netdef,                              #基因定义参数,可选
        'size':100,                                               #种群大小，必须
        'elitistSize':0.05,                                        #精英个体占比，小于1表示比例，大于等于1表示数量
        'species':{                                               #物种参数，可选
            'method':'neat_species',                              # 物种分类方法,在物种参数中必须
            'alg':'kmean',                                        # 算法名称
            'size': 5,                                            # 物种个体数量限制，0表示无限制或动态
            'iter':50,                                            # 算法迭代次数
        },
        'features':{                                              # 特征评估函数配置，必须
            'fitness' : Evaluator('fitness',[(fitness2,1.0)])      # 适应度评估器,如果评估器只包含一个函数,也可以写成Evaluator('fitness',fitness)
        }
    }


    # 定于运行参数
    runParam = {
        'terminated' : {
            'maxIterCount' : 100,                                 # 最大迭代次数，必须
            'maxFitness' : 1000,                                   # 最大适应度，必须
        },
        'log':{
            'individual' : 'elite',                                 # 日志中记录个体方式：记录所有个体，可以选择all,elite,maxfitness（缺省）,custom
            'debug': False                                        # 是否输出调试信息
        },
        'evalate':{
            'parallel':0,                                         # 并行执行评估的线程个数，缺省0，可选
        },
        'operations':{
            #'method' : 'neat',                                   # 已有的进化操作序列名称，与text两个只用一个
            'text' : 'neat_selection,neat_crossmate,neat_mutate'  # 进化操作序列
        },
        'mutate':{
            'propotion' : 0.1,                                      # 变异比例,有多少个个体参与变异，小于等于1表示比例，大于1表示固定数量
            'model':{
                'rate' : 0.0,                                     # 模型变异比例
                'range' : ''                                      # 可选的计算模型名称，多个用逗号分开，缺省是netdef中所有模型
            },
            'activation':{
                'rate' : 0.0,                                     # 激活函数的变异比率
                'range':'sigmod'                                  # 激活函数的
            },
            'topo' : {
                'addnode' : 0.4,                                  # 添加节点的概率
                'addconnection':0.4,                              # 添加连接的概率
                'deletenode':0.1,                                 # 删除节点的概率
                'deleteconnection':0.1                            # 删除连接的概率
            },
            'weight':{
                'epoch':3,                                          # 权重调整次数
            }
        }

    }

    evolutionTask = EvolutionTask(1,popParam,callback)
    evolutionTask.execute(runParam)






if __name__ == '__main__':
    run()