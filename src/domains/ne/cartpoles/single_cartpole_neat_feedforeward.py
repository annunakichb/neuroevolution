from domains.ne.cartpoles.cartpole import SingleCartPole
import domains.ne.cartpoles.cartpole as cartpole

import ne
import ne.neat as neat
from brain.networks import NetworkType
from brain.networks import NeuralNetwork
from brain.runner import NeuralNetworkTask
from evolution.env import Evaluator
from evolution.session import EvolutionTask


# 适应度计算函数
def fitness(ind,session):
    net = ind.getPhenome()

    fitnesses = []
    runs_per_net = 5
    simulation_seconds = 60.0

    for runs in range(runs_per_net):
        sim = SingleCartPole()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            inputs = sim.get_scaled_state()
            net.definition.task.test_x = inputs
            action = net.doTest()

            # Apply action to the simulated cart-pole
            force = cartpole.discrete_actuator_force(action)
            sim.step(force)

            # Stop if the network fails to keep the cart within the position or angle limits.
            # The per-run fitness is the number of time steps the network can balance the pole
            # without exceeding these limits.
            if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                break

            fitness = sim.t
        fitnesses.append(fitness)
    return min(fitnesses)

# 记录最优个体的平衡车运行演示视频
def callback(event,monitor):
    ne.callbacks(event,monitor)
    if event == 'session.end':
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
        'neuronCounts' : [4,1,1],                                 # list（初始）网络各层神经元数量,必须
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
                'weight':'uniform[-30.0,30.0]'                    # str 突触学习变量，均匀分布，必须
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
        'elitistSize':0.2,                                        #精英个体占比，小于1表示比例，大于等于1表示数量
        'species':{                                               #物种参数，可选
            'method':'',                                          #物种分类方法
            'size':0                                              #物种个体数量限制，0表示无限制或动态
        },
        'features':{                                              # 特征评估函数配置，必须
            'fitness' : Evaluator('fitness',{fitness,1.0})        # 适应度评估器,如果评估器只包含一个函数,也可以写成Evaluator('fitness',fitness)
        }
    }

    # 定于运行参数
    runParam = {
        'terminated' : {
            'maxIterCount' : 1000000,                             # 最大迭代次数，必须
            'maxFitness' : 100,                                   # 最大适应度，必须
        },
        'log':{
            'individual' : 'all',                                 # 日志中记录个体方式：记录所有个体，可以选择all,elite,maxfitness（缺省）,custom
        },
        'evalate':{
            'parallel':0,                                         # 并行执行评估的线程个数，缺省0，可选
        },
        'operations':{
            #'method' : 'neat',                                   # 已有的进化操作序列名称，与text两个只用一个
            'text' : 'neat_selection,neat_crossmate,neat_mutate'  # 进化操作序列
        },
        'mutate':{
            'propotion' : 2,                                      # 变异比例,有多少个个体参与变异，小于等于1表示比例，大于1表示固定数量
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
          'epoch':5,                                          # 权重调整次数
        }
    }
    }


    evolutionTask = EvolutionTask(10,popParam,neat.callbacks.neat_callback)
    evolutionTask.execute(runParam)

if __name__ == '__main__':
    run()