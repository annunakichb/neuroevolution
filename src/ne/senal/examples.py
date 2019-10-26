

####################################################
# 机械手学习推动一个箱子一端距离后松手，让其滑动到指定目标    #
# 地面摩擦力和推动距离固定但是未知                       #
# 以箱子初始位置为原点，箱子移动方向为坐标轴正向            #
# 网络输入为箱子位置（只有初始和最终停止），目标位置，力大小  #
# 网络输出为力大小                                    #
####################################################
from ne.senal.box import BoxGene
import brain
import evolution
from evolution.session import Session
from evolution.session import EvolutionTask
import ne.senal as senal

# 机械手推动一个箱子一段固定距离后松手，地面摩擦力、箱子质量、推动距离都未知，机械手能观察到箱子初始位置、箱子停止位置和目标位置
# 任务要求机械手能推动箱子到目标最近的位置
#region 各种事件
def handle_ind_begin(ind,session,**kwargs):
    '''
    个体的初始事件
    :param ind:     Indiviaual  个体
    :param session: Session     会话
    :param kwargs:  dict        预留参数
    :return:
    '''
    net = ind.getPhenome()
    net.reset()

    genome = ind.genome
    box = genome.find_box_by_expression('s')

    net.clock = 0
    net.activate([(box,0.)])

def handle_ind_action(ind,session,**kwargs):
    '''
    个体动作执行
    :param ind:
    :param session:
    :param kwargs:
    :return:
    '''
    net = ind.getPhenome()
    genome = ind.genome
    f_ = kwargs['actions'][0]

    # 力量感知事件
    box = genome.find_box_by_expression('f')
    net.clock += 1
    net.activate([(box, f_)])

    # 隐参数
    hidden_box_m = 1. # 箱子质量，千克
    hidden_dis = 0.5   # 隐变量，推动距离，米
    hidden_friction = 0.3 # 摩擦力，牛

    # 计算停止位置
    a = f_ / hidden_box_m # 加速度
    v0 = a * hidden_dis   # 松手时的初速
    a = hidden_friction / hidden_box_m # 减速度
    dis = v0 / a          # 移动距离
    box = genome.find_box_by_expression('s')
    net.clock += 1
    net.activate([(box, dis)])

def handle_set_target(ind,session,**kwargs):
    '''
    设定外部目标
    :param ind:
    :param session:
    :param kwargs:
    :return:
    '''
    target_pos = 0.8

    net = ind.getPhenome()
    genome = ind.genome
    box = genome.find_box_by_expression('s')
    box.expect = target_pos
#endregion



#region SENAL算法初始化
senal.senal_init()

# 定义网络参数，主要是输入和输出
netdef = brain.createNetDef(neuronCounts=[3,1])
netdef.inputboxs = [
    {'expression':'f','initsize':1,'group':'agent.hand.output','attributes':{'clip':[0.0,1.0],'caption':'推力'}},  # 对力量输出的感知
    {'expression':'s','initsize':1,'group':'env.s','attributes':{'caption':'箱子位置'}},   # 对箱子位置的感知
    {'expression':'o','initsize':1,'group':'env.o','attributes':{'caption':'目标位置'}}    # 对目标位置的感知
]
netdef.outputboxs = [{'expression':'f_','initsize':1,'group':'hand','attributes':{}}]

#endregion


#region 种群参数和运行参数
popParam = evolution.createPopParam(size=100,elitistSize=0.05,genomeDefinition=netdef)
runParam = evolution.createRunParam(1000,1.0,'activity',
    activity={
        'stability_threshold':0.8,'stability_resdual':0.2,
        'stability_action_count':3
    },
    handlers = {
        'ind.begin':handle_ind_begin,
        'ind.action':handle_ind_action,
        'env.target':handle_set_target
    }
)
#endregion


#region 启动进化环境
evo = EvolutionTask(1,popParam,None)
evo.execute(runParam)
#endregion







