# -*- coding: UTF-8 -*-

import pybullet
import gym
import numpy as np
import os
import matplotlib.pyplot as plt
from domains.antbullet.prey import PreyController
from pybullet_envs.bullet import bullet_client
import pybullet_envs.gym_locomotion_envs
rendering = False
log_path = 'work/log'



# 它根据控制参数生成猎物位置、速度和方向，控制参数取不同的值，得到不同的复杂度
preyController = PreyController()

def fitness(ind,session):
    '''
    max(0,（初始距离猎物的距离 - 最终距离猎物的距离）/初始距离猎物的距离)
    :param ind:
    :param session:
    :return:
    '''

    samples = preyController.sample()
    prey_positions,prey_states = samples

    net = ind.getPhenome()

    psyclient = pybullet.connect(pybullet.DIRECT)
    env = gym.make("AntBulletEnv-v0")
    if rendering:
        env.render(mode="human")
    init_obs = env.reset()
    init_pos = np.array(list(env.env.robot.body_xyz))
    _obs = init_obs

    count = 0
    while 1:
        prey_position = prey_positions[count]

        input = np.array(list(_obs) + [1] + prey_positions[count] + list(prey_states[count]))
        action = net.activate(input)
        _obs, r, done, _ = env.step(action)
        count += 1
        if count < 50 and not done: continue
        # posture = Posture(_obs)
        posture = np.array(list(env.env.robot.body_xyz))
        ind['lastpos'] = posture
        init_dis = np.sqrt(np.sum(np.square(init_pos - prey_position)))
        resu_dis = np.sqrt(np.sum(np.square(posture - prey_position)))
        pybullet.disconnect()
        return 0.0 if resu_dis > init_dis else 1- resu_dis / init_dis


import ne.callbacks as callbacks
import  gc
import csv
from brain.viewer import NetworkView

# 整个运行由若干个复杂度阶段组成，每个复杂度阶段又由若干运行轮次组成
# 每个复杂度阶段结束的条件(当达到以后则提升复杂度，进入下一轮)：
MAX_FITNESS = 0.75               # 每个复杂度阶段达到最大适应度
MAX_STATE_EPOCH = 20             # 每个复杂度执行的轮数

# 每个阶段的最大适应度和复杂度记录,每个记录是由适应度list和复杂度float值构成的元组
state_records = []


# 当前阶段的信息
cur_state_no = 0                                        # 当前阶段编号
cur_state_maxfitnesses_per_epoch = []                   # 当前阶段中每轮的所有最大适应度值
cur_state_maxfitness = 0.                               # 当前阶段出现的最大适应度
cur_state_maxfitness_ind = None                         # 当前阶段出现的最优个体

epoch_no = 0
mode = 'noreset'

maxfitness_records = []
# 每隔都少代提升一次复杂度
generation_promote_complex = 50
# 上次提升是第几代
last_promote_generation = 0
# 复杂度等级
complex_level = 1

def callback(event,monitor):
    callbacks.neat_callback(event,monitor)
    global cur_state_maxfitness
    global cur_state_maxfitness_ind
    global cur_state_maxfitnesses_per_epoch
    global cur_state_no
    global epoch_no
    global mode

    if event == 'epoch.end':
        gc.collect()
        epoch_no += 1

        # 记录这轮的最大适应度
        maxfitness = monitor.evoTask.curSession.pop.inds[0]['fitness']
        maxfitness_ind = monitor.evoTask.curSession.pop.inds[0]
        if maxfitness > cur_state_maxfitness:
            cur_state_maxfitness = maxfitness
            cur_state_maxfitness_ind = maxfitness_ind
        cur_state_maxfitnesses_per_epoch.append(maxfitness)

        # 如果最大适应度达到了env.max_notdone_count（说明对当前环境已经产生适应），或者进化迭代数超过10次（当前环境下适应达到最大）
        # 则提升复杂度
        if maxfitness >= MAX_FITNESS or epoch_no >= MAX_STATE_EPOCH:
            # 保存复杂度和对应最大适应度记录
            state_records.append((cur_state_maxfitnesses_per_epoch,preyController.current_complex))
            print([(max(r[0]),r[1]) for r in state_records])

            # 保存过程中每一步的最大适应度记录
            filename = log_path + os.sep + 'neat_' + mode + '.csv'
            out = open(filename, 'a', newline='')
            csv_write = csv.writer(out, dialect='excel')
            csv_write.writerow(state_records[-1])

            # 保存最优基因网络拓扑
            try:
                filename = log_path + os.sep + 'neat_' + mode + '_' + str(cur_state_no) + '_' + str(
                    maxfitness_ind.id) + '_' + \
                           str(maxfitness) + '.svg'
                netviewer = NetworkView()
                netviewer.drawNet(maxfitness_ind.genome, filename=filename, view=False)
            except RuntimeError as err:
                pass

            # 清空当前状态
            cur_state_no += 1
            cur_state_maxfitness = 0.
            cur_state_maxfitness_ind = None
            cur_state_maxfitnesses_per_epoch = []
            epoch_no

            # 提升复杂度
            changed = preyController.promote()
            if changed:
                print('复杂度提升,state_no=', str(cur_state_no), ',complex=', preyController.current_complex)
            else:
                maxfitness_records = [max(s[0]) for s in state_records]
                complex_records = [s[1] for s in state_records]
                plt.plot(complex_records, maxfitness_records, label='times')
                plt.xlabel('complexes')
                filename = log_path + os.sep + 'neat_' + mode + '.png'
                plt.savefig(filename)
                return

    elif event == 'session.end':
        pass
        #filename = 'singlecartpole.session.'+ str(monitor.evoTask.curSession.taskxh)+'.mov'
        #eliest = monitor.evoTask.curSession.pop.eliest
