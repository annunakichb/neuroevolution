#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import time
import os
import csv

import matplotlib.pyplot as plt
from domains.ne.cartpoles.enviornment.cartpole import SingleCartPoleEnv
from domains.ne.cartpoles.enviornment import force
from rl.ddqn import  DDeepQNetwork

import domains.ne.cartpoles.enviornment.runner as runner

env = SingleCartPoleEnv().unwrapped
RL = DDeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0])

def _do_learn(observation,action,reward,observation_,step,totalreward,total_step):
    RL.store_transition(observation, action, reward, observation_)
    if total_step > 10:
        RL.learn()

def train():
    complexes = []
    reward_list = []
    notdone_count_list = []
    steps = []

    episode_reward_list = []
    episode_notdone_count_list = []
    total_step = 0
    while True:
        # 执行一次
        notdone_count, episode_reward, step, total_step= runner.do_until_done(env,RL.choose_action,total_step,_do_learn)
        # 记录执行得到的奖励和不倒下次数
        episode_reward_list.append(episode_reward)
        episode_notdone_count_list.append(notdone_count)
        # 每执行100次,打印一下
        if total_step % 100 == 0 and total_step != 0:
            print("持续次数=", episode_notdone_count_list, ",平均=", np.average(episode_notdone_count_list))
            print("累计奖励=", episode_reward_list, ",平均=", np.average(episode_reward_list))

        # 判断是否可以提升复杂度
        if notdone_count > env.max_notdone_count or total_step >= 1500:
            # 记录复杂度和对应获得的奖励(平均还是最大)
            complexes.append(force.force_generator.currentComplex())
            reward_list.append(np.max(episode_reward_list))
            notdone_count_list.append(np.max(episode_notdone_count_list))
            steps.append(total_step)
            np.save('ddqn_result.npy', (complexes, notdone_count_list, reward_list,steps))
            print([(f, c) for f, c in zip(complexes, notdone_count_list)])

            # 记录过程记录
            filename = os.path.split(os.path.realpath(__file__))[0] + '\\datas\\ddqn.csv'
            out = open(filename, 'a', newline='')
            csv_write = csv.writer(out, dialect='excel')
            csv_write.writerow([complexes[-1]]+episode_notdone_count_list)

            episode_notdone_count_list,episode_reward_list = [],[]
            total_step = 0

            print('复杂度:', complexes)
            print('奖励:', reward_list)
            print("持续次数:", notdone_count_list)

            # 升级复杂度,为了加快执行速度,让复杂度增加幅度至少大于min_up
            changed, newcomplex, k, w, f, sigma = force.force_generator.promptComplex(5.0)
            if not changed:
                break  # 复杂度已经达到最大,结束
            print('新的环境复杂度=%.3f,k=%.2f,w=%.2f,f=%.2f,sigma=%.2f' % (newcomplex, k, w, f, sigma))
        else:

            episode_reward_list.append(episode_reward)
            if len(episode_reward_list) > 10:
                episode_reward_list = episode_reward_list[-10:]
            episode_notdone_count_list.append(notdone_count)
            if len(episode_notdone_count_list) > 10:
                episode_notdone_count_list = episode_notdone_count_list[-10:]

    #np.save('ddqn_result.npz', complexes, notdone_count_list, reward_list)
    RL.save()
    plt.plot(complexes, reward_list, label='reward')
    plt.plot(complexes, notdone_count_list, label='times')
    plt.xlabel('complexes')
    plt.savefig('./dqn_cartpole.png')


if __name__ == '__main__':
    train()




