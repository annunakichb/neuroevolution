# -*- coding: UTF-8 -*-

import pybullet
import gym
from domains.antbullet.ant import Posture
import numpy as np

rendering = False

maxv = 0

# 猎物位置,移动速度和方向
obj_pos = [80,80,0]
obj_speed = [0,0,0]
obj_direction = [0,0,0]


def fitness(ind,session):
    global maxv
    net = ind.getPhenome()

    pybullet.connect(pybullet.DIRECT)
    env = gym.make("AntBulletEnv-v0")
    if rendering:
        env.render(mode="human")
    init_obs = env.reset()
    init_posture = np.array(list(env.env.robot.body_xyz))
    #init_posture = Posture(init_obs)
    _obs = init_obs

    count = 0
    while 1:
        input = np.array(list(_obs)+[1]+obj_pos+obj_speed+obj_direction)
        action = net.activate(input)
        _obs, r, done, _ = env.step(action)
        count += 1
        if count < 100 or not done: continue
        #posture = Posture(_obs)
        posture = np.array(list(env.env.robot.body_xyz))
        v = np.sqrt(np.sum(np.square(posture-init_posture)))
        if v > maxv:
            maxv = v
            print('当前最大移动距离=',maxv,'cur pos=',posture,'init pos=',init_posture)
        return np.sqrt(np.sum(np.square(posture-init_posture)))

