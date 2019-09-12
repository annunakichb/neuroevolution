# -*- coding: UTF-8 -*-

import pybullet
import gym
from domains.antbullet.ant import Posture
import numpy as np
import os
import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from domains.antbullet.prey import PreyController
rendering = False
maxv = 0

# 每隔都少代提升一次复杂度
generation_promote_complex = 50
# 上次提升是第几代
last_promote_generation = 0
# 复杂度等级
complex_level = 1

# 猎物位置,移动速度和方向
obj_pos = [80,80,0]
obj_speed = [0,0,0]
obj_direction = [0,0,0]
env_complex = {}

preyController = PreyController()

def fitness(ind,session):
    '''
    距离猎物的距离
    :param ind:
    :param session:
    :return:
    '''
    global last_promote_generation
    global complex_level

    if (session.curTime - last_promote_generation) >= generation_promote_complex:
        preyController.promote()
        last_promote_generation = session.curTime
        complex_level += 1
        print('复杂度提升,level=',complex_level,',complex=',preyController.current_complex)
    samples = preyController.sample()
    prey_positions,_ = samples

    net = ind.getPhenome()

    pybullet.connect(pybullet.DIRECT)
    env = gym.make("AntBulletEnv-v0")
    if rendering:
        env.render(mode="human")
    init_obs = env.reset()
    init_posture = np.array(list(env.env.robot.body_xyz))
    # init_posture = Posture(init_obs)
    _obs = init_obs

    count = 0
    while 1:
        prey_position = prey_positions[count]

        input = np.array(list(_obs) + [1] + obj_pos + obj_speed + obj_direction)
        action = net.activate(input)
        _obs, r, done, _ = env.step(action)
        count += 1
        if count < 50 and not done: continue
        # posture = Posture(_obs)
        posture = np.array(list(env.env.robot.body_xyz))
        ind['lastpos'] = posture
        init_dis = np.sqrt(np.sum(np.square(init_posture - prey_position)))
        resu_dis = np.sqrt(np.sum(np.square(posture - prey_position)))
        return 0.0 if resu_dis > init_dis else 1- resu_dis / init_dis

def fitness2(ind,session):
    '''最大移动距离'''
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

