import pybullet
import gym
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from pybullet_envs.bullet import bullet_client
import pybullet_envs.gym_locomotion_envs
from domains.antbullet.ant import Posture

if __name__ == '__main__':
    psyclient = pybullet.connect(pybullet.DIRECT)
    env = gym.make("AntBulletEnv-v0")
    env.render(mode="human")
    init_obs = env.reset()
    init_posture = Posture(init_obs)
    init_posture.print()
    cur_posture = init_posture
    actions = [
        [-1,-1,-1,-1,1,1,1,1],
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [1,1,1,1,-1,-1,-1,-1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1]
    ]
    count = 0
    max = 100
    while 1:
        time.sleep(1. / 50)
        index = count % len(actions)
        action = actions[index]
        obs, r, done, _ = env.step(action)
        cur_posture = Posture(obs)
        print('\n count=%d,action=%d,%d' % (count,action[0],action[1]))
        cur_posture.print()
        still_open = env.render("human")
        count += 1
