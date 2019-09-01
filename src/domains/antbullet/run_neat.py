import pybullet
import gym
import time
import numpy as np
from functools import reduce

def fitness(ind,session):
    '''
    对以下进行打分：
    1.是否走直线：直线，指定距离的直线，指定距离和速度的直线
    2.转动：转动，指定顺（逆）时针转动，指定转动角度，指定转动角度和速度
    3.行走中转弯：行走转弯，行走按照特定方向转弯，同时转动特定角度，转动并走到特定位置，在限定的时间内
    4.走到指定位置：走到指定位置，直线走到指定位置，沿特定路线走到指定位置，沿特定路线给定速度走到指定位置
    5.变速：接收加减速指令，按照指定加速度行走，先加速再减速
    6.规避障碍物：规避单个静止障碍物，规避多个静止障碍物，规避一个移动障碍物，规避多个移动障碍物
    7.追逐猎物
    :param ind:
    :param session:
    :return:
    '''
    pybullet.connect(pybullet.DIRECT)
    env = gym.make("AntBulletEnv-v0")
    env.render(mode="human")
    env.reset()

class FitnessMap:
    def __init__(self,centers,sigmas,weights=[]):
        self.centers = centers
        self.sigmas = sigmas
        if weights is None or len(weights) <= 0:
            weights = [1/len(self.centers)]*len(self.centers)
        self.weights = weights

    def value(self,pos):
        return np.sum([weight*(np.exp((-1 * np.power(pos - self.center, 2)) / (2 * self.sigma * self.sigma))) for center,sigma,weight in zip(self.centers,self.sigmas,self.weights)])

fitnessMap = FitnessMap()
def fitness(ind,session):
    pass


class NeatPolicy:
    def __init__(self):
        pass
    