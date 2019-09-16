# -*- coding: UTF-8 -*-

import pybullet
import gym
import time
import numpy as np
from functools import reduce
from utils.properties import Properties
from evolution.agent import Individual
import paddle.fluid as fluid
import paddle

rendering = False

class Posture:
    '''机器人姿态数据'''
    def __init__(self,obs):
        self.pos = obs[:3]
        self.speed = obs[6:9]

        self.front_left_foot = Properties()
        self.front_left_foot.name = 'front_left'
        self.front_left_foot.up = Properties()
        self.front_left_foot.up.pos = obs[8]
        self.front_left_foot.up.angle = obs[9]
        self.front_left_foot.down = Properties()
        self.front_left_foot.down.pos = obs[10]
        self.front_left_foot.down.angle = obs[11]
        self.front_left_foot.ground = obs[24]

        self.front_right_foot = Properties()
        self.front_right_foot.name = 'front_right'
        self.front_right_foot.up = Properties()
        self.front_right_foot.up.pos = obs[12]
        self.front_right_foot.up.angle = obs[13]
        self.front_right_foot.down = Properties()
        self.front_right_foot.down.pos = obs[14]
        self.front_right_foot.down.angle = obs[15]
        self.front_left_foot.ground = obs[25]

        self.left_back_foot = Properties()
        self.left_back_foot.name = 'back_left'
        self.left_back_foot.up = Properties()
        self.left_back_foot.up.pos = obs[16]
        self.left_back_foot.up.angle = obs[17]
        self.left_back_foot.down = Properties()
        self.left_back_foot.down.pos = obs[18]
        self.left_back_foot.down.angle = obs[19]
        self.left_back_foot.ground = obs[26]

        self.right_back_foot = Properties()
        self.right_back_foot.name = 'back_right'
        self.right_back_foot.up = Properties()
        self.right_back_foot.up.pos = obs[20]
        self.right_back_foot.up.angle = obs[21]
        self.right_back_foot.down = Properties()
        self.right_back_foot.down.pos = obs[22]
        self.right_back_foot.down.angle = obs[23]
        self.right_back_foot.ground = obs[27]

        self.foots = [self.front_left_foot,self.front_right_foot,self.left_back_foot,self.right_back_foot]
    def groudfoots(self):
        '''
        取得着地脚的名字
        :return:
        '''
        return [foot.name for foot in self.foots if foot.groud != 0]
    def groudcode(self):
        '''
        着地脚的序号
        :return:
        '''
        return [0 if foot.ground==0 else 1 for foot in self.foots]
    def __str__(self):
        return '位置=%.3f,%.3f,%.3f;速度=%.3f,%.3f,%.3f;' % \
               (self.pos[0],self.pos[1],self.pos[2],
                self.speed[0],self.speed[1],self.speed[2])
    def print(self):
        print(str(self))
        print('左前脚上关节: pos=%.5f,angle=%.3f' % (self.front_left_foot.up.pos,self.front_left_foot.up.angle))
        print('左前脚下关节: pos=%.5f,angle=%.3f' % (self.front_left_foot.down.pos, self.front_left_foot.down.angle))

    def getKontPos(self):
        '''
        获取关节姿态
        :return:
        '''
        return [self.front_left_foot.up.pos,self.front_left_foot.up.angle,
                self.front_left_foot.down.pos,self.front_left_foot.down.angle,
                self.front_right_foot.up.pos,self.front_right_foot.up.angle,
                self.front_right_foot.down.pos,self.front_right_foot.down.angle,
                self.left_back_foot.up.pos,self.left_back_foot.up.angle,
                self.left_back_foot.down.pos,self.left_back_foot.down.angle,
                self.right_back_foot.up.pos,self.right_back_foot.up.angle,
                self.right_back_foot.down.pos,self.right_back_foot.down.angle]










