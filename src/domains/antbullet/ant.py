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
class Foot:
    '''足基本信息'''
    front_left = 0
    back_left = 1
    back_right = 2
    front_right = 3
    names = ['front_left','back_left','back_right','front_right']
class Knot:
    '''关节基本信息'''
    front_left_up = 0
    front_left_down = 1
    back_left_up = 2
    back_left_down = 3
    back_right_up = 4
    back_right_down = 5
    front_right_up = 6
    front_right_down = 7
    names = ['front_left_up','front_left_down',
             'back_left_up','back_left_down',
             'back_right_up','back_right_down',
             'front_right_up','front_right_down']


class Action:
    def __init__(self,actions):
        '''
        actions: list 动作序列
        机器人动作，调用env.step时需传入长度为8的list，list的每维表示一个关节的扭矩，其顺序为：
        front_left_up,front_left_down,back_left_up,back_left_down,back_right_up,back_right_down,front_right_up,front_right_down
        注：在可视化显示中左前关节下蓝上黄，左后关节下绿上红，右后关节下黄上蓝，左前关节下红上绿
        施加扭矩的正负决定其运动方向，其中所有的上关节是前后运动（相对于身体），下关节是内外运动（朝自身身体方向收缩运动和伸展运动），其每个关节的正负意义不同，如下：
        左前上关节：  + 向前       -向后
        左前下关节：  + 向内收缩    -向外伸展
        左后上关节：  + 向前       -向后
        左后下关节：  - 向内收缩    +向外伸展
        右后上关节：  + 向前       -向后
        右后下关节：  - 向内收缩    +向外伸展
        右前上关节：  + 向前       -向后
        右前下关节：  + 向内收缩    -向外伸展
        '''
        self.actions = actions

    @property
    def value(self,knot):
        '''
        取得某个关节的扭矩值
        :param Union(int,str) 关节的编号或者名称
        '''
        if isinstance(knot,str):
            knot = Knot.names.index(knot)
        return self.actions[knot]

    @value.setter
    def setValue(self,knot,value):
        '''
        修改某个关节的扭矩
        :param knot:  Union(int,str) 关节索引或名称
        :param value: float 扭矩值
        :return: None
        '''
        if isinstance(knot, str):
            knot = Knot.names.index(knot)
        self.actions[knot] = value


class Posture:
    '''机器人姿态数据'''
    FOOTS= ['front_left', 'front_right', 'left_back', 'right_back']
    KNOTS = ['front_left_up','front_left_down','front_right_up','front_right_down','left_back_up','left_back_down','right_back_up','right_back_down']
    def __init__(self,obs,robot):
        '''
        :param obs: 四足机器人的观测数据，
            四足机器人的输入维度是8，分别是四个腿上下两个关节的输入扭矩
                注：这四个关节的顺序总是'front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot‘
            四足机器人的输出维度是28，是四足机器人的姿态数据，包括
                # 0：z轴方向距离初始z的距离，
                # 1-2：面向行动目标的角度的sin值和cos值（啥是‘面向’，暂时我也不知道）
                # 3-5：相对于身体视点的三个方向的运动速度
                # 6-7：身体的body_rpy中的r和p
                # 8-23：8个关节，按照顺序的位置和速度，归一化到-1和+1之间（位置怎么理解？不知道）
                # 24-27:4个脚是否着地，1着地，0未着地
        '''
        self.pos = robot.body_xyz
        self.parts_xyz = np.array([p.pose().xyz() for p in robot.parts.values()]).flatten()

        body_pose = robot.robot_body.pose()
        self.body_rpy = body_pose.rpy() # r, p, yaw

        #self.orientation = robot.current_orientation()

        self.speed = obs[3:6]

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
        print('左前脚落地=', self.front_left_foot.ground)
        print('左前脚上关节: pos=%.5f,angle=%.3f' % (self.front_left_foot.up.pos, self.front_left_foot.up.angle))
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










