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
    def __init__(self,obs):
        self.pos = obs[:3]
        self.speed = obs[6:9]

        self.front_left_foot = Properties()
        self.front_left_foot.name = 'front_left'
        self.front_left_foot['up'] = Properties()
        self.front_left_foot.up.pos = obs[8]
        self.front_left_foot.up.angle = obs[9]
        self.front_left_foot['down'] = Properties()
        self.front_left_foot.down.pos = obs[10]
        self.front_left_foot.down.angle = obs[11]
        self.front_left_foot.ground = obs[24]

        self.front_right_foot = Properties()
        self.front_right_foot.name = 'front_right'
        self.front_right_foot['up'] = Properties()
        self.front_right_foot.up.pos = obs[12]
        self.front_right_foot.up.angle = obs[13]
        self.front_right_foot['down'] = Properties()
        self.front_right_foot.down.pos = obs[14]
        self.front_right_foot.down.angle = obs[15]
        self.front_left_foot.ground = obs[25]

        self.left_back_foot = Properties()
        self.left_back_foot.name = 'back_left'
        self.left_back_foot['up'] = Properties()
        self.left_back_foot.up.pos = obs[16]
        self.left_back_foot.up.angle = obs[17]
        self.left_back_foot['down'] = Properties()
        self.left_back_foot.down.pos = obs[18]
        self.left_back_foot.down.angle = obs[19]
        self.left_back_foot.ground = obs[26]

        self.right_back_foot = Properties()
        self.right_back_foot.name = 'back_right'
        self.right_back_foot['up'] = Properties()
        self.right_back_foot.up.pos = obs[20]
        self.right_back_foot.up.angle = obs[21]
        self.right_back_foot['down'] = Properties()
        self.right_back_foot.down.pos = obs[22]
        self.right_back_foot.down.angle = obs[23]
        self.right_back_foot.ground = obs[27]

        self.foots = [self.front_left_foot,self.front_right_foot,self.left_back_foot,self.right_back_foot]
    def groudfoots(self):
        return [foot.name for foot in self.foots if foot.groud != 0]
    def groudcode(self):
        return [0 if foot.ground==0 else 1 for foot in self.foots]
    def __str__(self):
        return '位置=%.3f,%.3f,%.3f;速度=%.3f,%.3f,%.3f;着地脚=%s' % \
               (self.pos[0],self.pos[1],self.pos[2],
                self.speed[0],self.speed[1],self.speed[2],str(self.groudfoots()))



def advancement(ind,session):
    '''
    个体行为先进度，根据四足的姿态信息确定适应度
    （1）四足机器人的输入输出
    四足机器人的输入维度是8，分别是四个腿上下两个关节的输入扭矩
    注：这四个关节的顺序总是'front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot‘
    四足机器人的输出维度是24，分别是四足机器人的姿态数据，包括
    1-3：四足机器人相对于起点的坐标x,y,z
    4-5:四足机器人相对于目标的x轴和y轴距离
    6-8：四足机器人在x，y，z三个方向的速度
    9-24：四足机器人每个关节的位置和角度度，其中奇数下标记录角度度，偶数下标记录位置（位置的含义目前不清楚...）
    25-28：四个足是否碰到地面，0是没有碰到
    你可能会奇怪为什么没有朝向，因为这个四足机器人是完全对称的
    （2）控制网络的输入输出
    控制网络的输入是 24 + 1 + 3 + 3 + 3
    分别是四足机器人姿态（24），目标类型（1），目标位置（3），目标移动速度（3），目标移动方向(3)
    目标类型：0为无效目标，1为猎物；2为原地转向（此时目标位置为转向）
    控制网络的输出是8，分别是按照下面顺序的四个腿的上关节和下关节的扭矩：'

    对机器人的行为进行行为先进度测试，测试的结果为两个值，第一个值是行为先进度等级：第二个值是该等级的行为能力值
    1 关节能动： 行为能力值为同时能动的关节数*（关节变化位置幅度+关节变化角度），归一化到0-1之间
    2 腿能离地： 行为能力值为腿能规则离地的次数
    3 能移动位置：行为能力值总为1
    4 能走直线：  行为能力值为直线的均方误差的倒数
    5 能转弯：行为能力值为
    6 能在行走中转弯
    7 能在行走中改变速度
    0 是否能动腿：有腿移动，控制关节位置在特定范围内变化，控制多个关节位置在特定范围组合内变化，控制多个关节按照一定时序在特定范围内变化
    1.是否走直线：直线
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
    init_obs = env.reset()

class ProgressivenesMap:
    pass

class Progressivenes:
    def __init__(self):
        self.name = ''
        self.epoch = 100
        self.command = [0,0,0,0]
    def execute(self,net,session):
        if isinstance(net, Individual):
            net = net.getPhenome()

        pybullet.connect(pybullet.DIRECT)
        env = gym.make("AntBulletEnv-v0")
        if rendering:
            env.render(mode="human")
        init_obs = env.reset()
        _obs = init_obs

        count = 0
        postures = []
        while count < 100:
            input = _obs + self.command
            action = net.activate(input)
            _obs, r, done, _ = env.step(action)
            if rendering:
                still_open = env.render("human")
                if still_open == False:break
            if done: break
            count += 1
            postures.append(Posture(_obs))
        if len(postures) <= 0:return 0.0
        return self.do_test(postures,net,session,env)
    def do_test(self,postures,net,session,env):pass

class ProgressivenesLeg(Progressivenes):
    def __init__(self):
        super.__init__(Progressivenes,self)
        self.name = '腿部灵活性测试'

    def do_test(self, postures, net, session, env):
        '''
        有腿着地发生变化算1分，三只腿着地算1分，且一条不着地腿规则轮换算4分
        所有分数相加除以姿态变化数
        :param postures:
        :param net:
        :param session:
        :param env:
        :return:
        '''
        threshold = 0.01
        if len(postures) <= 1: return 0.0
        score = 0.
        for i in range(len(postures)-1):
            # 测试腿是否能动作
            if postures[i].front_left_foot.ground != postures[i+1].front_left_foot.ground or \
               postures[i].front_right_foot.ground != postures[i+1].front_right_foot.ground or \
               postures[i].left_back_foot.ground != postures[i+1].left_back_foot.ground or \
               postures[i].right_back_foot.ground != postures[i+1].right_back_foot.ground:
                score += 1
        score = score / len(postures)-1  # 这个score最大是1



class ProgressivenesMove(Progressivenes):
    def __init__(self):
        super.__init__(Progressivenes,self)
        self.name = '移动能力测试'

    def do_test(self, postures, net, session, env):
        '''
        直线，指定距离的直线，指定距离和速度的直线
        :param postures :
        :param net:
        :param session:
        :param env:
        :return:
        '''
        x_data = np.array([p.pos[0] for p in postures])
        y_data = np.array([p.pos[1] for p in postures])
        # 做线性回归，看平均误差，确定走直线的程度
        ## 网络训练结构
        x = fluid.layers.data(name='x', shape=[1], dtype='float32')
        hidden = fluid.layers.fc(input=x, size=2, act='relu')
        net = fluid.layers.fc(input=hidden, size=1, act=None)

        ## 损失函数
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        cost = fluid.layers.square_error_cost(input=net, label=y)
        avg_cost = fluid.layers.mean(cost)
        ## 优化器
        optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01)
        opts = optimizer.minimize(avg_cost)
        ## 训练
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        train_cost = None
        for pass_id in range(10):
            train_cost = exe.run(program=fluid.default_main_program(),
                                 feed={'x': x_data, 'y': y_data},
                                 fetch_list=[avg_cost])
        score = 0
        if train_cost[0]<10:score = 1
        elif train_cost[0] < 1:score = 2







