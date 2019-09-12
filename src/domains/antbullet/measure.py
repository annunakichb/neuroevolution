# -*- coding: UTF-8 -*-

import pybullet
import gym
from domains.antbullet.ant import Posture
import numpy as np
import sklearn.cluster as skc  # 密度聚类
from sklearn import metrics   # 评估模型
import domains.antbullet.env as env

# 猎物位置,移动速度和方向
obj_pos = [80,80,0]
obj_speed = [0,0,0]
obj_direction = [0,0,0]

rendering = False

novelty_archive = []
novelty_k = 10
def novelty(ind,session):
    '''
    noelty测量
    :param ind:
    :param session:
    :return:
    '''
    # 计算与novelty_archive的欧几里得距离
    dis = np.array([np.sqrt(np.sum(np.square(ind['lastpos'] - n))) for n in novelty_archive])
    indexs = list(reversed(np.argsort()))[:novelty_k]
    dis = dis[indexs]
    novelty_archive.append(ind['lastpos'])
    return np.mean(dis)


def evolvabilty(ind,session):
    fitnesses = np.array([ind['fitness']])
    complexes = np.array([env.env_complex[ind.birth]])
    t = ind.parent
    while t is not None:
        fitnesses.append(t['fitness'])
        complexes.append(env.env_complex[t.birth])
    return 2 * np.mean(fitnesses) * np.mean(complexes) / (np.mean(fitnesses) + np.mean(complexes))


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
    3 能移动位置：行为能力值为移动的总距离
    4 能走直线：  行为能力值为直线的均方误差的倒数
    5 能转弯：行为能力值为转弯角度和速度的多样性
    6 能规避障碍物：碰到障碍物的次数的倒数
    7.追逐猎物：与猎物的最终距离的倒数
    :param ind:
    :param session:
    :return:
    '''
    # 执行N次
    N = 100
    net = ind.getPhenome()

    pybullet.connect(pybullet.DIRECT)
    env = gym.make("AntBulletEnv-v0")
    env.render(mode="human")
    init_obs = env.reset()
    _obs = init_obs

    count = 0
    postures = []
    while count < N:
        input = _obs + obj_pos + obj_speed + obj_direction
        action = net.activate(input)
        _obs, r, done, _ = env.step(action)
        if rendering:
            still_open = env.render("human")
            if still_open == False: break
        if done: break
        count += 1
        postures.append(Posture(_obs))
    if len(postures) <= 0: return 0.0

    adv = [0.] * 7
    # 先进度测量1
    kont_pos = np.array([p.getKontPos() for p in postures])
    db = skc.DBSCAN(eps=1.5, min_samples=3).fit(kont_pos)  # DBSCAN聚类方法 还有参数，matric = ""距离计算方法
    adv[0] = len(db.labels_)  # 和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声
    if adv[0] <= 1:return adv
    # 先进度测量2

def do_test(self,postures,net,session,env):
    pass