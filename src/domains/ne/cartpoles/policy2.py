#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import gym
import os
import csv
import copy
from domains.ne.cartpoles.enviornment.cartpole import SingleCartPoleEnv
import domains.ne.cartpoles.enviornment.runner as runner
from domains.ne.cartpoles.enviornment import force

batch_size = 2
learning_rate = 1e-1

class PolicyNet:
    def __init__(self):
        # 输入层
        self.input_dimension = 4
        self.var_input_x = tf.placeholder(tf.float32, [None, self.input_dimension], name="input_x")
        # 隐藏层１(50个神经元+Relu)
        self.H1 = 50
        self.W1 = tf.get_variable("w1", shape=[self.input_dimension, self.H1],
                     initializer=tf.contrib.layers.xavier_initializer())
        self.L1 = tf.nn.relu(tf.matmul(self.var_input_x, self.W1))
        # 隐藏层2(50个神经元+Relu)
        self.H2 = 50
        self.W2 = tf.get_variable("w2", shape=[self.H1, self.H2],
                     initializer=tf.contrib.layers.xavier_initializer())
        self.L2 = tf.nn.relu(tf.matmul(self.L1, self.W2))
        # 隐藏层2(50个神经元+Relu)
        self.H3 = 50
        self.W3 = tf.get_variable("w3", shape=[self.H2, self.H3],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.L3 = tf.nn.relu(tf.matmul(self.L2, self.W3))

        # 输出层
        self.W4 = tf.get_variable("w4", shape=[self.H3, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
        self.L4 = tf.matmul(self.L3, self.W4)
        self.var_output = tf.nn.sigmoid(self.L4)# 根据概率来求损失和梯度
        self.var_input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.var_reward  = tf.placeholder(tf.float32, name="reward")

        self.loglik = tf.log(self.var_input_y * (self.var_input_y - self.var_output) +
                (1 - self.var_input_y) * (self.var_input_y + self.var_output))
        self.loss = -tf.reduce_mean(self.loglik * self.var_reward )

        self.tvars = tf.trainable_variables()
        self.newGrads = tf.gradients(self.loss, self.tvars)# 根据梯度优化训练两层神经网络
        self.learning_rate = 1e-1
        self.adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.W1grad = tf.placeholder(tf.float32, name="batch_grad1")
        self.W2grad = tf.placeholder(tf.float32, name="batch_grad2")
        self.W3grad = tf.placeholder(tf.float32, name="batch_grad3")
        self.W4grad = tf.placeholder(tf.float32, name="batch_grad4")
        self.batchGrad = [self.W1grad,self.W2grad,self.W3grad,self.W4grad]
        self.updateGrads = self.adam.apply_gradients(zip(self.batchGrad, self.tvars))

    def getCorrectAction(self,observation,action):
        theta = observation[0][2]
        return 0 if theta > 0 else 1
        #return 0 if action >= 1 else 1
        '''
        if observation[0][2] > 0 and observation[0][3] > 0: # 向右加速倒下
            if observation[0][1] > 0: # 小车正在右移
                return action
            else:
                return 0
        elif observation[0][2] > 0 and observation[0][3] < 0: #向右减速倒下
            if observation[0][1] > 0:
                return action
            else:
                return 0
        elif observation[0][2] < 0 and observation[0][3] < 0:
            if observation[0][1] > 0:  # 小车正在右移
                return 1
            else:
                return action
        elif observation[0][2] < 0 and observation[0][3] > 0:
            if observation[0][1] > 0:
                return 1
            else:
                return action

        return 0 if action >= 1 else 1
        '''
        #theta = observation[0][2]
        #return 0 if theta > 0 else 1

env = SingleCartPoleEnv().unwrapped
net = PolicyNet()

mode = 'noreset'
maxepochcount = 1000
complexunit = 20.

def _dountilpoledown(session):
    # 反复执行直到杆子倒下
    notdone_count = 0
    input_x_records = []
    input_y_records = []
    reward_records = []
    step = 0
    observation = env.reset()
    while True:
        # 执行一次
        #ob = copy.deepcopy(observation)
        # ob = np.append(observation,np.array([env.wind]))
        # x = np.reshape(ob, [1, net.input_dimension])
        x = np.reshape(observation, [1, net.input_dimension])
        tfprob = session.run(net.var_output, feed_dict={net.var_input_x: x})
        action = 1 if np.random.uniform() < tfprob else 0
        input_x_records.append(x)
        input_y_records.append(net.getCorrectAction(x,action))
        env.wind = force.force_generator.next(step * env.tau)
        observation, reward, done, info = env.step(action)

        if not done:
            notdone_count += 1
            if notdone_count >= env.max_notdone_count:
                break
            step += 1
            continue
        break
    reward_records = [1.0 / len(input_x_records) for t in input_x_records]
    return notdone_count,input_x_records,input_y_records,reward_records,step



def run(**kwargs):
    global mode
    global maxepochcount
    global complexunit
    global env
    global net

    #mode = 'noreset' if 'mode' not in kwargs.keys() else kwargs['mode']
    #maxepochcount = 1000 if 'maxepochcount' not in kwargs.keys() else int(kwargs['maxepochcount'])
    #complexunit = 20.0 if 'complexunit' not in kwargs.keys() else float(kwargs['complexunit'])
    #xh = None if 'xh' not in kwargs.keys() else int(kwargs['xh'])
    mode = 'noreset' if 'mode' not in kwargs else kwargs['mode']
    maxepochcount = 1000 if 'maxepochcount' not in kwargs else int(kwargs['maxepochcount'])
    complexunit = 20.0 if 'complexunit' not in kwargs else float(kwargs['complexunit'])
    xh = None if 'xh' not in kwargs else int(kwargs['xh'])

    execute(xh,mode)

def execute(xh=None,mode='noreset'):
    global env
    global  net
    complexes = []
    notdone_count_list = []

    # 初始化
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)
    observation = env.reset()
    gradBuff = session.run(net.tvars)
    for ix, grad in enumerate(gradBuff):
        gradBuff[ix] = grad * 0

    episode_number = 0
    episode_notdone_count_list = []

    if xh is None or str(int(xh)) == '':
        xh = ''
    else:
        xh = "_" + str(int(xh))
    while True:
        # 执行一次
        notdone_count,input_x_records,input_y_records,reward_records,step = _dountilpoledown(session)
        episode_notdone_count_list.append(notdone_count)
        # 计算新梯度
        tGrad = session.run(net.newGrads, feed_dict={net.var_input_x: np.vstack(input_x_records),
                                                  net.var_input_y: np.vstack(input_y_records),
                                                  net.var_reward: np.vstack(reward_records)})
        for ix, grad in enumerate(tGrad):
            gradBuff[ix] += grad
        # 更新梯度
        episode_number += 1
        if episode_number % batch_size == 0:
            session.run(net.updateGrads, feed_dict={net.W1grad: gradBuff[0],net.W2grad: gradBuff[1],
                                                    net.W3grad: gradBuff[2],net.W4grad: gradBuff[3]})
            for ix, grad in enumerate(gradBuff):
                gradBuff[ix] = grad * 0

        #if episode_number % 100 == 0 and episode_number != 0:
        #    print("持续次数=", episode_notdone_count_list, ",平均=", np.average(episode_notdone_count_list))
        if notdone_count > env.max_notdone_count or episode_number >= maxepochcount:
            # 记录复杂度和对应获得的奖励
            complexes.append(force.force_generator.currentComplex())
            notdone_count_list.append(np.max(episode_notdone_count_list))
            #filename = os.path.split(os.path.realpath(__file__))[0] + '\\datas\\policy' + str(xh) + '.npy'
            #np.save(filename, (complexes, notdone_count_list))
            print([(f, c) for f, c in zip(complexes, notdone_count_list)])

            # 记录过程记录
            filename = os.path.split(os.path.realpath(__file__))[0] + os.sep + 'datas_' + mode + os.sep + \
                       'policy' + os.sep + 'policy' + str(xh) + '.csv'
            out = open(filename, 'a', newline='')
            csv_write = csv.writer(out, dialect='excel')
            csv_write.writerow([complexes[-1]]+episode_notdone_count_list)
            episode_notdone_count_list = []

            # 升级复杂度,为了加快执行速度,让复杂度增加幅度至少大于min_up
            changed, newcomplex, k, w, f, sigma = force.force_generator.promptComplex(complexunit)
            if not changed or newcomplex is None:
                break  # 复杂度已经达到最大,结束
            print('新的环境复杂度=%.3f,k=%.2f,w=%.2f,f=%.2f,sigma=%.2f' % (newcomplex, k, w, f, sigma))
            #session.close()
            #session = tf.Session()
            #init = tf.global_variables_initializer()
            #session.run(init)
            #observation = env.reset()
            #gradBuff = session.run(net.tvars)
            #for ix, grad in enumerate(gradBuff):
            #    gradBuff[ix] = grad * 0

            episode_number = 0
            #episode_notdone_count_list = []
            if mode == 'reset':
                env = SingleCartPoleEnv().unwrapped
                net = PolicyNet()

    tf.reset_default_graph()
    session.close()


if __name__ == '__main__':
    force.init()

    for i in range(10):
        run(mode='noreset', maxepochcount=2000, complexunit=20.,xh=i)
        env = SingleCartPoleEnv().unwrapped
        net = PolicyNet()
        force.force_generator = force.ForceGenerator(0.0, 0.0, 0.0, 1.01)

