import numpy as np
import tensorflow as tf
import gym
import os

from domains.ne.cartpoles.enviornment.cartpole import SingleCartPoleEnv
import domains.ne.cartpoles.enviornment.runner as runner
from domains.ne.cartpoles.enviornment import force

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 25
learning_rate = 1e-1
gamma = 0.99

xs, ys, drs = [], [], []
reward_sum = 0

total_episodes = 1000# 根据当前的环境状态根据隐藏节点求action为1的概率


class PolicyNet:
    def __init__(self):
        # 输入层
        self.input_dimension = 4
        self.var_input_x = tf.placeholder(tf.float32, [None, self.input_dimension], name="input_x")
        # 隐藏层１(50个神经元+Relu)
        self.H1 = 50
        self.W1 = tf.get_variable("w1", shape=[D, self.H1],
                     initializer=tf.contrib.layers.xavier_initializer())
        self.L1 = tf.nn.relu(tf.matmul(self.var_input_x, self.W1))
        # 输出层
        self.W2 = tf.get_variable("w2", shape=[self.H1, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
        self.L2 = tf.matmul(self.L1, self.W2)

        self.var_output = tf.nn.sigmoid(self.L2)# 根据概率来求损失和梯度

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
        self.batchGrad = [self.W1grad,self.W2grad]
        self.updateGrads = self.adam.apply_gradients(zip(self.batchGrad, self.tvars))

    def discount_reward(r):
        # 根据每个reward:r和gamma来求每次的潜在价值
        discount_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(r.size)):
            running_add = running_add * gamma + r[t]
            discount_r[t] = running_add
        return discount_r

    def run(self):
        # Session执行
        with tf.Session() as sess:
            rendering = False
            init = tf.global_variables_initializer()
            sess.run(init)
            observation = env.reset()
            gradBuff = sess.run(self.tvars)
            for ix, grad in enumerate(gradBuff):
                gradBuff[ix] = grad * 0
            episode_number = 1
            while episode_number <= total_episodes:
                if reward_sum / batch_size > 100 or rendering == True:
                    rendering = True
                    env.render()

                x = np.reshape(self.var_input_x, [1, self.input_dimension])

                tfprob = sess.run(self.var_output , feed_dict={self.var_input_x: x})
                action = 1 if np.random.uniform() < tfprob else 0
                xs.append(x)
                y = 1 - action
                ys.append(y)

                observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward)
        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs, ys, drs = [], [], []
            discount_epr = discount_reward(epr)
            discount_epr -= np.mean(discount_epr)
            discount_epr /= np.std(discount_epr)

            tGrad = sess.run(newGrads, feed_dict={observate:epx,
                                                  input_y:epy,
                                                  advantages: discount_epr})
            for ix, grad in enumerate(tGrad):
                gradBuff[ix] += grad
                if episode_number % batch_size == 0:
                    sess.run(updateGrads, feed_dict={W1grad:gradBuff[0],
                                                 W2grad:gradBuff[1]})
                    for ix, grad in enumerate(gradBuff):
                        gradBuff[ix] = grad * 0
                print('Average reward for episode %d: %f.' % \
                      (episode_number, reward_sum/batch_size))
                if reward_sum/batch_size > 200:
                    print('Task solve in', episode_number, 'episodes!')
                    break

                reward_sum = 0

            observation = env.reset()

env = SingleCartPoleEnv().unwrapped
net = PolicyNet()


def _do_learn(observation,action,reward,observation_,step,totalreward,total_step):
    RL.store_transition(observation, action, reward, observation_)
    if total_step > 10:
        RL.learn()

def run():
    complexes = []
    reward_list = []
    notdone_count_list = []
    steps = []

    episode_reward_list = []
    episode_notdone_count_list = []
    total_step = 0
    while True:
        # 执行一次
        notdone_count, episode_reward, step, total_step= runner.do_until_done(env,net.choose_action,total_step,_do_learn)
        # 判断是否可以提升复杂度
        if notdone_count > env.max_notdone_count or total_step >= 1500:
            complexes.append(force.force_generator.currentComplex())
            reward_list.append(np.average(episode_reward_list))
            notdone_count_list.append(np.average(episode_notdone_count_list))
            steps.append(total_step)
            np.save('dqn_result.npz', (complexes, notdone_count_list, reward_list,steps))

            episode_notdone_count_list,episode_reward_list = [],[],
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
            if total_step % 100 == 0 and total_step != 0:
                print("持续次数=", episode_notdone_count_list, ",平均=", np.average(episode_notdone_count_list))
                print("累计奖励=", episode_reward_list, ",平均=", np.average(episode_reward_list))
            episode_reward_list.append(episode_reward)
            if len(episode_reward_list) > 10:
                episode_reward_list = episode_reward_list[-10:]
            episode_notdone_count_list.append(notdone_count)
            if len(episode_notdone_count_list) > 10:
                episode_notdone_count_list = episode_notdone_count_list[-10:]

    #np.save('dqn_result.npz', complexes, notdone_count_list,reward_list)
    RL.save()
    plt.plot(complexes, reward_list, label='reward')
    plt.plot(complexes, notdone_count_list, label='times')
    plt.xlabel('complexes')
    plt.savefig('dqn_cartpole.png')

#作者：郎朗坤
#链接：https://www.imooc.com/article/36790
#来源：慕课网