import tensorflow as tf
import numpy as np

import os
import csv
import copy
from domains.ne.cartpoles.enviornment.cartpole import SingleCartPoleEnv
from domains.ne.cartpoles.enviornment import force

# 引用自：https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Policy%20Gradients/Cartpole/Cartpole%20REINFORCE%20Monte%20Carlo%20Policy%20Gradients.ipynb

env = SingleCartPoleEnv().unwrapped

## ENVIRONMENT Hyperparameters
state_size = 4
action_size = env.action_space.n

## TRAINING Hyperparameters
learning_rate = 0.01
gamma = 0.95 # Discount rate


mode = 'noreset'
maxepochcount = 1000
complexunit = 20.

class PolicyGradients:
    def __init__(self):
        with tf.name_scope("inputs"):
            self.input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
            self.actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
            self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards")

            # Add this placeholder for having this variable in tensorboard
            self.mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")

            with tf.name_scope("fc1"):
                fc1 = tf.contrib.layers.fully_connected(inputs=self.input_,
                                                        num_outputs=10,
                                                        activation_fn=tf.nn.relu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("fc2"):
                fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                                        num_outputs=action_size,
                                                        activation_fn=tf.nn.relu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("fc3"):
                fc3 = tf.contrib.layers.fully_connected(inputs=fc2,
                                                        num_outputs=action_size,
                                                        activation_fn=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("softmax"):
                self.action_distribution = tf.nn.softmax(fc3)

            with tf.name_scope("loss"):
                # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
                # If you have single-class labels, where an object can only belong to one class, you might now consider using
                # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=self.actions)
                self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_)

            with tf.name_scope("train"):
                self.train_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def _dounitdown(self,sess):
        '''
        执行直到杆子倒下
        :param sess:
        :return:
        '''
        state = env.reset()
        episode_states = []
        episode_actions = []
        not_downcount = 0

        while True:

            # 执行一次
            action_probability_distribution = sess.run(self.action_distribution,
                                                       feed_dict={self.input_: state.reshape([1, 4])})

            action = np.random.choice(range(action_probability_distribution.shape[1]),
                                      p=action_probability_distribution.ravel())  # select action w.r.t the actions prob

            # Perform a
            new_state, reward, done, info = env.step(action)

            # Store s, a, r
            episode_states.append(state)

            # For actions because we output only one (the index) we need 2 (1 is for the action taken)
            # We need [0., 1.] (if we take right) not just the index
            action_ = np.zeros(action_size)
            action_[action] = 1

            episode_actions.append(action_)
            #episode_rewards.append(reward)

            if not done:
                not_downcount += 1
                state = new_state
                continue

            return not_downcount,episode_states,episode_actions

    def discount_and_normalize_rewards(self,episode_rewards):
        '''
        计算伽马折扣奖励
        :return:
        '''
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * gamma + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative

        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

        return discounted_episode_rewards

    def run(self,**kwargs):
        global mode
        global maxepochcount
        global complexunit
        global env

        mode = 'noreset' if 'mode' not in kwargs.keys() else kwargs['mode']
        maxepochcount = 1000 if 'maxepochcount' not in kwargs.keys() else int(kwargs['maxepochcount'])
        complexunit = 20.0 if 'complexunit' not in kwargs.keys() else float(kwargs['complexunit'])
        xh = None if 'xh' not in kwargs.keys() else int(kwargs['xh'])

        complexes_list = []
        notdone_count_list = []
        episode_number = 0
        saver = tf.train.Saver()

        if xh is None or str(int(xh)) == '':
            xh = ''
        else:
            xh = "_" + str(int(xh))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        while 1:
            not_downcount,episode_states,episode_actions = self._dounitdown(sess)
            episode_rewards = [1. for item in range(not_downcount+1)]
            discounted_episode_rewards = self.discount_and_normalize_rewards(episode_rewards)

            # Feedforward, gradient and backpropagation
            loss_, _ = sess.run([self.loss, self.train_opt],
                                feed_dict={self.input_: np.vstack(np.array(episode_states)),
                                            self.actions: np.vstack(np.array(episode_actions)),
                                            self.discounted_episode_rewards_: discounted_episode_rewards
                                            })
            notdone_count_list.append(not_downcount)
            episode_number += 1

            if not_downcount > env.max_notdone_count or episode_number >= maxepochcount:
                complexes_list.append(force.force_generator.currentComplex())

                print([(f, c) for f, c in zip(complexes_list, notdone_count_list)])

                # 记录过程记录
                filename = os.path.split(os.path.realpath(__file__))[0] + os.sep + 'datas_' + mode + os.sep + \
                           'policy' + os.sep + 'policy' + str(xh) + '.csv'
                out = open(filename, 'a', newline='')
                csv_write = csv.writer(out, dialect='excel')
                csv_write.writerow([complexes_list[-1]] + notdone_count_list)
                notdone_count_list = []

                # 升级复杂度,为了加快执行速度,让复杂度增加幅度至少大于min_up
                changed, newcomplex, k, w, f, sigma = force.force_generator.promptComplex(complexunit)
                if not changed or newcomplex is None or newcomplex == complexes_list[-1]:
                    sess.close()
                    return False
                    break  # 复杂度已经达到最大,结束
                print('新的环境复杂度=%.3f,k=%.2f,w=%.2f,f=%.2f,sigma=%.2f' % (newcomplex, k, w, f, sigma))

                if mode == 'reset':
                    sess.close()
                    return True

                episode_number = 0

        sess.close()

if __name__ == '__main__':
    force.init()

    for i in range(10):
        kwargs = {'mode': 'noreset', 'xh':i,'maxepochcount' : 500,'complexunit':20.0}
        env = SingleCartPoleEnv().unwrapped
        net = PolicyGradients()
        while 1:
            result = net.run(**kwargs)
            if kwargs['mode'] == 'noreset':
                break
            if not result:
                break
            env = SingleCartPoleEnv().unwrapped
            net = PolicyGradients()
        force.force_generator = force.ForceGenerator(0.0, 0.0, 0.0, 1.01)