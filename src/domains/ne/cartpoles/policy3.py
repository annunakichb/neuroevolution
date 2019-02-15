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
max_episodes = 300
learning_rate = 0.01
gamma = 0.95 # Discount rate


def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards

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

    def run(self,xh=None,mode='noreset'):
        allRewards = []
        total_rewards = 0
        maximumRewardRecorded = 0
        episode = 0
        episode_states, episode_actions, episode_rewards = [], [], []

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for episode in range(max_episodes):

                episode_rewards_sum = 0

                # Launch the game
                state = env.reset()

                #env.render()

                while True:

                    # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
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

                    episode_rewards.append(reward)
                    if done:

                        # Calculate sum reward
                        '''
                        episode_rewards_sum = np.sum(episode_rewards)

                        allRewards.append(episode_rewards_sum)

                        total_rewards = np.sum(allRewards)

                        # Mean reward
                        mean_reward = np.divide(total_rewards, episode + 1)

                        maximumRewardRecorded = np.amax(allRewards)

                        print("==========================================")
                        print("Episode: ", episode)
                        print("Reward: ", episode_rewards_sum)
                        print("Mean Reward", mean_reward)
                        print("Max reward so far: ", maximumRewardRecorded)
                        '''
                        # Calculate discounted reward
                        discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)

                        # Feedforward, gradient and backpropagation
                        loss_, _ = sess.run([self.loss, self.train_opt], feed_dict={self.input_: np.vstack(np.array(episode_states)),
                                                                          self.actions: np.vstack(np.array(episode_actions)),
                                                                          self.discounted_episode_rewards_: discounted_episode_rewards
                                                                          })


                        # Write TF Summaries
                        '''
                        summary = sess.run(write_op, feed_dict={self.input_: np.vstack(np.array(episode_states)),
                                                                self.actions: np.vstack(np.array(episode_actions)),
                                                                self.discounted_episode_rewards_: discounted_episode_rewards,
                                                                self.mean_reward_: mean_reward
                                                                })

                        writer.add_summary(summary, episode)
                        writer.flush()
                        '''

                        # Reset the transition stores
                        episode_states, episode_actions, episode_rewards = [], [], []

                        break

                    state = new_state

                # Save Model
                if episode % 100 == 0:
                    saver.save(sess, "./models/model.ckpt")
                    print("Model saved")


if __name__ == '__main__':
    net = PolicyGradients()
    net.run()