import sys
import inspect
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils.collections as collections

# 自然进化策略：修改自https://github.com/callaunchpad/NES/blob/master/algorithm.py
class NESConfig():
    def __init__(self,learning_rate=0.2,noise_std_dev=0.4,n_populations=100,n_individuals=5,max_rewards=0.8,save_directory=''):
        self.learning_rate = learning_rate
        self.noise_std_dev = noise_std_dev
        self.n_populations = n_populations
        self.n_individuals = n_individuals
        self.max_rewards = max_rewards
        self.save_directory = save_directory

class NES():
    def __init__(self,config=None):
        self.config = config
        if self.config is None:
            self.config = NESConfig()

    def run(self,reward_func,init_params,context,plot=False,logging=False):
        n_reached_target = []
        population_rewards = []
        master_params = init_params
        max_reward = 0.
        for p in range(self.config.n_populations):
            noise_samples = np.random.randn(self.config.n_individuals,len(master_params))
            rewards = np.zeros(self.config.n_individuals)
            n_individual_target_reached = 0
            for i in range(self.config.n_individuals):
                sample_params = [p+n  for p,n in zip(master_params,noise_samples[i])]
                rewards[i] = reward_func(sample_params,context)
                n_individual_target_reached += rewards[i] == 1
            if max(rewards)>max_reward:
                max_reward = max(rewards)
                print('NES evolution->max_reward=',max_reward,'context=',str(context))
            master_params = self.update(master_params,noise_samples,rewards)
            n_reached_target.append(n_individual_target_reached)
            population_rewards.append(sum(rewards)/len(rewards))
            if plot:
                self.plot_graphs([range(p+1), range(p+1)], [population_rewards, n_reached_target], ["Average Reward per population", "Number of times target reached per Population"], ["reward.png", "success.png"], ["line", "scatter"])

            if collections.any(rewards,lambda r:r>=self.config.max_rewards):
                return master_params

        return master_params

    def update(self, master_params, noise_samples, rewards):
        '''参数每次list(迭代更新实现在这里，没有完全看懂，好像跟论文中说的不一样，先照用'''
        normalized_rewards = (rewards - np.mean(rewards))
        if np.std(rewards) != 0.0:
            normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        p = self.config.learning_rate / (self.config.n_individuals * self.config.noise_std_dev)
        delta = p * np.dot(noise_samples.T,normalized_rewards)
        master_params = [p+d for p,d in zip(master_params,delta)]
        #master_params += p * np.dot(noise_samples.T,normalized_rewards)

        return master_params

    def plot_graphs(self,x_axes,y_axes,titles,filenames,types):
        for i in range(len(x_axes)):
            plt.title(titles[i])
            if types[i] == 'lines':plt.plot(x_axes[i],y_axes[i])
            if types[i] == 'scatter':
                plt.scatter(x_axes[i],y_axes[i])
                plt.plot(np.unique(x_axes[i]),np.poly1d(np.polyfit(x_axes[i], y_axes[i], 1))(np.unique(x_axes[i])))
            if self.config.save_directory is not None:
                plt.savefig(self.config.save_directory + filenames[i])
            plt.clf()
