# 修改自https://github.com/ShangtongZhang/DistributedES/blob/master/natural_es.py
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import SimpleQueue
import numpy as np
from utils import *
import pickle
import time


class NaturalEvolutionStrategies:
    def __init__(self,f,init_param,context=None,pop_size=64,sigma=0.1,weight_decay=0.005,learning_rate=0.1,num_works=1,end_max_steps=int(1e7),end_max_fvalue=0.0,*kwargs):
        '''
        自然进化策略
        :param f:          function 最大化的函数，该函数传入待优化参数和上下文，返回浮点数
        :param init_param  list     初始待优化的参数，例如对于神经网络，参数为网络权重和偏置
        :param context     any      任意对象，最大化函数决定
        :param pop_size:   int      种群中个体数量
        :param sigma:      float
        :param weight_decay  float
        :param learning_rate: float 学习率
        :param num_works:     int    多进程的最大进程数量
        :param end_max_steps:  int   最大迭代次数
        :param end_max_fvalue: int   最大函数值
        :param kwargs:
        '''
        self.f = f
        self.init_param = init_param
        self.param = init_param
        self.context = context
        self.pop_size = pop_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_works = num_works
        self.end_max_steps = end_max_steps
        self.end_max_fvalue = end_max_fvalue
        self.opt = Adam()
    def run(self):
        p = mp.Process(target=self.train, args=())
        p.start()
        p.join()
        return self.param

    def train(self):
        task_queue = SimpleQueue()
        result_queue = SimpleQueue()
        stop = mp.Value('i', False)
        param = torch.FloatTensor(torch.from_numpy(self.param))
        param.share_memory_()
        workers = [Worker(id, param, task_queue, result_queue, stop, self) for id in
                   range(self.num_workers)]
        for w in workers: w.start()

        training_rewards = []
        training_steps = []
        training_timestamps = []
        initial_time = time.time()
        total_steps = 0
        iteration = 0
        while not stop.value:
            for i in range(self.pop_size):
                task_queue.put(i)
            rewards = []
            epsilons = []
            steps = []
            while len(rewards) < self.pop_size:
                if result_queue.empty():
                    continue
                epsilon, fitness, step = result_queue.get()
                epsilons.append(epsilon)
                rewards.append(fitness)
                steps.append(step)

            total_steps += np.sum(steps)
            r_mean = np.mean(rewards)
            r_std = np.std(rewards)
            # rewards = (rewards - r_mean) / r_std
            print('Train: iteration %d, %f(%f)' % (iteration, r_mean, r_std / np.sqrt(self.pop_size)))
            iteration += 1
            if r_mean > self.end_max_fvalue:
                stop.value = True
                break
            if self.end_max_steps and total_steps > self.end_max_steps:
                stop.value = True
                break

            rewards = self.fitness_shift(rewards)
            gradient = np.asarray(epsilons) * np.asarray(rewards).reshape((-1, 1))
            gradient = np.mean(gradient, 0) / self.sigma
            gradient -= self.weight_decay * gradient
            gradient = self.opt.update(gradient)
            gradient = torch.FloatTensor(gradient)
            param.add_(self.learning_rate * gradient)

        for w in workers: w.join()
        return [training_rewards, training_steps, training_timestamps]

    def fitness_shift(x):
        x = np.asarray(x).flatten()
        ranks = np.empty(len(x))
        ranks[x.argsort()] = np.arange(len(x))
        ranks /= (len(x) - 1)
        ranks -= .5
        return ranks


class Worker(mp.Process):
    def __init__(self, id, param, task_q, result_q, stop, es):
        mp.Process.__init__(self)
        self.id = id
        self.task_q = task_q
        self.param = param
        self.result_q = result_q
        self.stop = stop
        self.es = es

    def run(self):
        config = self.config
        np.random.seed()
        while not self.stop.value:
            if self.task_q.empty():
                continue
            self.task_q.get()
            disturbed_param = np.copy(self.param.numpy().flatten())
            epsilon = np.random.randn(len(disturbed_param))
            disturbed_param += config.sigma * epsilon
            #fitness, steps = self.evaluator.eval(disturbed_param)
            fitness, steps = self.es.f(disturbed_param,self.es.context)
            self.result_q.put([epsilon, -fitness, steps])

class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_t = self.beta2_t = 1
        self.epsilon = epsilon
        self.m = 0
        self.v = 0

    def update(self, g):
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(g, 2)
        m_ = self.m / (1 - self.beta1_t)
        v_ = self.v / (1 - self.beta2_t)
        return m_ / (np.sqrt(v_) + self.epsilon)