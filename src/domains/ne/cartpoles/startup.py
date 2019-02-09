#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import numpy as np
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.contrib.completers import WordCompleter
import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import  gc
import os

import domains.ne.cartpoles.enviornment.force as force
from domains.ne.cartpoles.enviornment.force import  ForceGenerator

import domains.ne.cartpoles.dqn_cartpole as dqnrunner
import domains.ne.cartpoles.ddqn_cartpole as ddqnrunner
import domains.ne.cartpoles.neat_feedforeward as neatrunner
import domains.ne.cartpoles.hyperneat_feedforeward as hyperneatrunner
import  domains.ne.cartpoles.policy as policyrunner

# The following is a complete experiment of the paper "Evolvability Of TWEANN In Dynamic Environment"
# .



def __param_to_dict(params):
    if params is None:
        return {}
    r = {}
    for p in params:
        kv = p.split('=')
        r[kv[0]] = eval(kv[1])
    return r

def create_samples(k, w, f, sigma, t_min=0., t_max=2., t_step=0.02, count=2):
    '''
    画出并显示采样数据
    :param k:
    :param w:
    :param f:
    :param sigma:
    :param t_min:
    :param t_max:
    :param t_step:
    :param count:
    :return:
    '''
    samples = []
    ts = []
    t = t_min
    while t < t_max:
        ts.append(t)
        wind = k * np.sin(w * t + f) + np.random.normal(0, sigma, 1)
        samples.append(wind)
        t += t_step
    return ts,samples

def show_cr_curve():
    '''
    显示Complexity-Reward Curve
    :return:
    '''
    result_files = {'dqn':os.path.split(os.path.realpath(__file__))[0] + "\\dqn_result.npy",
                    'ddqn':os.path.split(os.path.realpath(__file__))[0] + "\\ddqn_result.npy",
                    'neat':os.path.split(os.path.realpath(__file__))[0] + "\\neat_result.npy",
                    'hyperneat':os.path.split(os.path.realpath(__file__))[0] + "\\hyperneat_result.npy",
                    'policy':os.path.split(os.path.realpath(__file__))[0] + "\\policy_result.npy"
                    }
    cr = {}
    for key,filename in result_files.items():
        d = np.load(filename)
        cr[key] = d['arr_0'], d['arr_1']

    plt.title('C(Complexity)-R(Reward) Curve')
    colors = ['green','red','skyblue','blue']
    colorindex = 0
    for key,crdata in cr.items():
        complexity,reward = crdata
        plt.plot(complexity,reward,color=colors[colorindex],label=key)
        colorindex += 1

    plt.legend()  # 显示图例

    plt.xlabel('Complexity')
    plt.ylabel('Reward')
    plt.show()
    plt.savefig('cr.png')

def runepoch(name,**kwargs):
    if name == 'dqn':
        dqnrunner.train()
    elif name == 'ddqn':
        ddqnrunner.train()
    elif name == 'neat':
        neatrunner.run()
    elif name == 'hyperneat':
        hyperneatrunner.run()
    elif name == 'policyrunner':
        policyrunner.run()



if __name__ == '__main__':
    while 1:
        user_input = sys.stdin.readline()
        print(user_input)
        if user_input.strip().lower() == 'quit' or user_input.strip().lower() == 'exit':
            break

        inputs = user_input.split(' ')
        if len(inputs)<=0:
            help()
            continue
        command = inputs[0]
        if command is None or command.lower() == '':
            help()
            continue
        params = __param_to_dict(inputs[1:])

        # 显示复杂度曲面
        if command.lower() == 'complexity':
            gc.disable()
            ForceGenerator.compute_all_complex(**params)
        # 显示几种特殊复杂度曲线
        elif command.strip().lower() == 'sin':
            ps = [{'k':5.,'w':0.,'f':np.pi/2,'sigma':0.},    # 风力为常数
                  {'k':5.,'w':0.,'f':np.pi/2,'sigma':1.01},   # 以5为均值,0.1为方差
                  {'k':1.,'w':0.01,'f':0,'sigma':1.01},      # 近线性增长的数据          #
                  {'k':5.,'w':5.,'f':0,'sigma':1.01}         # 周期性数据
                 ]
            subgraphic = 221
            for p in ps:
                ts,samples = create_samples(p['k'],p['w'],p['f'],p['sigma'])
                plt.subplot(subgraphic)
                plt.xlim(xmax=2., xmin=0.)
                plt.ylim(ymax=10, ymin=-10.)
                plt.plot(ts, samples, 'b')
                plt.show()
                subgraphic += 1
        # 执行neat复杂度-奖励曲线计算过程
        elif command.strip().lower() == 'neat':
            neatrunner.run()
        # 执行hyperneat复杂度-奖励曲线计算过程
        elif command.strip().lower() == 'hyperneat':
            hyperneatrunner.run()
        elif command.strip().lower() == 'dqn':
            dqnrunner.train()
        elif command.strip().lower == 'ddqn':
            ddqnrunner.train()

        # 显示所有算法的复杂度奖励曲线
        elif command.strip().lower() == 'crcurve':
            show_cr_curve()

