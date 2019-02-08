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

def help(command):
    if command is None or command == '':
        print('commands:complexity,sin,dqn,ddqn,neat,hyperneat,policy,quit')
        return
    elif str(command).lower() == 'complexity':
        print('useage:complexity noise=$V file=$filename')
        print('\t\t\t生成复杂度数据和三维图,当$V为有效数值的时候,表示固定噪音方差缺省是force.py的SIMGA_配置中的最小值')
        print('\t\t\t                         当$V的值为"interval"的时候,根据force.py的配置生成多个复杂度三维图')
        print('\t\t\t                         当$V的值为"dimension"的时候,根据force.py的配置高维复杂度数据,不生成三维图')
        print('\t\t\t                         $filename为np格式的复杂度文件名,缺省为force.npz')
    elif str(command).lower() == 'dqn':
        pass

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
        if user_input.strip().lower() == 'quit':
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

        if command.lower() == 'complexity':
            gc.disable()
            ForceGenerator.compute_all_complex(**params)
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
        elif command.strip().lower() == 'crcurve':
            show_cr_curve()

