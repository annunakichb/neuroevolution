#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import numpy as np
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
#from prompt_toolkit.contrib.completers import WordCompleter
#import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import  gc
import os
import csv

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
        r[kv[0]] = kv[1]
        #r[kv[0]] = eval(kv[1])
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

def show_cr_curve(*params):
    '''
    显示Complexity-Reward Curve
    :return:
    '''
    result_files = {'dqn':os.path.split(os.path.realpath(__file__))[0] + "\\dqn_result.npy",
                    'ddqn':os.path.split(os.path.realpath(__file__))[0] + "\\ddqn_result.npy",
                    'policy':os.path.split(os.path.realpath(__file__))[0] + "\\policy_result.npy"
                    }
    cr = {}
    for key,filename in result_files.items():
        d = np.load(filename)
        print(key,d)
        complexity, reward= d[0], d[1]
        cr[key] = (complexity,reward)
    # 读取neat
    result_file = os.path.split(os.path.realpath(__file__))[0] + "\\datas\\neat.csv"
    complexity = []
    reward = []
    with open(result_file) as f:
        reader = list(csv.reader(f))
        for row in reader:
            c = float(row[0])
            r = np.max(list(map(lambda x:float(x),row[1:])))
            complexity.append(c)
            reward.append(r)
    cr['neat'] = (complexity,reward)
    # 读取hyperneat
    result_file = os.path.split(os.path.realpath(__file__))[0] + "\\datas\\hyperneat.csv"
    complexity = []
    reward = []
    with open(result_file) as f:
        reader = list(csv.reader(f))
        for row in reader:
            c = float(row[0])
            r = np.max(list(map(lambda x: float(x), row[1:])))
            complexity.append(c)
            reward.append(r)
    cr['hyperneat'] = (complexity,reward)


    # 画出CR曲线
    plt.title('C(Complexity)-R(Reward) Curve')
    colors = ['skyblue','green','red','blue','yellow']
    colorindex = 0
    for key,crdata in cr.items():
        complexity,reward = crdata
        #plt.plot(complexity,reward,color=colors[colorindex],label=key)
        plt.plot(complexity, reward, label=key)
        colorindex += 1

    plt.legend()  # 显示图例

    plt.xlabel('Complexity')
    plt.ylabel('Reward')
    plt.show()
    plt.savefig('cr.png')

def help():
    print('complexity 显示复杂度曲面')
    print('sin 显示特征参数和复杂度曲线')
    print('run alg=算法名 mode=reset|noreset complexunit=20 运行算法,算法名称可以是neat,hyperneat,dqn,ddqn,police,reset表示在每个复杂度重置,complexunit表示每次复杂度提升的值')
    print('crcurve mode=[reset|noreset] epoch 显示复杂度-奖励曲线,epoch包括每次迭代的奖励')
    print('resetcurve alg=算法名')

def createavgcomplex(alg,mode='noreset',count=10):
    datas = []
    complex_len_max,rewardlist_len_max = 0,0
    for i in range(count):
        file = os.path.split(os.path.realpath(__file__))[0] + os.sep + "datas_" + mode + os.sep + \
               alg.strip() + os.sep + alg.strip() + "_" + str(int(i)) +".csv"
        complexity, _, rewardlist = loadcomplex(alg,mode,step=0.,file=file)
        datas.append((complexity,rewardlist))
        if complex_len_max < len(complexity):
            complex_len_max = len(complexity)
        m = max([len(l) for l in rewardlist])
        if rewardlist_len_max < m:
            rewardlist_len_max = m

    complexityes = []
    for i in range(complex_len_max):
        num, sum = 0, 0.0
        for j in range(len(datas)):
            complexity = datas[j][0]
            if len(complexity) <= i:
                continue
            sum += complexity[i]
            num += 1
        complexityes.append(sum / num)

    rs = []
    for i in range(len(complexityes)):
        rlist = []
        for j in range(rewardlist_len_max):
            num, sum = 0, 0.
            for k in range(len(datas)):
                datafiles_rewards = datas[k][1]
                if i >= len(datafiles_rewards):
                    continue
                rewardlist = datafiles_rewards[i]
                if j >= len(rewardlist):
                    continue
                sum += rewardlist[j]
                num += 1
            if sum > 0 and num != 0:
                rlist.append(sum /num)
        rs.append([complexityes[i]]+rlist)

    filename = os.path.split(os.path.realpath(__file__))[0] + os.sep + "datas_" + mode + os.sep + alg.strip() + ".csv"
    out = open(filename, 'w', newline='')
    out.truncate()
    csv_write = csv.writer(out, dialect='excel')
    for row in rs:
        csv_write.writerow(row)
    out.close()

def loadcomplex(alg,mode='noreset',step=0.,file = None):
    if file is None:
        file = os.path.split(os.path.realpath(__file__))[0] + "\\datas_" + mode + "\\" + alg.strip() + ".csv"
    complexity,reward,reawrdlist = [], [],[]
    level = 0.
    with open(file) as f:
        reader = list(csv.reader(f))
        for row in reader:
            row = [float(r) for r in row if r != '']
            c = float(row[0])
            r = np.max(list(map(lambda x: float(x), row[1:])))
            if c > level:
                complexity.append(c)
                reward.append(r)
                reawrdlist.append(row[1:])
                level += step

    return complexity,reward,reawrdlist

if __name__ == '__main__':
    while 1:
        print('输入命令(help查看):')
        user_input = sys.stdin.readline()
        if user_input.strip().lower() == 'quit' or user_input.strip().lower() == 'exit':
            break

        inputs = user_input.split(' ')
        if len(inputs)<=0:
            help()
            continue
        command = inputs[0]
        if command is None or command.lower() == '' or command.lower() == 'help':
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
        elif command.strip().lower() == 'run':
            name = params['alg']
            if name == 'neat':
                neatrunner.run(**params)
            elif name == 'hyperneat':
                hyperneatrunner.run(**params)
            elif name == 'dqn':
                dqnrunner.train(**params)
            elif name == 'ddqn':
                ddqnrunner.train(**params)
            elif name == 'policy':
                policyrunner.run(**params)
        elif command.strip().lower() == 'total':
            names = ['dqn','ddqn','policy']
            mode = params.get('mode','')
            for name in names:
                if mode == '' or mode.__contains__('noreset'):
                    createavgcomplex(name,mode='noreset',count=10)
                if mode == '' or mode.__contains__('reset'):
                    createavgcomplex(name,mode='reset',count=10)
        # 显示所有算法的复杂度奖励曲线
        elif command.strip().lower() == 'crcurve':
            algs = ['neat', 'hyperneat', 'dqn', 'ddqn', 'policy']
            datas = {}
            for alg in algs:
                complex, reward, _ = loadcomplex(alg, 'noreset')
                datas[alg] = (complex, reward)
            plt.title('C(Complexity)-R(Reward) Curve')
            colors = ['skyblue', 'green', 'red', 'blue', 'yellow']
            colorindex = 0
            for key, crdata in datas.items():
                (complexity, reward) = crdata
                # plt.plot(complexity,reward,color=colors[colorindex],label=key)
                plt.plot(complexity, reward, label=key)
                colorindex += 1

            plt.legend()  # 显示图例

            plt.xlabel('Complexity')
            plt.ylabel('Reward')
            plt.show()
            plt.savefig('cr.png')

        elif command.strip().lower() == 'resetcurve':
            algs = ['neat','hyperneat','dqn','ddqn','policy']
            modes = ['reset','noreset']
            datas = {}
            for alg in algs:
                datas[alg] = {}
                for mode in modes:
                    if mode == 'noreset':
                        complex,reward,_ = loadcomplex(alg,mode,step=100)
                    else:
                        complex, reward, _ = loadcomplex(alg, mode)
                    datas[alg][mode] = (complex,reward)

            fig = plt.figure()
            plt.title('noreset-reset Curve')
            for index,alg in enumerate(algs):
                ax = fig.add_subplot(3, 2, index+1)
                ax.set_title(alg)
                ax.plot(datas[alg]['noreset'][0], datas[alg]['noreset'][1], label='noreset')
                ax.plot(datas[alg]['reset'][0], datas[alg]['reset'][1], label='reset')
                ax.legend()

            plt.show()
            plt.savefig('reset.png')


