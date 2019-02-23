#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import numpy as np
import scipy.stats as stats
import os
import csv
import matplotlib.pyplot as plt

from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
#from prompt_toolkit.contrib.completers import WordCompleter
#import click
import matplotlib as mpl
import  gc

from domains.ne.cartpoles.enviornment.force import  ForceGenerator
from domains.ne.cartpoles.enviornment.cartpole import  SingleCartPoleEnv
import domains.ne.cartpoles.dqn_cartpole as dqnrunner
import domains.ne.cartpoles.ddqn_cartpole as ddqnrunner
import domains.ne.cartpoles.neat_feedforeward as neatrunner
import domains.ne.cartpoles.hyperneat_feedforeward as hyperneatrunner
import  domains.ne.cartpoles.policy as policyrunner
import utils.files as files

# The following is a complete experiment of the paper "Evolvability Of TWEANN In Dynamic Environment"
# .

datapath = files.get_data_path() + '\\evolvability\\'


def help():
    print('fig1 显示特征参数和复杂度曲线,对应论文图1')
    print('fig2 显示复杂度曲面,论文图2')
    print('runa alg=算法名 mode=reset|noreset complexunit=20 运行实验,算法名称可以是neat,hyperneat,dqn,ddqn,police,reset表示在每个复杂度重置,complexunit表示每次复杂度提升的值')
    print('runatotal 对多次试验A数据进行汇总')
    print('fig5 mode=[reset|noreset] 显示复杂度-奖励曲线,缺省mode=noreset')
    print('lean 测试在初始状态下自然倾倒所需要的时间')
    print('table1 对实验A的数据执行Mann-Whitney U test')
    print('table2 计算试验A的进化能力值')
    print('fig6 显示奖励范围曲线')
    print('table3 计算试验B中的健壮性')
    print('fig7 显示独立进化和连续进化对比曲线')
    print('table45 计算试验C的Mann-Whitney U test,计算试验C的独立进化和连续进化进化能力和健壮性对比')
    print('table6 不同复杂度跨度下的进化能力和健壮性对比')
    print('fig8 模块度计算')

def run():
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


        docommand(command,params)

def fig1(params):
    ps = [{'k': 5., 'w': 0., 'f': np.pi / 2, 'sigma': 0.},  # 风力为常数
          {'k': 5., 'w': 0., 'f': np.pi / 2, 'sigma': 1.01},  # 以5为均值,0.1为方差
          {'k': 1., 'w': 0.01, 'f': 0, 'sigma': 1.01},  # 近线性增长的数据          #
          {'k': 5., 'w': 5., 'f': 0, 'sigma': 1.01}  # 周期性数据
          ]
    subgraphic = 221
    for p in ps:
        ts, samples = create_samples(p['k'], p['w'], p['f'], p['sigma'])
        plt.subplot(subgraphic)
        plt.xlim(xmax=2., xmin=0.)
        plt.ylim(ymax=10, ymin=-10.)
        plt.plot(ts, samples, 'b')
        plt.show()
        subgraphic += 1
def fig2(params):
    ForceGenerator.draw_complex()
def runa(params):
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
def runatotal(params):
    names = ['dqn', 'ddqn', 'policy']
    mode = params.get('mode', '')
    for name in names:
        if mode == '' or mode.__contains__('noreset'):
            createavgcomplex(name, mode='noreset', count=10)
        if mode == '' or mode.__contains__('reset'):
            createavgcomplex(name, mode='reset', count=10)

def fig5(params):
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
        plt.plot(complexity, reward, label=key)
        colorindex += 1

    plt.legend()  # 显示图例

    plt.xlabel('Complexity')
    plt.ylabel('Reward')
    plt.show()

def table1(params):
    '''执行Mann-Whitney U test'''
    algs = ['neat', 'hyperneat', 'dqn', 'ddqn', 'policy']
    for i, alg1 in enumerate(algs):
        algs2 = algs[i + 1:]
        for j, alg2 in enumerate(algs2):
            complex1, reward1, _ = loadcomplex(alg1, 'noreset')
            complex2, reward2, _ = loadcomplex(alg2, 'noreset')
            u_stat, p_val = stats.mannwhitneyu(reward1, reward2)
            lessthan0_05 = bool(p_val < 0.05)
            print(alg1 + '-' + alg2 + '的u_stat,pvalue为' + str(u_stat) + ',' + str(p_val) + ',p值小于0.05为' + str(
                lessthan0_05))

def table2(params):
    #complexityupperlimit = params['upper'] if 'upper' in params.keys() else 2000.0
    complexityupperlimit = params['upper'] if 'upper' in params else 2000.0
    '''采用公式8'''
    algs = ['neat', 'hyperneat', 'dqn', 'ddqn', 'policy']
    evolvability1 = {}
    t = 0.
    for i, alg in enumerate(algs):
        complex, reward, _ = loadcomplex(alg, 'noreset')
        complex = [c for c in complex if c <= complexityupperlimit]
        reward = reward[:len(complex)]
        avg_complex = np.average(complex)
        avg_reward = np.average(reward)
        evolvability1[alg] = (2 * avg_reward + avg_complex) / (avg_reward + avg_complex)
    ts = np.array(list(evolvability1.values()))
    mean, std = np.average(ts), np.std(ts)
    min, max = np.min(ts), np.max(ts)
    # for k,v in evolvability1.items():
    #    evolvability1[k] = (evolvability1[k]-min)*100/((max-min)*max)
    # evolvability1[k] = np.exp(evolvability1[k])

    print(evolvability1)

    '''按照面积再计算一遍'''
    evolvability2 = {}
    for i, alg in enumerate(algs):
        complex, reward, _ = loadcomplex(alg, 'noreset')
        area = 0.
        for j in range(len(complex) - 1):
            if complex[j] > complexityupperlimit:
                break
            area += (reward[j] + reward[j + 1]) * (complex[j + 1] - complex[j]) / 2
        evolvability2[alg] = area
    ts = np.array(list(evolvability2.values()))
    mean, std = np.average(ts), np.std(ts)
    min, max = np.min(ts), np.max(ts)
    # for k,v in evolvability2.items():
    #    evolvability2[k] = (evolvability2[k]-mean)/(std)

    print(evolvability2)
def fig6(params):
    algs = ['neat', 'hyperneat', 'dqn', 'ddqn', 'policy']
    fig = plt.figure()
    index = 0
    for t, alg in enumerate(algs):
        complex, _, rewardlist = loadcomplex(alg, 'noreset')
        complex = [c for c in complex if c <= 1200]
        rewardlist = rewardlist[:len(complex)]
        ax = fig.add_subplot(5, 1, index + 1)
        ax.set_title(alg)
        if t < len(algs) - 1:
            ax.get_xaxis().set_visible(False)
        index += 1
        xtocks = []
        w = 20.
        for i, c in enumerate(complex):
            ax.vlines(c, np.min(rewardlist[i]), np.max(rewardlist[i]), colors="b")
            # rect = plt.Rectangle((c,np.max(rewardlist[i])-w), w, np.max(rewardlist[i]) - np.min(rewardlist[i]))
            # ax.add_patch(rect)
            xtocks.append(c)
    plt.xticks(xtocks, rotation=60)
    plt.tight_layout()  # 设置默认的间距
    plt.show()

def table3(params):
    algs = ['neat', 'hyperneat', 'dqn', 'ddqn', 'policy']
    robustness = {}
    for t, alg in enumerate(algs):
        complex, _, rewardlist = loadcomplex(alg, 'noreset')
        complex = [c for c in complex if c <= 1200]
        rewardlist = rewardlist[:len(complex)]

        robustness[alg] = compute_robustness(complex, rewardlist)
    print(robustness)

def fig7(params):
    # algs = ['neat','hyperneat','dqn','ddqn','policy']
    algs = ['neat', 'hyperneat', 'policy']
    modes = ['reset', 'noreset']
    datas = {}

    for alg in algs:
        datas[alg] = {}
        for mode in modes:
            if mode == 'noreset':
                complex, reward, rewardlist = loadcomplex(alg, mode, step=100)
            else:
                complex, reward, rewardlist = loadcomplex(alg, mode)
            datas[alg][mode] = (complex, reward, rewardlist)

    fig = plt.figure()
    # plt.title('Continuous-Independent Curve')
    for index, alg in enumerate(algs):
        ax = fig.add_subplot(2, 1, index + 1)
        ax.set_title(alg)
        ax.plot(datas[alg]['noreset'][0], datas[alg]['noreset'][1], label='continuous')
        ax.plot(datas[alg]['reset'][0], datas[alg]['reset'][1], label='independent')
        ax.set_xticks(complex)
        ax.legend()
    plt.tight_layout()  # 设置默认的间距
    # plt.xticks(complex, rotation=60)
    plt.show()

def table45(params):
    algs = ['neat', 'hyperneat', 'policy']
    modes = ['reset', 'noreset']
    datas = {}

    for alg in algs:
        datas[alg] = {}
        for mode in modes:
            if mode == 'noreset':
                complex, reward, rewardlist = loadcomplex(alg, mode, step=100)
            else:
                complex, reward, rewardlist = loadcomplex(alg, mode)
            datas[alg][mode] = (complex, reward, rewardlist)

    # U-test
    for index, alg in enumerate(algs):
        complex1, reward1, rewardlist1 = datas[alg]['noreset']
        complex2, reward2, rewardlist2 = datas[alg]['reset']
        u_stat, p_val = stats.mannwhitneyu(reward1, reward2)
        lessthan0_05 = bool(p_val < 0.05)
        print(
            alg + '的noreset-reset-的u_stat,pvalue为' + str(u_stat) + ',' + str(p_val) + ',p值小于0.05为' + str(lessthan0_05))

        robustness_noreset = compute_robustness(complex1, rewardlist1)
        print(alg + '的noreset' + '健壮性值是' + str(robustness_noreset))
        robustness_reset = compute_robustness(complex2, rewardlist2)
        print(alg + '的reset' + '健壮性值是' + str(robustness_reset))

        avg_complex = np.average(complex1)
        avg_reward = np.average(reward1)
        evo_noreset = (2 * avg_reward + avg_complex) / (avg_reward + avg_complex)
        print(alg + '的noreset' + '进化能力值是' + str(evo_noreset))

        avg_complex = np.average(complex2)
        avg_reward = np.average(reward2)
        evo_reset = (2 * avg_reward + avg_complex) / (avg_reward + avg_complex)
        print(alg + '的reset' + '进化能力值是' + str(evo_reset))

def table6(params):
    labels = ['a', 'b', 'c', 'd']
    filenames = ['neat25.csv', 'neat50.csv', 'neat100.csv', 'neat150.csv']
    datas = {}
    evolvability = {}
    robustness = {}
    for index, filename in enumerate(filenames):
        file = datapath + 'experimentD\\' + filename
        complexity, reward, rewardlist = loadcomplex('neat', mode='noreset', step=0., file=file, rewardtype='avg')
        datas[labels[index]] = (complexity, reward,rewardlist)

        avg_complex = np.average(complexity)
        avg_reward = np.average(reward)
        evolvability[labels[index]] = (2 * avg_reward + avg_complex) / (avg_reward + avg_complex)
        robustness[labels[index]] = compute_robustness(complexity,rewardlist)
    print('进化能力:')
    print(evolvability)
    print('健壮性')
    print(robustness)

def fig8(params):
    file = datapath + 'experimentE\\modulars.csv'
    complex,modularity = [],[]
    with open(file) as f:
        reader = list(csv.reader(f))
        for row in reader:
            complex.append(float(row[0]))
            modularity.append(float(row[1]))

    fig = plt.figure()
    plt.xlabel('complexity')
    plt.ylabel('modularity')

    plt.plot(complex,modularity)
    plt.show()




def docommand(command,params):
        # 显示复杂度曲面
        if command.lower() == 'complexity':
            fig2()
        # 显示几种特殊复杂度曲线
        elif command.strip().lower() == 'fig1':
            fig1()
        # 执行neat复杂度-奖励曲线计算过程
        elif command.strip().lower() == 'runa':
            runa()
        elif command.strip().lower() == 'runatotal':
            runatotal()
        # 显示所有算法的复杂度奖励曲线
        elif command.strip().lower() == 'fig5':
            fig5()
        elif command.strip().lower() == 'lean':
            env = SingleCartPoleEnv()
            env.lean()
        elif command.strip().lower() == 'table1':
            table1()
        elif command.strip().lower() == 'table2':
            table2()
        elif command.strip().lower() == 'fig6':
            fig6(params)
        elif command.strip().lower() == 'table3':
            table3(params)

        elif command.strip().lower() == 'fig7':
            fig7(params)
        elif command.strip().lower() == 'table45':
            table45(params)
        elif command.strip().lower() == 'table6':
            table6(params)
        elif command.strip().lower() == 'fig8':
            fig8(params)

def __param_to_dict(params):
    '''
    输入命令的参数转dict
    :param params: list ['a=1','b=2']
    :return: dict {'a':'1','b':'2'}
    '''
    if params is None:
        return {}
    r = {}
    for p in params:
        kv = p.split('=')
        r[kv[0]] = kv[1]
    return r

def create_samples(k, w, f, sigma, t_min=0., t_max=2., t_step=0.02, count=2):
    '''
    取得时间范围内的风力值
    :param k:
    :param w:
    :param f:
    :param sigma:
    :param t_min:
    :param t_max:
    :param t_step:
    :param count:
    :return: Tuple(list,list) 第一个list是时间序列,第二个list是风力值
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

def show_cr_curve(datas):
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
                sum += rewardlist[j] if rewardlist[j]<=150 else 150.
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

def loadcomplex(alg,mode='noreset',step=0.,file = None,rewardtype='max'):
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
            if rewardtype == 'avg':
                r = np.average(list(map(lambda x: float(x), row[1:])))
            if c > level:
                complexity.append(c)
                reward.append(r)
                reawrdlist.append(row[1:])
                level += step

    return complexity,reward,reawrdlist

def compute_robustness(complex,rewardlist):
    f1, f2 = 0.0, 0.0
    for i in range(len(complex) - 1):
        rewardlist[i] = np.sort(rewardlist[i])
        downpercent = abs(rewardlist[i][-1] - rewardlist[i + 1][0]) / rewardlist[i][-1]
        if downpercent <= 0:
            continue
        uppercent = abs(rewardlist[i + 1][-1] - rewardlist[i + 1][0]) / rewardlist[i + 1][0]
        percent = uppercent / downpercent
        percent = 1.0 if percent > 1 else percent
        f1 += np.log(complex[i])
        f2 += np.log(complex[i]) * percent
    return f2 / f1

if __name__ == '__main__':
    run()


