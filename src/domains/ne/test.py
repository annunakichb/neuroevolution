import numpy as np
import matplotlib.pyplot as plt

# 该程序用于演示高斯过程

def plot_gauss_sample(D):
    '''
    D维高斯过程的采样，并把每次采样化成线段的形式
    :param D:
    :return:
    '''
    p = plt.figure()

    xs = np.linspace(0,1,D)
    print(xs)

    means = np.zeros(D)
    print(means)
    vari = np.eye(D)
    print(vari)
    for i in range(100):
        ys = np.random.multivariate_normal(means,vari)
        print(ys)
        plt.plot(xs,ys)

    plt.show()

def m(x):
    '''
    高斯过程的均值函数，我也不知道为啥取0
    :param x:
    :return:
    '''
    return np.zeros_like(x)

def k(x1,x2,sigma=1,l=1):
    '''
    高斯过程的协方差矩阵生成函数，采用squared exponetial kernal
    :param x1:
    :param x2:
    :param sigma:
    :param l:
    :return:
    '''
    dx = np.expand_dims(x1,1) - np.expand_dims(x2,0)

    return (sigma**2)*np.exp(-((dx/l)**2)/2)

if __name__ == '__main__':
    plot_gauss_sample(2)