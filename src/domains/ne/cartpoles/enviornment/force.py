import numpy as np
import os
import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from numba import jit

import utils.files as files

class ForceGenerator():
    K_MIN = 0.
    K_MAX = 50.
    K_STEP = 0.05
    OMEGE_MIN = -1.0
    OMEGE_MAX = 1.0
    OMEGE_STEP = 0.01
    SIGMA_MIN = 0.01
    SIGMA_MAX = 2.5
    SIGMA_STEP = 0.5

    Complexity = {}
    complex_interval = 0.1
    complex_name = 'cramer_rao'

    #region 初始化和风力计算
    def __init__(self,k,w,f,sigma,**kwargs):
        '''
        风力生成器 ,风力为周期性函数 wind = k * sin(w*t + f) + g(0,sigma) + o(t,a),其中g为高斯噪音,o为动作耦合函数
                    其中耦合函数目前是固定的:若动作连续正确,风力会逐渐减小
        :param k:       float,若k为0,则风力只是高斯噪音,或者高斯噪音+耦合函数
        :param w:       float,若w为0,则风力只是固定值+高斯噪音
        :param f:       float,相位
        :param sigma:   float 高斯噪音方差
        :param kwargs   dict  参数
                              action_func function 动作函数,
                              action_vaild Bool    动作函数是否有效
                              complex_interval float   复杂度计算间隔,缺省是0.1
                              complex_name str 复杂度名称 'LMC-Renyi','lmc','stat','cramer_rao'
        '''
        self.initk,self.k = k,k
        self.initw,self.w = w,w
        self.initf,self.f = f,f
        self.initsigma,self.sigma = sigma,sigma
        self.action_func = kwargs['action_func'] if 'action_func' in kwargs else self._default_action_func
        self.action_vaild = kwargs['action_vaild'] if 'action_vaild' in kwargs else False
        ForceGenerator.complex_interval = kwargs['complex_interval'] if 'complex_interval' in kwargs else 0.1
        ForceGenerator.complex_name = kwargs['complex_name'] if 'complex_name' in kwargs else 'cramer_rao'

    def next(self,t,actions=None):
        '''
        计算下一个风力
        :param t:
        :param actions:
        :return:
        '''
        wind = self.k * np.sin(self.w*t + self.f)
        if self.sigma > 0:
            wind += np.random.normal(0,self.sigma)
        if actions is None or not self.action_vaild or self.action_func is None:
            return wind
        return self.action_func(actions)
    #endregion


    #region 取得下一个更高的复杂度计算
    def promptComplex(self, min_up=0.):
        '''
        提升复杂度
        :param min_up: float 复杂度提升的最小幅度
        :return:
        '''
        curComplex = self.currentComplex()
        if min_up <= 0.:
            return self._promptComplex()
        while True:
            changed, newcomplex, k, w, f, sigma = self._promptComplex()
            if newcomplex is None:
                return False,curComplex,k, w, f, sigma
            if not changed or newcomplex - curComplex >= min_up:
                return changed, newcomplex, k, w, f, sigma

    def _promptComplex(self):
        '''
        按照梯度提升复杂度
        :return:
        '''
        maxk,maxw = self.k,self.w
        maxcomplex = self.find_complex(self.k,self.w,self.f,self.sigma)
        changed = False

        count = 1
        while not changed and count < 100:
            ks = np.arange(self.k - ForceGenerator.K_STEP * count, self.k + ForceGenerator.K_STEP * (count+1),
                           ForceGenerator.K_STEP)
            ks[ks < ForceGenerator.K_MIN] = ForceGenerator.K_MIN
            ks[ks > ForceGenerator.K_MAX] = ForceGenerator.K_MAX
            ws = np.arange(self.w - ForceGenerator.OMEGE_STEP * count, self.w + ForceGenerator.OMEGE_STEP * (count+1),
                           ForceGenerator.OMEGE_STEP)
            ws[ws < ForceGenerator.OMEGE_MIN] = ForceGenerator.OMEGE_MIN
            ws[ws > ForceGenerator.OMEGE_MAX] = ForceGenerator.OMEGE_MAX

            for k in ks:
                for w in ws:
                    if k == self.k and w == self.w:
                        continue
                    complex = self.find_complex(k,w,self.f,self.sigma)
                    if complex is None or maxcomplex is None or complex > maxcomplex:
                        maxcomplex = complex
                        maxk,maxw = k,w
                        changed = True
            if changed:
                break
            count += 1
        self.k = maxk
        self.w = maxw
        return changed,maxcomplex,self.k,self.w,self.f,self.sigma

    #endregion

    #region 计算所有复杂度
    @classmethod
    def compute_all_complex(cls,**kwargs):
        '''
        计算所有的复杂度
        :param kwargs:
              noise tuple(float,str)  当noise为有效数值的时候,表示固定噪音方差缺省是force.py的SIMGA_配置中的最小值
                                      当为"interval"的时候,根据force.py的配置生成多个复杂度三维图'
                                      当为"dimension"的时候,根据force.py的配置高维复杂度数据,不生成三维图'
              file str                复杂度数据的文件名(np格式,缺省为force.npz),以及三维图文件名(缺省为force.$noise.npz)
        :return: bool,str,Union(float,str),tuple
        '''
        noise = kwargs['noise'] if 'noise' in kwargs.keys() else ForceGenerator.SIGMA_MIN
        file = kwargs['file'] if 'file' in kwargs.keys() else 'complex.npz'
        complexfilename = os.path.split(os.path.realpath(__file__))[0] + '\\datas\\' + file

        if noise == 'dimension':
            print('准备高维复杂度计算,这可能会花费数分钟...')
            k = np.arange(ForceGenerator.K_MIN, ForceGenerator.K_MAX, ForceGenerator.K_STEP)
            w = np.arange(ForceGenerator.OMEGE_MIN, ForceGenerator.OMEGE_MAX, ForceGenerator.OMEGE_STEP)
            s = np.arange(ForceGenerator.SIGMA_MIN, ForceGenerator.SIGMA_MAX, ForceGenerator.SIGMA_STEP)
            K, W,S = np.meshgrid(k, w,s)
            C = [[ForceGenerator.compute_complex(k, w, np.pi / 2 if w == 0 else 0.,s) for k, w,s in zip(ks, ws,ss)] for ks, ws,ss in
                 zip(K, W,S)]
            C = np.array(C)

            np.savez(complexfilename, K, W, S,C)
            return True, '', noise, (K, W, S, C)
        else:
            s = []
            if noise == 'interval':
                s = np.arange(ForceGenerator.SIGMA_MIN, ForceGenerator.SIGMA_MAX, ForceGenerator.SIGMA_STEP)
            elif type(noise) is float:
                s = [float(noise)]
            elif type(noise) is list:
                s = noise

            k = np.arange(ForceGenerator.K_MIN, ForceGenerator.K_MAX, ForceGenerator.K_STEP)
            w = np.arange(ForceGenerator.OMEGE_MIN, ForceGenerator.OMEGE_MAX, ForceGenerator.OMEGE_STEP)
            K, W = np.meshgrid(k, w)
            filepath,filename,fileext = files.spiltfilename(complexfilename)
            for sigma in s:
                print('生成噪音方差为'+str(sigma)+'的复杂度数据...')

                starttime = datetime.datetime.now()

                C = [[ForceGenerator.compute_complex(k, w, np.pi / 2 if w == 0 else 0., sigma) for k, w in zip(ks, ws)] for ks, ws in
                    zip(K, W)]
                C = np.array(C)

                np.savez(filepath+'\\'+filename+'.'+str(sigma)+fileext, K, W, C)
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                surf = ax.plot_surface(K, W, C, cmap=cm.coolwarm,
                                       linewidth=0, antialiased=False)
                fig.colorbar(surf, shrink=0.5, aspect=5)
                plt.savefig(filepath +'\\'+filename+'.'+str(sigma)+'.png')
                # long running
                endtime = datetime.datetime.now()
                print('生成结束，耗时'+str((endtime-starttime).seconds)+'秒')
            return True,'',noise

        return False,'参数错误:'+noise,noise
    def currentComplex(self):
        '''
        当前复杂度
        :return:
        '''
        return ForceGenerator.find_complex(self.k,self.w,self.f,self.sigma)

    @classmethod
    def load_complex(cls,sigma=None):
        '''
        读取复杂度数据
        :param sigma: Union(float,list) None,读取高维复杂度数据,list,读取特噪声方差的复杂度数据
        :return:
        '''
        if sigma is None:
            complexfilename = [os.path.split(os.path.realpath(__file__))[0] + '\\datas\\complex.npz']
            d = np.load(complexfilename)
            K, W, S,C = d['arr_0'], d['arr_1'], d['arr_2'],d['arr_3']
            ForceGenerator.Complexity['total'] = (K,W,S,C)
            return
        if type(sigma) is float:
            sigma = [sigma]
        elif type(sigma) is list:
            if len(sigma)<=0:
                sigma = np.arange(ForceGenerator.SIGMA_MIN, ForceGenerator.SIGMA_MAX, ForceGenerator.SIGMA_STEP)
        for s in sigma:
            complexfilename = os.path.split(os.path.realpath(__file__))[0] + '\\datas\\complex' + '.' + str(s) + '.npz'
            d = np.load(complexfilename)
            K, W, C = d['arr_0'], d['arr_1'], d['arr_2']
            ForceGenerator.Complexity[s] = K, W, C

    @classmethod
    def compute_complex(cls, k, w, f, sigma, t_min=0., t_max=2., t_step=0.02, count=2):
        '''
        给定一组参数（k,w,f,sigma）计算复杂度
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
        complexes = []
        for i in range(count):
            t = t_min
            samples = []
            while t < t_max:
                t += t_step
                wind = k * np.sin(w * t + f) + np.random.normal(0, sigma, 1)
                samples.append(wind)
            complexes.append(ForceGenerator.complex(samples))
        return np.average(complexes)

    # endregion


    @classmethod
    def draw_complex(cls,sigma=None):
        '''
        画出复杂度图表
        @:param sigma float 特定噪音方差的复杂度数据
        :return:
        '''

        K, W, C = ForceGenerator.Complexity[sigma]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(K, W, C, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


    @classmethod
    def find_complex(cls,k,w,f,sigma):
        '''
        寻找对应k，w，f和sigma的复杂度
        :param k:
        :param w:
        :param f:
        :param sigma:
        :return:
        '''
        if sigma not in ForceGenerator.Complexity.keys():
            ForceGenerator.load_complex(sigma)
        K, W, C = ForceGenerator.Complexity[sigma]

        windex = -1
        min_error = 0.00001
        for index,ws in enumerate(W):
            if np.abs(ws[0] - w) < min_error:
                windex = index
                break
        kindex = -1
        for index,ks in enumerate(K[0]):
            if np.abs(ks - k) < min_error:
                kindex = index
                break
        #kindex = np.argwhere(np.abs(self.K[0]-k)<min_error)
        if windex == -1 or kindex == -1:
            return None
        return C[windex][kindex]


    #region 复杂度公式实现
    @classmethod
    def complex(cls,samples,name = None):
        '''复杂度函数 '''
        if name is None or name == '':
            name = ForceGenerator.complex_name.lower()

        if name == 'lmc-renyi':
            return ForceGenerator.complex_lmcrenyi(samples)
        elif name == 'lmc':
            return ForceGenerator.complex_lmc(samples)
        elif name == 'stat':
            return ForceGenerator.complex_stat(samples)
        else:
            return ForceGenerator.complex_cramer_rao(samples)

    @classmethod
    def complex_lmcrenyi(cls,samples,alpha=0.5,beta=2.0):
        '''
        LMC-R´enyi complexity
        :param samples:
        :param alpha:
        :param beta:
        :return:
        '''
        stat = ForceGenerator._compute_stat(samples, ForceGenerator.complex_interval)
        Ra,Rb = 0.0,0.0
        for key, value in stat.items():
            p = value / len(samples)
            Ra += p ** alpha
            Rb += p ** beta
        Ra = np.log(Ra) / (1-alpha)
        Rb = np.log(Rb) / (1-beta)
        return np.exp(Ra-Rb)

    @classmethod
    def complex_stat(cls,samples,k=1.3806505 * (10**-23)):
        '''
        简单统计复杂度
        :param samples:
        :param interval:
        :param k:
        :return:
        '''
        stat = ForceGenerator._compute_stat(samples, ForceGenerator.complex_interval)
        s = 0.0
        for key, value in stat.items():
            p = value / len(samples)
            s += p * np.log(p)
        S = -1 * s / np.log(len(stat))
        return S * (1-S)

    @classmethod
    def complex_lmc(cls,samples):
        '''
        LMC复杂度
        :param samples:
        :param interval:
        :return:
        '''
        stat = ForceGenerator._compute_stat(samples, ForceGenerator.complex_interval)
        H,D = 0.,0.
        for key, value in stat.items():
            p = value / len(samples)
            H += p * np.log(p)
            D += (p - 1/len(stat))*(p - 1/len(stat))
        return -1 * H * D


    @classmethod
    def complex_cramer_rao(cls,samples):
        '''
        The　Cr´amer-Rao complexity 复杂度
        :param samples:
        :param interval:
        :return:
        '''
        stat = ForceGenerator._compute_stat(samples,ForceGenerator.complex_interval)
        entropy = 0.
        for key, value in stat.items():
            p = value / len(samples)
            entropy += p * np.log(p)
        return -1 * entropy * np.var(samples)

    @classmethod
    def _compute_stat(cls,samples,interval=0.1):
        datas = [np.round(x, int(np.log10(1 / interval))) for x in samples]
        datas = np.array(datas)
        keys = np.unique(datas)
        stat = {}
        for k in keys:
            mask = (datas == k)
            arr_new = datas[mask]
            v = arr_new.size
            stat[k] = v
        return stat
    #endregion

    #region 交互式风力
    def _default_action_func(self):
        pass
    #endregion

ForceGenerator.load_complex(sigma=1.01)
force_generator = ForceGenerator(0.0,0.0,0.0,1.01)

if __name__ == '__main__':
    ForceGenerator.load_complex(sigma=1.01)
    changed, maxcomplex, k, w, f, sigma = force_generator.promptComplex(5.0)
    print('环境复杂度=%.3f,k=%.2f,w=%.2f,f=%.2f,sigma=%.2f' % (maxcomplex, k, w, f, sigma))

    #generator = ForceGenerator(0,0,0,0)
    ForceGenerator.draw_complex(sigma=0.01)

