import numpy as np
import os
import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
#from numba import jit

INIT_XYZ = [80,80,0]
S_MIN = 0.
S_MAX = 10.
S_PROB_STEP = 0.1
S_VALUE_STEP = 0.5
D_MIN = -30.
D_MAX = 30.
D_PROB_STEP = 0.1
D_VALUE_STEP = 1.

SAMPLE_COUNT = 100
TIME_STEP = 0.01

complex_interval = 0.1
complex_name = 'cramer_rao'
complex_file = 'd:\\complex.npz'



class PreyController():
    #region 位置生成器
    def __init__(self,init_xyz=INIT_XYZ,init_speed_prob=0.0,init_speed_sigma=0.0,init_direction_prob=0.0,init_direction_sigma=0.0):
        '''
        位置生成器 ,位置由速度改变概率，速度改变幅度，方向改变概率，方向改变幅度四个参数决定
        '''
        self.init_xyz,self.position = init_xyz,init_xyz
        self.init_speed_prob,self.speed_prob = init_speed_prob,init_speed_prob
        self.init_speed_sigma,self.speed_sigma = init_speed_sigma,init_speed_sigma
        self.init_direction_prob,self.direction_prob = init_direction_prob,init_direction_prob
        self.init_direction_sigma,self.direction_sigma = init_direction_sigma,init_direction_sigma
        self.current_complex = self.complex()
        self.complexType = 0

    def sample(self):
        '''
        根据当前控制参数采样
        :return:
        '''
        return PreyController._sample(self.init_xyz,SAMPLE_COUNT,TIME_STEP,self.speed_prob,self.speed_sigma,self.direction_prob,self.direction_sigma)

    @classmethod
    def _sample(cls,init_xyz,count,timestep,speed_prob,speed_sigma,direction_prob,direction_sigma):
        '''
        根据控制参数采样
        :param init_xyz: list 初始位置
        :param count:    int 采样数
        :param timestep: int 采用时间间隔,s
        :param speed_prob: float 速度变化概率
        :param speed_sigma: float 速度变化方差
        :param direction_prob: float 方向变化概率
        :param direction_sigma: float 方向变化方差
        :return:
        '''
        np.random.seed(0)
        postions = [init_xyz]
        states = []
        pos = init_xyz
        speed = 0.0
        direction = 0.0
        for i in range(count):
            # 判断是否要改变速度
            if speed_prob > 0 and np.random.rand() < speed_prob:
                speed = np.random.normal(0.0, speed_sigma)
            speed = np.maximum(np.minimum(speed,S_MAX),S_MIN)
            # 判断方向是否要改变
            if direction_prob > 0 and np.random.rand() < direction_prob:
                direction = direction + np.random.normal(0.0,direction_sigma)
            direction = np.maximum(np.minimum(direction,D_MAX),D_MIN)
            # 根据速度和方向生成下一个位置
            states.append((speed,direction))
            step = [speed * timestep * np.cos(direction),speed*timestep*np.sin(direction),0.]
            new_pos = [pos[0]+step[0],pos[1]+step[1],pos[2]+step[2]]
            postions.append(new_pos)
            pos = new_pos
        states.append(states[-1])
        return postions,states

    def promote(self):
        '''
        将复杂度提升一级
        :return:
        '''
        new_complex = self.current_complex
        count = 0
        while new_complex <= self.current_complex:
            if self.complexType == 0:
                self.speed_prob += S_PROB_STEP
            elif self.complexType == 1:
                self.speed_sigma += S_VALUE_STEP
            elif self.complexType == 2:
                self.direction_prob += D_PROB_STEP
            else:
                self.direction_sigma += D_VALUE_STEP

            new_complex = self.complex()
            self.complexType = 0 if self.complexType == 3 else self.complexType + 1
            count += 1
            if count >= 30:
                self.current_complex = new_complex
                return False
        self.current_complex = new_complex
        return True


    def list_complex(self,count=SAMPLE_COUNT,timestep = TIME_STEP):
        '''
        显示控制参数-复杂度分布图
        :param count:
        :param timestep:
        :return:
        '''

        speed_prob, speed_sigma, direction_prob, direction_sigma = self.init_speed_prob,self.init_speed_sigma,self.init_direction_prob,self.init_direction_sigma
        params = []
        complexes = []
        while 1:
            params.append([speed_prob, speed_sigma, direction_prob, direction_sigma])
            complexes.append(self.complex(
                PreyController._sample(self.init_xyz, count, timestep, speed_prob, speed_sigma, direction_prob,direction_sigma)))
            print('param=', params[-1], 'complex=', complexes[-1])

            speed_prob += S_PROB_STEP
            params.append([speed_prob, speed_sigma, direction_prob, direction_sigma])
            complexes.append(self.complex(
                PreyController._sample(self.init_xyz, count, timestep, speed_prob, speed_sigma, direction_prob,
                                       direction_sigma)))
            print('param=', params[-1], 'complex=', complexes[-1])

            speed_sigma += S_VALUE_STEP
            params.append([speed_prob, speed_sigma, direction_prob, direction_sigma])
            complexes.append(self.complex(
                PreyController._sample(self.init_xyz, count, timestep, speed_prob, speed_sigma, direction_prob,
                                       direction_sigma)))
            print('param=', params[-1], 'complex=', complexes[-1])

            direction_prob += D_PROB_STEP
            params.append([speed_prob, speed_sigma, direction_prob, direction_sigma])
            complexes.append(self.complex(
                PreyController._sample(self.init_xyz, count, timestep, speed_prob, speed_sigma, direction_prob,
                                       direction_sigma)))
            print('param=', params[-1], 'complex=', complexes[-1])

            direction_sigma += D_VALUE_STEP
            params.append([speed_prob, speed_sigma, direction_prob, direction_sigma])
            complexes.append(self.complex(
                PreyController._sample(self.init_xyz, count, timestep, speed_prob, speed_sigma, direction_prob,
                                       direction_sigma)))
            print('param=', params[-1], 'complex=', complexes[-1])

            if speed_prob >= 1 or speed_sigma>S_MAX or direction_prob>1 or direction_sigma > D_MAX:
                return

    @classmethod
    def next(cls, speed_prob, speed_sigma, direction_prob, direction_sigma, complex, filename=complex_file,
             interval=0.01):
        data = np.load(filename)
        p, complexes = data['arr_0'], data['arr_1']
        t_speed_prob, t_speed_sigma, t_direction_prob, t_direction_sigma = speed_prob, speed_sigma, direction_prob, direction_sigma
        t = 1
        while t < 100:
            for i in np.arange(-t * S_PROB_STEP, t * S_PROB_STEP, S_PROB_STEP):
                for j in np.arange(-t * S_VALUE_STEP, t * S_VALUE_STEP, S_VALUE_STEP):
                    for m in np.arange(-t * D_PROB_STEP, t * D_PROB_STEP, D_PROB_STEP):
                        for k in np.arange(-t * D_VALUE_STEP, t * D_VALUE_STEP, D_VALUE_STEP):
                            if i == 0 and j == 0 and m == 0 and k == 0: continue
                            index = PreyController.__index(p, speed_prob + i, speed_sigma + j, direction_prob + m,
                                                           direction_sigma + k)
                            if index < 0: continue
                            c = float(complexes[index])
                            if interval > 0 and c - complex >= interval:
                                return p[index], c
                            elif interval == 0 and c > complex:
                                return p[index], c
            t += 1
        return None

    @classmethod
    def __index(cls,params,speed_prob,speed_sigma,direction_prob,direction_sigma):
        for index,p in enumerate(params):
            if p[0] == speed_prob and p[1] == speed_sigma and \
                p[2] == direction_prob and p[3] == direction_sigma:
                return index
        return -1


    def show_complex_distribution(self,count=SAMPLE_COUNT,timestep=TIME_STEP,speed_prob_step=S_PROB_STEP,speed_sigma_step=S_VALUE_STEP,direction_prob_step=D_PROB_STEP,direction_sigma_step=D_VALUE_STEP):
        '''
        显示三维复杂些散点图
        :param self:
        :return:
        '''
        complexes = []
        params = []
        speed_probs = np.linspace(self.init_speed_prob,1.0,(1.0-self.init_speed_prob)/speed_prob_step)
        speed_sigma = 0.5 #np.linspace(self.init_speed_sigma,S_MAX,(S_MAX-self.init_speed_sigma)/speed_sigma_step)
        direction_probs = np.linspace(self.init_direction_prob,1.0,(1.0-self.init_direction_prob)/direction_prob_step)
        direction_sigma = .1

        X,Y = np.meshgrid(speed_probs, direction_probs)

        C = [[self.complex(PreyController._sample(self.init_xyz, count, timestep, speed_prob, speed_sigma, direction_prob,
                                             direction_sigma)) for speed_prob, direction_prob in zip(ks, ws)] for
             ks, ws in
             zip(X, Y)]
        C = np.array(C)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, C, rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


    # region 复杂度公式实现
    def complex(self, samples=None, name=None):
        if samples is None:
            samples = self.sample()
        _,samples = samples
        '''复杂度函数 '''
        if name is None or name == '':
            name = complex_name.lower()

        if name == 'lmc-renyi':
            return PreyController.complex_lmcrenyi(samples)
        elif name == 'lmc':
            return PreyController.complex_lmc(samples)
        elif name == 'stat':
            return PreyController.complex_stat(samples)
        else:
            return PreyController.complex_cramer_rao(samples)

    @classmethod
    def complex_lmcrenyi(cls, samples, alpha=0.5, beta=2.0):
        '''
        LMC-R´enyi complexity
        :param samples:
        :param alpha:
        :param beta:
        :return:
        '''
        stat = PreyController._compute_stat(samples, PreyController.complex_interval)
        Ra, Rb = 0.0, 0.0
        for key, value in stat.items():
            p = value / len(samples)
            Ra += p ** alpha
            Rb += p ** beta
        Ra = np.log(Ra) / (1 - alpha)
        Rb = np.log(Rb) / (1 - beta)
        return np.exp(Ra - Rb)


    @classmethod
    def complex_lmc(cls, samples):
        '''
        LMC复杂度
        :param samples:
        :param interval:
        :return:
        '''
        stat = PreyController._compute_stat(samples, PreyController.complex_interval)
        H, D = 0., 0.
        for key, value in stat.items():
            p = value / len(samples)
            H += p * np.log(p)
            D += (p - 1 / len(stat)) * (p - 1 / len(stat))
        return -1 * H * D

    @classmethod
    def complex_cramer_rao(cls, samples):
        '''
        The　Cr´amer-Rao complexity 复杂度
        :param samples:
        :param interval:
        :return:
        '''

        stat = PreyController._compute_stat(samples, complex_interval)
        entropy = 0.
        for key, value in stat.items():
            p = value / len(samples)
            entropy += p * np.log(p)
        return -1 * entropy * np.var(samples)

    @classmethod
    def _compute_stat(cls,samples,interval=0.1):
        datas = [[np.round(s, 2),np.round(d, 2)] for s,d in samples]
        datas = np.array(datas)
        keys = np.unique(datas)
        stat = {}
        for k in keys:
            mask = (datas == k)
            arr_new = datas[mask]
            v = arr_new.size
            stat[k] = v
        return stat



if __name__ == '__main__':
    controller = PreyController()
    controller.list_complex()
    #controller.show_complex_distribution()