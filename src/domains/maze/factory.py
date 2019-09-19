import numpy as np

from mazelib import Maze
from mazelib.generate.CellularAutomaton import CellularAutomaton
import matplotlib.pyplot as plt
from mazelib.solve.BacktrackingSolver import BacktrackingSolver


init_size = 5
max_size = 30
size_step = 2
init_orgin_complex = 0.1
orgin_complex_step = 0.1
default_density = 1.0

class MazeFactory:
    def __init__(self):
        pass
    def demo(self):
        m = self._generate(10, 10, 1.0)
        m.generate_entrances()
        m.solve()
        print(m)
        print(m.solutions)
        print(m.grid)
    def generateAll(self):
        size = init_size

        while size<=max_size:
            orgin_complex = init_orgin_complex
            while orgin_complex <= 1.0:
                m = self._generate(size,size,orgin_complex)
                complex,features = self.__complex__(m)
                print('complex=',complex,'features=',features)
                orgin_complex += orgin_complex_step
            size += size_step
        #print(m)
        #print(m.solutions)
        #self.showPNG(m.grid)

    def _generate(self,w,h,complex):
        '''
        产生一个特定的迷宫
        :param w:        int 迷宫宽度
        :param h:        int 迷宫高度
        :param complex:  float 算法复杂度，与搜索复杂度不同
        :return: Maze 迷宫对象
        '''
        m = Maze()
        m.generator = CellularAutomaton(w, h,complex,default_density)
        m.generate()
        m.solver = BacktrackingSolver()
        #m.start = (1, 1)
        #m.end = (w, h)
        m.generate_entrances()
        m.solve()
        return m

    def __complex__(self,m):
        '''
        计算特定迷宫的控制参数和复杂度
        :param m:
        :return:
        '''
        grid = m.grid
        #print(grid)
        solutions = m.solutions[0]
        stat = {}
        w,h = np.array(grid).shape

        #print(solutions)
        selcounts = []
        for cur in solutions:
            x,y = cur
            feature = []
            sel = 0
            for i in [x-1,x,x+1]:
                for j in [y-1,y,y+1]:
                    if i < 0 or i >= w:continue
                    if j < 0 or j >= h:continue
                    if i == x and j == y:continue
                    feature.append(str(1-grid[i][j]))
                    if grid[i][j] == 0:sel += 1
            feature = ''.join(feature)
            if feature not in stat:stat[feature] = 1
            else:stat[feature] += 1
            selcounts.append(sel)
        #print(stat)
        length = len(solutions)
        entropy = 0.
        for key, value in stat.items():
            p = value / length
            entropy += p * np.log(p)
        complex = -1 * entropy * np.var(np.array(solutions))

        turncount = 0
        backcount = 0
        end = np.array(list(solutions[-1]))
        for i in range(len(solutions)-1):
            dis1 = self._get_manhattan_dis(np.array(list(solutions[i])),end)
            dis2 = self._get_manhattan_dis(np.array(list(solutions[i+1])),end)
            if dis2 > dis1:turncount += 1
            if i == 0:continue
            if solutions[i-1][0] == solutions[i][0]:
                if solutions[i][1] == solutions[i+1][1]:turncount += 1
                elif solutions[i][0] != solutions[i+1][0]:turncount += 1
            elif solutions[i-1][1] == solutions[i][1]:
                if solutions[i][0] == solutions[i+1][0]:turncount += 1
                elif solutions[i][1] != solutions[i+1][1]:turncount += 1
            else:turncount += 1
        return complex,(w,length,turncount)



    def _get_manhattan_dis(self,vector1,vector2):
        #return np.sum(np.abs(vector1 - vector2))
        return np.linalg.norm(vector1 - vector2, ord=1)

    def showPNG(self,grid):
        """Generate a simple image of the maze."""
        plt.figure(figsize=(10, 10))
        plt.imshow(grid, cmap=plt.cm.binary, interpolation='nearest')
        plt.xticks([]), plt.yticks([])
        plt.show()

if __name__ == '__main__':
    f = MazeFactory()
    f.generateAll()
    #f.demo()