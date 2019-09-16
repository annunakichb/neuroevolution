

class MazeFactory:
    def __init__(self,width,height,stepdistance,backcount):
        '''
        创建迷宫的控制参数
        :param width:           float 迷宫宽度
        :param height:          float 迷宫高度
        :param stepdistance:    float 从起点到终点需走过的距离
        :param backcount:       int   出现回退（即选择远离目标才能走对的次数）
        '''