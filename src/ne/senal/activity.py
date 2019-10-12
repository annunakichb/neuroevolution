

class ActivityOperation:
    def execute(self,session):
        for ind in session.pop:
            self.__do_activity(ind,session)

    def __do_activity(self,ind,session):
        net = ind.getPhenome()
        # 发育阶段
        ## 执行随机目标设定算法
        env_sensors = net.findEnvSensorBox()
        while len(env_sensors) > 0:
            tboxes = net.
        for env_sensor in env_sensors:

        ## 根据随机目标，推理计算输出活动
        ## 与环境交互输出活动，获取新感知
        ## 执行盒子自适应分布调整算法
        # 学习阶段
        ## 执行外部目标设定算法
        ## 执行推理算法输出活动
        ## 与环境交互输出活动，获取新感知
        ## 执行盒子自适应分布调整算法

