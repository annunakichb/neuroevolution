

class ActivityOperation:
    def execute(self,session):
        '''
        分别实现所有个体的活动情况
        :param session:
        :return:
        '''
        for ind in session.pop:
            self.__do_activity(ind,session)

    def __do_activity(self,ind,session):
        net = ind.getPhenome()

        while not self._activity_terminted_cond(ind,session):
            # 计算各个盒子的稳定度
            stability = self._get_stability(ind,session)
            # 发育阶段（各个盒子还不稳定）
            if min(stability) < session.runParam.activity.stability_threshold:
                ## 执行随机目标设定算法
                env_sensors = net.findEnvSensorBox()  # 所有感知盒子
                while len(env_sensors) > 0:
                    tboxes = net.findTBox(effect=env_sensors[0].expression)
                    aboxes = net.findABox()
        for env_sensor in env_sensors:

        ## 根据随机目标，推理计算输出活动
        ## 与环境交互输出活动，获取新感知
        ## 执行盒子自适应分布调整算法
        # 学习阶段
        ## 执行外部目标设定算法
        ## 执行推理算法输出活动
        ## 与环境交互输出活动，获取新感知
        ## 执行盒子自适应分布调整算法

    def _activity_terminted_cond(self,ind,session):
        '''
        是否终止：目标设定的完成程度不再变化，或已经达到预先设定的次数
        :param ind:
        :param session:
        :return:
        '''
        pass