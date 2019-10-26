import numpy as np



event_ind_begin = 'ind.begin'
event_ind_action = 'ind.action'


def do_activity(ind,session):
    # 初始化
    handlers = session.runParam.handlers
    if event_ind_begin in handlers:
        handlers[event_ind_begin](ind,session)

    _do_growth(ind,session)
    return __do_task(ind,session)


def _do_growth(self,ind,session):
    '''
    执行发育过程
    :param ind:     Individual 个体
    :param session: Session    执行会话
    :return:
    '''
    net = ind.getPhenome()
    # 随机动作执行阶段
    n = 0
    while n < session.runParam.activity.stability_max_count:
        # 计算各个盒子的稳定度
        stability = self._get_stability(ind, session)
        last_stability = stability
        # 若盒子都比较稳定，则跳过随机动作阶段
        if min(stability) >= session.runParam.activity.stability_threshold:
            break
        # 若所有盒子的稳定度都不再变化，则跳出随机动作阶段
        if np.average(stability - last_stability) <= session.runParam.activity.stability_resdual:
            break
        # 找到所有的可执行动作
        actions = net.findActionBox()
        # 选取count个可执行动作组合
        for count in range(len(actions)):
            action_composites = self._get_action_composites(actions, count)
            # 对每个动作组合随机选择若干次输出
            for i in range(session.runParam.activity.stability_action_count):
                net.clear_expect()
                for actions in action_composites:
                    for action in actions:
                        action.setExpect(action.random())
                # 向环境输出动作，得到新的观察
                obs = session.runParam.handlers[event_ind_action](ind,session,[action.expect for action in actions])
                # 网络观察新的环境
                # net.observe(obs)
        n += 1
    # 自主目标执行阶段，为因果推理设定可信度
    tboxes = net.findTBox(sorted='depth')
    aboxes = net.findABox(sorted='depth')
    env_sensors = net.findEnvSensorBox()  # 所有感知盒子
    reliability_list = []
    for tbox in tboxes:
        for i in range(session.runParam.activity.autonomous_targe.count):
            causebox = net.getEffectBoxInTbox(tbox)  # 取得当前tbox的原因部分
            effectbox = net.getEffectBoxInTbox(tbox)  # 取得当前tbox的结果部分
            net.clear_expect()  # 清除所有的期望值
            effectbox.setEffect(effectbox.random())  # 给结果部分随机设定一个期望
            expect_actions = net.inference(tbox, effect=effectbox.effect)  # 执行推理
            obs = session.env.step([action.expect for action in expect_actions])  # 执行动作
            net.observ(obs)  # 观察新环境
            v = net.compute_believe(tbox, causebox, effectbox)  # 计算该tbox的信度
            reliability_list.append(v)
        reliability = np.average(reliability_list)
        tbox.reliability = reliability
    reliability_list = []
    for abox in aboxes:
        for i in range(session.runParam.activity.autonomous_targe.count):
            leftbox_list = net.getLeftInABox(abox)
            rightbox_list = net.getRightInABox(abox)
            net.clear_expect()
            expect_actions = net.inference(abox, left=leftbox_list)
            obs = session.env.step([action.expect for action in expect_actions])  # 执行动作
            net.observ(obs)  # 观察新环境
            v = net.compute_believe(tbox, causebox, effectbox)  # 计算该tbox的信度
            reliability_list.append(v)
            net.clear_expect()
            expect_actions = net.inference(abox, right=rightbox_list)
            obs = session.env.step([action.expect for action in expect_actions])  # 执行动作
            net.observ(obs)  # 观察新环境
            v = net.compute_believe(tbox, causebox, effectbox)  # 计算该tbox的信度
            reliability_list.append(v)
        reliability = np.average(reliability_list)
        abox.reliability = reliability

def __do_task(self,ind,session):
    '''
    执行任务阶段
    :param ind:      Individual 个体
    :param session:  Session    会话
    :return:
    '''
    net = ind.getPhenome()
    extern_target_box = session.env.select_target(ind, session)  # 执行目标设定算法：根据box的类型，选择合适的目标
    if extern_target_box is None or (isinstance(extern_target_box, list) and len(extern_target_box) <= 0):
        return
    inference_box_list = net.select_inference_box(box=None, target=extern_target_box)
    inference_box_list_fitness = []
    #obs = session.env.step([action.expect for action in expect_actions])  # 执行动作
    #net.observ(obs)  # 观察新环境




def _activity_terminted_cond(self,ind,session):
    '''
    是否终止：目标设定的完成程度不再变化，或已经达到预先设定的次数
    :param ind:
    :param session:
    :return:
    '''
    pass