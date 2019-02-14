import domains.ne.cartpoles.enviornment.force as force


def do_evaluation(count,env,action_method,notify_state_changed=None,max_notdone=None):
    '''
    反复执行count次
    :param count: 　　　　　　　　　重复执行次数
    :param env: 　　　　　　　　　　SingleCartPoleEnv
    :param action_method: 　　　　　function 选择动作的函数 action = function(observation)
    :param notify_state_changed: 　 function 通知状态函数 function(observation, action, reward, observation_,step,totalreward)
    :param max_notdone:             int      维持不倒的上限,缺省是SingleCartPoleEnv的max_notdone_count
    :return reward_list,notdone_count_list
    '''
    reward_list = []
    notdone_count_list = []
    total_step = 0

    for i in range(count):
        notdone_count, totalreward, step,total_step = do_until_done(env,action_method,total_step,notify_state_changed=None,max_notdone=None)
        reward_list.append(totalreward)
        notdone_count_list.append(notdone_count)

    return reward_list,notdone_count_list

def do_until_done(env,action_method,total_step,notify_state_changed=None,max_notdone=None):
    '''
    持续执行直到杆子倒下
    :param env:                       SingleCartPoleEnv
    :param action_method:             function 选择动作的函数 action = function(observation)
    :param notify_state_changed:      function 通知状态函数 function(observation, action, reward, observation_,step,totalreward)
    :param max_notdone:               int      维持不倒的上限,缺省是SingleCartPoleEnv的max_notdone_count
    :return:  (notdone_count, totalreward, step)　　持续次数，累计奖励
    '''
    if max_notdone is None:
        max_notdone = env.max_notdone_count

    notdone_count = 0  # 表示连续维持成功(done)的最大次数
    step = 0

    totalreward = 0.  # 累计奖励
    observation = env.reset()
    while True:
        # 执行策略
        observation_, reward, done, info,action = do_action(env,observation,step,action_method)
        # x是车的水平位移，theta是杆离垂直的角度
        x, x_dot, theta, theta_dot = observation_
        # 累计奖励
        totalreward += reward

        # 通知状态改变
        if notify_state_changed is not None:
            notify_state_changed(observation, action, reward, observation_,step,totalreward,total_step)

        observation = observation_

        # 计算连续不倒的次数
        if done:
            return notdone_count,totalreward,step,total_step
        notdone_count += 1
        step += 1
        total_step += 1

        if notdone_count > max_notdone:
            return notdone_count, totalreward, step,total_step



def do_action(env,observation,step,action_method):
    '''
    执行一次动作
    :param env:            SingleCartPoleEnv
    :param observation:    x, x_dot, theta, theta_dot  小车位置(中心为0),小车速度(正向右),偏转弧度(垂直夹角,向右正向),偏转角速度
    :param step:           int 第几步
    :param action_method:  function 选择动作的函数 action = function(observation)
    :return: (observation_, reward, done, info,action)
    '''
    # 执行策略
    env.wind = force.force_generator.next(step * env.tau)
    #input_observation = observation + (env.wind,)
    action = action_method(observation)
    action = 1 if action > 0.5 else 0
    observation_, reward, done, info = env.step(action)

    # x是车的水平位移，theta是杆离垂直的角度
    x, x_dot, theta, theta_dot = observation_

    # 计算奖励
    '''
    prev_theta = observation[2]
    prev_theta_dot = observation[3]
    if prev_theta == 0 and theta == 0:
        reward = np.pi / 4
    elif (prev_theta > 0 and theta > 0) or (prev_theta < 0 and theta < 0):
        if np.abs(prev_theta) >= np.abs(theta):
            reward = np.abs(prev_theta) - np.abs(theta)
        elif np.abs(prev_theta_dot) > np.abs(theta_dot):
            reward = np.abs(prev_theta_dot) - np.abs(theta_dot)
        else:
            reward = np.abs(prev_theta) - np.abs(theta)
    else:
        reward = min(np.abs(prev_theta),np.abs(theta))
    '''
    reward = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5

    return observation_, reward, done, info,action

