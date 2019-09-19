import pybullet
import gym
import time
from domains.antbullet.ant import Posture
import pybullet_envs  # 这个不能去掉，否则gym不加载AntBulletEnv环境
#############################################################
###################演示机器人的基本动作##########################
#############################################################

def demo_front_left_up():
    '''
    该函数演示如何将左前足抬起向前伸展（这实际上是能走路的第一步）
    具体做法是：在开始机器人没有落地之前，让四足全部收缩直立，落地以后让左前足向前伸展
    :return:
    '''
    psyclient = pybullet.connect(pybullet.DIRECT)
    env = gym.make("AntBulletEnv-v0")
    env.render(mode="human")
    init_obs = env.reset()
    init_posture = Posture(init_obs, env.env.robot)
    init_posture.print()
    cur_posture = init_posture

    count = 0
    action = [0., 0., 0., 0., 0., 0., 0., 0.]
    while 1:
        time.sleep(1. / 2)
        if cur_posture.front_left_foot.ground == 1.:
            action = [-1, -1, 0, 0, 0, 0, 0, 0]  # 落地以后让左前足向前伸展
        else:
            action = [0, 1, 0, -1, 1, -1, 0, 1]  # 没有落地之前，让四足全部收缩直立
        obs, r, done, _ = env.step(action)
        cur_posture = Posture(obs, env.env.robot)
        cur_posture.print()

        still_open = env.render("human")
        count += 1






if __name__ == '__main__':
    demo_front_left_up()