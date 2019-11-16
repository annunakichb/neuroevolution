import pybullet
import gym
import time
from domains.antbullet.ant import Posture
import pybullet_envs  # 这个不能去掉，否则gym不加载AntBulletEnv环境
#############################################################
###################演示机器人的基本动作##########################
#############################################################

def test():
    psyclient = pybullet.connect(pybullet.DIRECT)
    env = gym.make("Walker2DBulletEnv-v0")
    env.render(mode="human")
    init_obs = env.reset()
    cur_obs = init_obs

    action = [0.,0.,0.,0.,0.,0.]
    parts = ['thigh','leg','foot','thigh_left','leg_left','foot_left','floor']
    parts_posture = {}
    for part in parts:
        parts_posture[part] = []
    count = 0
    while 1:
        time.sleep(1. / 1.)

        obs, r, done, _ = env.step(action)
        still_open = env.render("human")

        print('count',count)
        for part in parts:
            xyz = env.env.robot.parts[part].pose().xyz()
            orientation = env.env.robot.parts[part].get_orientation()
            print('%s=%.2f,%.2f,%.2f\n' % (part,xyz[0],xyz[2],orientation[0]))
            parts_posture[part].append((xyz[0],xyz[2],orientation[0]))


        count += 1
        if count >= 10:
            break;
    for key,value in enumerate(parts_posture):
        print(key,'=',value)



if __name__ == '__main__':
    test()