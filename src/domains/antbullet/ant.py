import pybullet
import gym
import time

class AntBullet0:
    def __init__(self,policy,epoch,frame,interval=1/60):
        self.policy = policy
        self.interval = interval
        self.epoch = epoch
        self.frame = frame


    def run(self,render=False):
        pybullet.connect(pybullet.DIRECT)
        env = gym.make("AntBulletEnv-v0")
        if render:
            env.render(mode="human")
        env.reset()

        scores = []
        for i in range(self.epoch):
            frame = 0
            score = 0
            obs = env.reset()

            for j in range(self.frame):
                time.sleep(self.interval)
                action = self.policy.act(obs)
                _obs, r, done, _ = env.step(action)
                score += r
                print('epoch=%d,frame=%d,score=%.5f,action=%s' % (i,j,score,action))
                if render:
                    still_open = env.render("human")
                    if still_open == False:break
                if done:break

            scores.append(score)
        return scores
