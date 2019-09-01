import roboschool
import gym

env = gym.make('RoboschoolAnt-v1')
while True:
    env.step(env.action_space.sample())
    env.render()