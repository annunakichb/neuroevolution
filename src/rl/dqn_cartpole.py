import gym
import universe  # register the universe environments
from universe import wrappers

env = gym.make('gym-core.PongDeterministic-v0')