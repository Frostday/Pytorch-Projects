import gym
import numpy as np
from brain import Agent

env = gym.make('CartPole-v0')
N = 20
batch_size = 5
n_epochs = 4
alpha = 0.0003
agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, 
                n_epochs=n_epochs, input_dims=env.observation_space.shape)
agent.load_models()

observation = env.reset()
done = False
score = 0
while not done:
    action, prob, val = agent.choose_action(observation)
    observation_, reward, done, info = env.step(action)
    env.render()
    score += reward
    observation = observation_
print("Score:", score)