import gym
import numpy as np
import os
from brain import Agent
from utils import plot_learning_curve

os.environ['KMP_DUPLICATE_LIB_OK']='True'
env = gym.make('CartPole-v0')
N = 20
batch_size = 5
n_epochs = 4
alpha = 0.0003
agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, 
                n_epochs=n_epochs, input_dims=env.observation_space.shape)
n_games = 300

filename = 'cart_pole.png'
figure_file = os.path.join('CartPole\PPO', filename)

best_score = env.reward_range[0]
score_history = []

learn_iters = 0
avg_score = 0
n_steps = 0

for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        n_steps += 1
        score += reward
        agent.remember(observation, action, prob, val, reward, done)
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    
    if avg_score > best_score:
        best_score = avg_score
        print('new best')
        # agent.save_models()

    print(f'episode {i+1}: ' 'score=%.2f' % score, 'average_score=%.2f' % avg_score, 'learning_steps=%d' % learn_iters)

agent.save_models()
x = [i+1 for i in range(n_games)]
plot_learning_curve(x, score_history, figure_file)