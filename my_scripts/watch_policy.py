import gym
import statistics
import os

from stable_baselines3 import DQN

env = gym.make("CartPole-v1")
model = DQN.load(os.path.join("checkpoints/2021.03.26-13:08:41", "best_model"), env=env)

# testing
obs = env.reset()
lengths = []
for i in range(10):
    for s in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
            print("Trajectory length: {}".format(s))
            lengths.append(s)
            break

print("\nAvg. trajectory length: {}".format(statistics.mean(lengths)))
env.close()
