import gym
from datetime import datetime
import os
import matplotlib.pyplot as plt

from stable_baselines3 import DQN

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results

from utils import SaveOnBestTrainingRewardCallback


# setup
CHECKPOINT_STR = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
CHECKPOINT_DIR = "checkpoints/" + CHECKPOINT_STR
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

env_name = "CartPole-v1"
env = gym.make(env_name)
env = Monitor(env, os.path.join(CHECKPOINT_DIR, "training_progress"))  # this monitors the training for later inspection

# MLP policy_name is predefined for DQN, see file stable_baselines3/dqn/policies.py
policy_name = "MlpPolicy"
model = DQN(policy_name, env, verbose=1)

# callback for model training
# saves checkpoint if current version of model is better than all before...
callback = SaveOnBestTrainingRewardCallback(
    check_freq=1000,
    log_dir=CHECKPOINT_DIR,
    save_path=CHECKPOINT_DIR)

# training
total_timesteps = 150000
model.learn(total_timesteps=total_timesteps, callback=callback)

# we don't have to save manually when we use the callback in the model.learn call
# model.save(os.path.join(CHECKPOINT_DIR, "mlp_dqn_cartpole"))

plot_results([CHECKPOINT_DIR],
             num_timesteps=total_timesteps,
             x_axis=results_plotter.X_TIMESTEPS,
             task_name="{} DQN on {}".format(policy_name, env_name),
             figsize=(8, 4))
plt.savefig(os.path.join(CHECKPOINT_DIR, "training_progress.png"))
plt.show()

# restore model from saved...
del model

model = DQN.load(os.path.join(CHECKPOINT_DIR, "best_model"), env=env)

# testing
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()


