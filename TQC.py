from math import gamma
from random import seed
from termios import VT1
from typing import Any, Dict

import gym
import torch as th

import panda_gym

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3 import HerReplayBuffer

import wandb
from wandb.integration.sb3 import WandbCallback

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": int(1.7e6),
    "env_name": "CartPole-v0",
    "env": "PandaPickAndPlace-v1",
    "seed":int(117454387) , 
    "tensorboard_log":"model_log/tqc_dense2sparse_rew", 
    "verbose":1, 
    "batch_size": 2048, 
    "buffer_size": 1000000, 
    "gamma": 0.95, 
    "learning_rate": 0.001, 
    "num-threads": 1,
    "policy_kwargs": dict(net_arch=[512, 512, 512], n_critics=2), 
    "replay_buffer_class": "HerReplayBuffer",
    "callback": "VideoRecorderCallback & WandBCallback",
    "tau": 0.05,  
    "replay_buffer_kwargs": dict( online_sampling=True, goal_selection_strategy='future', n_sampled_goal=4),
}
run = wandb.init(
    project="TQC",
    notes="Dense2Sparse",
    tags=["D2S", "50k"],
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
    entity="panda_pandp_rew_engineering"
)

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor([screens]), fps=30),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True
th.set_num_threads(8)
# th.cuda.is_available = lambda : False
# device = th.device('cuda' if th.cuda.is_available() else 'cpu')

if __name__ == "__main__":
      env_id = "PandaPickAndPlace-v1"     
      env = gym.make(env_id)

      model = TQC( "MultiInputPolicy", env, seed=int(117454387) , tensorboard_log="model_log/tqc_vidd", verbose=1, batch_size=2048, buffer_size=1000000, gamma=0.95, learning_rate=0.001, policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2), replay_buffer_class=HerReplayBuffer,tau=0.05, replay_buffer_kwargs=dict( online_sampling=True, goal_selection_strategy='future', n_sampled_goal=4))
     
      wandb_callback = WandbCallback(model_save_path=f"models/{run.id}", verbose=1, model_save_freq=100000)
      video_recorder = VideoRecorderCallback(eval_env=gym.make(env_id), render_freq=50000, n_eval_episodes= 10)
      callback = CallbackList([video_recorder])
      
      model.learn(
            total_timesteps=int(2e6), 
            callback=callback, 
            n_eval_episodes=5, 
            eval_freq = 50000, 
            log_interval = -1 )
      
    #   model.save("saved_model/tqc/tqc_dense2sparse_rew")
      