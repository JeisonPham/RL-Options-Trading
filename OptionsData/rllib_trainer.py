import ray
from ray.rllib.algorithms import ppo
from ray import tune
import gymnasium as gym
from ray.tune.registry import register_env


class CustomExperimentClass():
    def __init__(self, env_generator, env_kwargs):
        ray.shutdown()
        ray.init(num_cpus=2, num_gpus=0)

        register_env("my_env", env_generator)

        self.config = {
            'env': 'my_env',
            'env_config': env_kwargs,
            'framework': 'torch',
            'num_workers': 4,
            'num_gpus': 0,
            'lr': tune.grid_search([1e-4, 1e-3, 1e-2]),
            'gamma': tune.grid_search([0.99, 0.95, 0.9]),
            'entropy_coef': tune.grid_search([0.01, 0.1, 0.5]),
            "model": {
                "fcnet_hiddens": [512, 512],
                "use_lstm": True,
                'max_seq_len': 10
            },
            'train_bath_size': 200,
            'sgd_minibatch_size': 32,
            'num_sgd_iter': 10,
            'rollout_fragment_length': 20,
            'use_gae': True,
            'lambda': 0.95,
            "clip_param": 0.2,
            "vf_clip_param": 10.0,
            "kl_target": 0.01,
            "vf_loss_coeff": 0.5,
        }

    def train(self):
        results = tune.run("PPO", verbose=1,
                           config=self.config,
                           stop={"training_iteration": 500},
                           checkpoint_at_end=True,
                           metric='episode_reward_mean',
                           mode='max')
        checkpoints = results.get_best_trial(metric='episode_reward_mean')
