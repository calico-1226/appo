"""This module contains the Runner class, which is the main class for training and evaluation of agents."""

import os
from copy import deepcopy

import torch

from appo.components.logger import setup_logger_kwargs
from appo.components import EnvironmentEvaluator
from appo.components.utils import get_defaults_kwargs_yaml
from appo.algorithms import REGISTRY


class Runner(object):
    def __init__(
        self,
        algo: str,  # algorithms
        env_id: str,  # environment-name
        log_dir: str,  # the path of log directory
        init_seed: int,  # the seed of experiment
        unparsed_args: list = (),
        use_mpi: bool = False,  # use MPI for parallel execution.
    ):
        """
        Initial Parameters
        """

        self.algo = algo
        self.env_id = env_id
        self.log_dir = log_dir
        self.init_seed = init_seed
        # if MPI is not used, use Python's multi-processing
        self.multiple_individual_processes = False
        self.num_runs = 1
        self.num_cores = 1  # set by compile()-method
        self.training = False
        self.compiled = False
        self.trained = False
        self.use_mpi = use_mpi

        self.default_kwargs = get_defaults_kwargs_yaml(algo=algo, env_id=env_id)
        self.kwargs = self.default_kwargs.copy()
        self.kwargs["seed"] = init_seed
        # update algoorithm kwargs with unparsed arguments from command line
        keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
        values = [eval(v) for v in unparsed_args[1::2]]
        unparsed_dict = {k: v for k, v in zip(keys, values)}
        self.kwargs.update(**unparsed_dict)
        # e.g. Safexp-PointGoal1-v0/ppo
        self.exp_name = os.path.join(self.env_id, self.algo)
        self.logger_kwargs = setup_logger_kwargs(
            base_dir=self.log_dir, exp_name=self.exp_name, seed=init_seed
        )
        # assigned by class methods
        self.model = None
        self.env = None
        self.scheduler = None

    def _evaluate_model(self):
        evaluator = EnvironmentEvaluator(log_dir=self.logger_kwargs["log_dir"])
        evaluator.eval(env=self.env, ac=self.model, num_evaluations=128)
        # Close opened files to avoid number of open files overflow
        evaluator.close()

    def compile(
        self,
        num_runs=1,
        num_cores=os.cpu_count(),
        target="_run_mp_training",
        **kwargs_update
    ):
        """
        Compile the model.

        Either use mpi for parallel computation or run N individual processes.

        Usually, we use parallel computation.

        If MPI is not enabled, but the number of runs is greater than 1, then
            start num_runs parallel processes, where each process is runs individually
            Users can try this situation on your own, we exclude it.

        Args:
            num_runs: Number of total runs that are executed.
            num_cores: Number of total cores that are executed.
            use_mpi: use MPI for parallel execution.
            target:
            kwargs_update


        """
        self.kwargs.update(kwargs_update)
        self.compiled = True
        self.num_cores = num_cores

    def _eval_once(self, actor_critic, env, render) -> tuple:
        done = False
        self.env.render() if render else None
        x = self.env.reset()
        ret = 0.0
        costs = 0.0
        episode_length = 0
        while not done:
            self.env.render() if render else None
            obs = torch.as_tensor(x, dtype=torch.float32)
            action, value, info = actor_critic(obs)
            x, r, done, info = env.step(action)
            costs += info.get("cost", 0)
            ret += r
            episode_length += 1
        return ret, episode_length, costs

    def eval(self, **kwargs) -> None:
        self.model.eval()
        self._evaluate_model()
        self.model.train()  # switch back to train mode

    def train(self, epochs=None, env=None):
        """
        Train the model for a given number of epochs.

        Args:
            epochs: int
                Number of epoch to train. If None, use the standard setting from the
                defaults.yaml of the corresponding algorithm.
            env: gym.Env
                provide a virtual environment for training the model.

        """
        assert self.compiled, "Call model.compile() before model.train()"

        # single model training
        if epochs is None:
            epochs = self.kwargs.pop("epochs")
        else:
            self.kwargs.pop("epochs")  # pop to avoid double kwargs

        # train() can also take a custom env, e.g. a virtual environment
        env_id = self.env_id if env is None else env
        defaults = deepcopy(self.kwargs)
        defaults.update(epochs=epochs)
        defaults.update(logger_kwargs=self.logger_kwargs)
        algo = REGISTRY[self.algo](env_id=env_id, **defaults)
        algo.logger.save_config(algo.params)
        self.model, self.env = algo.learn()

        self.trained = True

    def play(self):
        """Visualize model after training."""
        # assert self.trained, 'Call model.train() before model.play()'
        # self.eval(episodes=5, render=True)

        env_id = self.env_id
        epochs = self.kwargs.pop("epochs")
        defaults = deepcopy(self.kwargs)
        defaults.update(epochs=epochs)
        defaults.update(logger_kwargs=self.logger_kwargs)
        algo = REGISTRY[self.algo](env_id=env_id, **defaults)
        return algo

    def summary(self):
        """print nice outputs to console."""
        raise NotImplementedError
