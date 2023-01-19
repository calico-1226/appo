# Augmented Proximal Policy Optimization (APPO)
This project provides the open source multi-core parallel implementation of the APPO method introduced in the paper: "Augmented Proximal Policy
Optimization for Safe Reinforcement Learning".

APPO augments the Lagrangian function of the primal constrained problem via attaching a quadratic deviation term.
The constructed multiplier-penalty function dampens cost oscillation for stable convergence while being equivalent to the primal constrained problem to precisely control safety costs.

## Installation
### Create a virtual environment
We recommend using a virtual environment to install the dependencies.
```bash
conda create -n appo python=3.7
conda activate appo
```
You need to install `pytorch` and `torchvision` with your CUDA version, e.g.:
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
### Install APPO
To install APPO from pypi, install it with:
```bash
git clone git@github.com:calico-1226/appo.git
cd appo
pip install -e .
```
### Experimental Environments
For a comprehensive evaluation, we use three safe RL environment to evaluate our APPO and baseline algorithms.
Thus, you need to install environments before using this repository.
We provide copies of Safety-Gym and Bullet-Safety-Gym in `environments` folder, so you can install them with:
```bash
# install Safety-Gym
cd envs/safety-gym
pip install -e .

# install Bullet-Safety-Gym
cd envs/Bullet-Safety-Gym
pip install -e .
```
Please refer to
[Mujoco](https://mujoco.org/),
[Safety-Gym](https://github.com/openai/safety-gym),
[Bullet-Safety-Gym](https://github.com/SvenGronauer/Bullet-Safety-Gym/tree/master/bullet_safety_gym/envs)
for more details of experimental environments.
