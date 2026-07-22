"""
rl_lab — Reinforcement Learning Lab for Crazy Asian Poker (CAP).

This package trains poker agents for CAP and all supported variants.
It communicates with the CAP Engine through well-defined interfaces only;
it never duplicates poker logic.

Package structure
-----------------
scenarios/      Initial conditions for RL episodes.
envs/           RL episode lifecycle (reset, step, reward, terminal).
observations/   Feature extraction from PokerState for neural networks.
actions/        Abstract action space and mapping to Engine actions.
agents/         Agent implementations (random, rule-based, policy).
models/         Neural network architectures (policy, value).
trainers/       Learning algorithms (PPO, future: Deep CFR, NFSP).
replay/         Replay recording and serialisation.
evaluation/     Benchmark evaluation against fixed opponents.
experiments/    Experiment tracking, configuration loading, checkpointing.
configs/        YAML configuration files.
utils/          Shared utilities.
tests/          Automated tests for all public modules.
"""

__version__ = "0.1.0"