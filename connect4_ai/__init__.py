"""
Connect 4 AI Module
Contains the neural network agent, trainer, and evaluation functions.
"""

from .agent import Connect4Agent
from .evaluator import PositionEvaluator
from .trainer import SelfPlayTrainer

__all__ = ['Connect4Agent', 'PositionEvaluator', 'SelfPlayTrainer']
