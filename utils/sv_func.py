"""Possible slowly varying functions for the schemes"""
import numpy as np

exp_sv = lambda x, alpha, hyperparameter = 1: np.exp(-x * hyperparameter)
power_sv = lambda x, alpha, hyperparameter=-1: (1 + x) ** (hyperparameter - alpha)
