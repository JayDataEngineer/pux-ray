"""Stub for audiotools.ml.BaseModel — parent class of DAC.

Must inherit from nn.Module so DAC gets .apply(), .parameters(), etc.
"""
from torch import nn


class BaseModel(nn.Module):
    pass
