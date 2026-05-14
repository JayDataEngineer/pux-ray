"""Compatibility shim: redirect transformers.initialization to torch.nn.init.

Newer transformers (>=4.50) removed the initialization submodule.
Some model code (e.g. MOSS) does `from transformers import initialization as init`.
This shim makes that import work by aliasing torch.nn.init functions.
"""
import torch.nn.init

normal_ = torch.nn.init.normal_
zeros_ = torch.nn.init.zeros_
ones_ = torch.nn.init.ones_
uniform_ = torch.nn.init.uniform_
constant_ = torch.nn.init.constant_
xavier_uniform_ = torch.nn.init.xavier_uniform_
xavier_normal_ = torch.nn.init.xavier_normal_
kaiming_uniform_ = torch.nn.init.kaiming_uniform_
kaiming_normal_ = torch.nn.init.kaiming_normal_
