#!/usr/bin/env python3
# PyLCG - Linear Congruential Generator for IP Sharding - Developed by acidvegas in Python (https://github.com/acidvegas/pylcg)
# pylcg/__init__.py

from .core import LCG, IPRange, ip_stream, jump_params

__version__ = '3.0.0'
__author__  = 'acidvegas'
__all__     = ['LCG', 'IPRange', 'ip_stream', 'jump_params']
