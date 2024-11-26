#!/usr/bin/env python
# PyLCG - Linear Congruential Generator for IP Sharding - Developed by acidvegas ib Python (https://github.com/acidvegas/pylcg)
# pylcg/__init__.py

from .core import LCG, IPRange, ip_stream

__version__ = "1.0.3"
__author__  = "acidvegas"
__all__     = ["LCG", "IPRange", "ip_stream"] 
