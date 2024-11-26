#!/usr/bin/env python
# PyLCG - Linear Congruential Generator for IP Sharding - Developed by acidvegas ib Python (https://github.com/acidvegas/pylcg)
# pylcg/state.py

import os
import tempfile


def save_state(seed: int, cidr: str, shard: int, total: int, lcg_current: int):
	'''
	Save LCG state to temp file

	:param seed: Random seed for LCG
	:param cidr: Target IP range in CIDR format
	:param shard: Shard number (1-based)
	:param total: Total number of shards
	:param lcg_current: Current LCG state
	'''

	file_name  = f'pylcg_{seed}_{cidr.replace("/", "_")}_{shard}_{total}.state'
	state_file = os.path.join(tempfile.gettempdir(), file_name)

	with open(state_file, 'w') as f:
		f.write(str(lcg_current))
