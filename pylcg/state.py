#!/usr/bin/env python
# PyLCG - Linear Congruential Generator for IP Sharding - Developed by acidvegas in Python (https://github.com/acidvegas/pylcg)
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

	# Name the file with the seed, cidr, shard, and total number of shards for uniqueness & identification
	file_name  = f'pylcg_{seed}_{cidr.replace("/", "_")}_{shard}_{total}.state'

	# Store the state file in the temporary directory
	state_file = os.path.join(tempfile.gettempdir(), file_name)

	# Write the current LCG state to the file
	with open(state_file, 'w') as f:
		f.write(str(lcg_current))
