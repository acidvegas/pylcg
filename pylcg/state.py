#!/usr/bin/env python
# PyLCG - Linear Congruential Generator for IP Sharding - Developed by acidvegas in Python (https://github.com/acidvegas/pylcg)
# pylcg/state.py

import os
import tempfile


class StateManager:
	'''Resource-efficient state manager that maintains an open file handle'''
	
	def __init__(self, seed: int, cidr: str, shard: int, total: int):
		'''
		Initialize the state manager
		
		:param seed: Random seed for LCG
		:param cidr: Target IP range in CIDR format
		:param shard: Shard number (1-based)
		:param total: Total number of shards
		'''
		# Name the file with the seed, cidr, shard, and total number of shards for uniqueness & identification
		file_name  = f'pylcg_{seed}_{cidr.replace("/", "_")}_{shard}_{total}.state'
		
		# Store the state file in the temporary directory
		self.state_file = os.path.join(tempfile.gettempdir(), file_name)
		
		# Open file handle with line buffering (bufsize=1)
		self.handle = open(self.state_file, 'w', buffering=1)
	

	def update(self, lcg_current: int) -> None:
		'''
		Update state efficiently
		
		:param lcg_current: Current LCG state
		'''
		
		self.handle.seek(0)
		self.handle.write(str(lcg_current))
		self.handle.truncate()
	

	def close(self) -> None:
		'''Close the file handle'''

		if hasattr(self, 'handle'):
			self.handle.close()
	

	def __enter__(self):
		'''Context manager entry'''

		return self
	

	def __exit__(self, exc_type, exc_val, exc_tb):
		'''Context manager exit'''

		self.close()


def save_state(seed: int, cidr: str, shard: int, total: int, lcg_current: int):
	'''
	Legacy save_state function for backwards compatibility
	
	:param seed: Random seed for LCG
	:param cidr: Target IP range in CIDR format
	:param shard: Shard number (1-based)
	:param total: Total number of shards
	:param lcg_current: Current LCG state
	'''
	
	with StateManager(seed, cidr, shard, total) as manager:
		manager.update(lcg_current)
