#!/usr/bin/env python3
# PyLCG - Linear Congruential Generator for IP Sharding - Developed by acidvegas in Python (https://github.com/acidvegas/pylcg)
# pylcg/state.py

import os


# Leading token of a state file, so a file from an older layout is rejected rather
# than misread as a position in the current one
STATE_MARKER = 'pylcg'

# A long run is worthless if its progress is wiped by a reboot, so state lives in
# the home directory rather than a temp directory
STATE_DIR = os.path.join(os.path.expanduser('~'), '.pylcg')


class StateManager:
	'''Resource-efficient state manager that maintains an open file handle'''

	def __init__(self, cidr: str, shard: int, total: int, seed: int, fingerprint: str, path: str = None, enabled: bool = True, step: int = 0, before_write=None):
		'''
		Initialize the state manager

		:param cidr: Target IP range in CIDR format
		:param shard: Shard number (1-based)
		:param total: Total number of shards
		:param seed: Random seed for LCG
		:param fingerprint: Digest of the effective exclusions
		:param path: Explicit path for the state file (default: None for ~/.pylcg)
		:param enabled: Whether to write the state file at all (default: True)
		:param step: Step the run is starting from, so a run interrupted before it completes a single step rewinds to where it resumed rather than to the beginning (default: 0)
		:param before_write: Callable run immediately before each write, used to flush the consumer so the recorded position can never run ahead of delivered output (default: None)
		'''

		self.cidr         = cidr
		self.shard        = shard
		self.total        = total
		self.seed         = seed
		self.fingerprint  = fingerprint
		self.step         = step
		self.before_write = before_write
		self.handle       = None

		if not enabled:
			self.state_file = None

			return

		# Name the file with the seed, cidr, shard, total and exclusion digest so two
		# runs that differ only by their exclude list cannot share a state file
		if path:
			self.state_file = path
		else:
			file_name       = f'{STATE_MARKER}_{seed}_{cidr.replace("/", "_").replace(":", "_")}_{shard}_{total}_{fingerprint}.state'
			self.state_file = os.path.join(STATE_DIR, file_name)

		# Create the containing directory either way, so an explicit --state-file behaves
		# the same as the default rather than failing on a path that does not exist yet
		directory = os.path.dirname(self.state_file)

		if directory:
			os.makedirs(directory, exist_ok=True)

		# Open without truncating so a resumed file keeps its position until the first
		# write replaces it, rather than sitting empty if the run is killed before then
		self.handle = os.fdopen(os.open(self.state_file, os.O_RDWR | os.O_CREAT, 0o644), 'r+')


	def close(self):
		'''Flush the final position and close the file handle'''

		if self.handle:
			try:
				self.write()
			except BrokenPipeError:
				# The consumer is gone, so the last position could not be confirmed as
				# delivered. Leave the previous one on disk and resume from there.
				pass

			self.handle.close()
			self.handle = None


	@staticmethod
	def load(filepath: str) -> dict:
		'''
		Load state from a state file

		:param filepath: Path to the state file
		'''

		with open(filepath, 'r') as f:
			data = f.read().strip()

		if not data.startswith(f'{STATE_MARKER} '):
			raise ValueError(f'{filepath} is not a pylcg state file (files written by pylcg < 2.0 store a raw LCG position and cannot be resumed, restart the run instead)')

		state = {}

		for field in data.split(' ')[1:]:
			key, _, value = field.partition('=')
			state[key]    = value

		for key in ('seed', 'cidr', 'shard', 'total', 'excludes', 'step'):
			if key not in state:
				raise ValueError(f'{filepath} is missing the "{key}" field')

		state['seed']  = int(state['seed'])
		state['shard'] = int(state['shard'])
		state['total'] = int(state['total'])
		state['step']  = int(state['step'])

		return state


	def write(self):
		'''Write the current position to disk'''

		if not self.handle:
			return

		# Flush the consumer first so the recorded position can never describe IPs
		# still sitting in a buffer. A failed flush is left to propagate, so the run
		# stops instead of recording work that was never delivered.
		if self.before_write:
			self.before_write()

		self.handle.seek(0)
		self.handle.write(f'{STATE_MARKER} seed={self.seed} cidr={self.cidr} shard={self.shard} total={self.total} excludes={self.fingerprint} step={self.step}')
		self.handle.truncate()
		self.handle.flush()


	def __enter__(self):
		'''Context manager entry'''

		return self


	def __exit__(self, exc_type, exc_val, exc_tb):
		'''Context manager exit'''

		self.close()
