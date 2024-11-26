#!/usr/bin/env python
# PyLCG - Linear Congruential Generator for IP Sharding - Developed by acidvegas ib Python (https://github.com/acidvegas/pylcg)
# pylcg/core.py

import ipaddress
import random


class LCG:
	'''Linear Congruential Generator for deterministic random number generation'''

	def __init__(self, seed: int, m: int = 2**32):
		self.m       = m
		self.a       = 1664525
		self.c       = 1013904223
		self.current = seed

	def next(self) -> int:
		'''Generate next random number'''

		self.current = (self.a * self.current + self.c) % self.m
		return self.current


class IPRange:
	'''Memory-efficient IP range iterator'''

	def __init__(self, cidr: str):
		network    = ipaddress.ip_network(cidr)
		self.start = int(network.network_address)
		self.total = int(network.broadcast_address) - self.start + 1

	def get_ip_at_index(self, index: int) -> str:
		'''
		Get IP at specific index without generating previous IPs

		:param index: The index of the IP to get
		'''

		if not 0 <= index < self.total:
			raise IndexError('IP index out of range')

		return str(ipaddress.ip_address(self.start + index))


def ip_stream(cidr: str, shard_num: int = 1, total_shards: int = 1, seed: int = 0, state: int = None):
	'''
	Stream random IPs from the CIDR range. Optionally supports sharding.
	Each IP in the range will be yielded exactly once in a pseudo-random order.

	:param cidr: Target IP range in CIDR format
	:param shard_num: Shard number (1-based), defaults to 1
	:param total_shards: Total number of shards, defaults to 1 (no sharding)
	:param seed: Random seed for LCG (default: random)
	:param state: Resume from specific LCG state (default: None)
	'''

	# Convert to 0-based indexing internally
	shard_index = shard_num - 1

	# Initialize IP range and LCG
	ip_range = IPRange(cidr)

	# Use random seed if none provided
	if not seed:
		seed = random.randint(0, 2**32-1)

	# Initialize LCG
	lcg = LCG(seed + shard_index)
	
	# Set LCG state if provided
	if state is not None:
		lcg.current = state

	# Calculate how many IPs this shard should generate
	shard_size = ip_range.total // total_shards

	# Distribute remainder
	if shard_index < (ip_range.total % total_shards):
		shard_size += 1

	# Remaining IPs to yield
	remaining = shard_size

	while remaining > 0:
		index = lcg.next() % ip_range.total
		if total_shards == 1 or index % total_shards == shard_index:
			yield ip_range.get_ip_at_index(index)
			remaining -= 1
			# Save state every 1000 IPs
			if remaining % 1000 == 0:
				from .state import save_state
				save_state(seed, cidr, shard_num, total_shards, lcg.current)
