#!/usr/bin/env python
# PyLCG - Linear Congruential Generator for IP Sharding - Developed by acidvegas in Python (https://github.com/acidvegas/pylcg)
# pylcg/core.py

import ipaddress
import random

from .exclude import parse_excludes, optimize_ranges, calculate_excluded_count
from .state   import StateManager

class LCG:
	'''Linear Congruential Generator for deterministic random number generation'''

	def __init__(self, seed: int, m: int = 2**32):
		'''
		Initialize the LCG with a seed and modulus

		:param seed: The seed for the LCG
		:param m: The modulus for the LCG (should be >= the range of numbers needed)
		'''
		
		self.m       = m          # Modulus (should be >= total_valid)
		self.a       = 1664525    # Multiplier (chosen for good period)
		self.c       = 1013904223 # Increment (chosen for good period)
		self.current = seed       # Current state


	def next(self) -> int:
		'''Generate next random number'''
		self.current = (self.a * self.current + self.c) % self.m
		return self.current



class IPRange:
	'''Memory-efficient IP range iterator'''

	def __init__(self, cidr: str, exclude_list: list = None):
		'''
		Initialize the IPRange with a CIDR and optional exclusion list

		:param cidr: The CIDR range to iterate over
		:param exclude_list: List of IPs or CIDRs to exclude (default: None)
		'''
		
		# Validate target CIDR
		try:
			network = ipaddress.ip_network(cidr, strict=True)
		except ValueError as e:
			raise ValueError(f'Invalid target CIDR: {cidr}') from e
		
		self.start = int(network.network_address)   # Start of the CIDR
		self.end   = int(network.broadcast_address) # End   of the CIDR
		self.total = self.end - self.start + 1      # Total number of IPs in the CIDR
		
		# Handle exclusions
		if exclude_list:
			self.excluded_ranges = optimize_ranges(parse_excludes(exclude_list))  # Optimize the excluded ranges
			self.total          -= calculate_excluded_count(self.excluded_ranges) # Calculate the number of excluded IPs
		else:
			self.excluded_ranges = []


	def get_ip_at_index(self, index: int) -> str:
		'''
		Get IP at specific index without generating previous IPs

		:param index: The index of the IP to get
		'''

		# Validate index
		if not 0 <= index < self.total:
			raise IndexError('IP index out of range')

		# If no exclusions, direct mapping
		if not self.excluded_ranges:
			return str(ipaddress.ip_address(self.start + index))

		# Calculate the target IP
		target     = index
		current_ip = self.start

		# Iterate over each excluded range
		for range_start, range_end in self.excluded_ranges:
			# Calculate valid IPs before this exclusion
			gap = range_start - current_ip

			# Check if our IP is in this gap
			if target < gap:
				return str(ipaddress.ip_address(current_ip + target))
			
			# Adjust target and move past exclusion
			target     -= gap
			current_ip  = range_end + 1
		
		# If we get here, the index falls after all exclusions
		return str(ipaddress.ip_address(current_ip + target))



def ip_stream(cidr: str, shard_num: int = 1, total_shards: int = 1, seed: int = 0, state: int = None, exclude_list: list = None):
	'''
	Stream random IPs from the CIDR range. Optionally supports sharding.
	Each IP in the range will be yielded exactly once in a pseudo-random order.
	'''
	# Convert to 0-based indexing internally
	shard_index = shard_num - 1

	# Initialize IP range with exclusions
	ip_range    = IPRange(cidr, exclude_list)
	total_valid = ip_range.total

	# Use random seed if none provided
	if not seed:
		seed = random.randint(0, 2**32-1)

	# Initialize LCG with a power of 2 modulus >= total_valid
	modulus = 1
	while modulus < total_valid:
		modulus *= 2
	lcg = LCG(seed, modulus)
	
	# Calculate shard size
	shard_size = total_valid // total_shards
	if shard_index < (total_valid % total_shards):
		shard_size += 1

	yielded = 0

	# If resuming from state, set LCG state and count yielded
	if state is not None:
		temp_lcg = LCG(seed, modulus)
		while temp_lcg.current != state:
			idx = temp_lcg.next()
			if idx < total_valid and idx % total_shards == shard_index:
				yielded += 1
		lcg.current = state

	# Initialize state manager
	with StateManager(seed, cidr, shard_num, total_shards) as state_mgr:
		# Generate IPs
		while yielded < shard_size:
			idx = lcg.next()
			if idx < total_valid and idx % total_shards == shard_index:
				yield ip_range.get_ip_at_index(idx)
				yielded += 1
				# Update state for every IP
				state_mgr.update(lcg.current)
