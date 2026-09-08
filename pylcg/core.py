#!/usr/bin/env python3
# PyLCG - Linear Congruential Generator for IP Sharding - Developed by acidvegas in Python (https://github.com/acidvegas/pylcg)
# pylcg/core.py

import bisect
import hashlib
import ipaddress
import random

from .exclude import parse_excludes, optimize_ranges, calculate_excluded_count
from .state   import StateManager


# Numerical Recipes constants. They satisfy the Hull-Dobell theorem for every
# power-of-2 modulus, which is what guarantees the generator visits each index
# exactly once before repeating.
LCG_MULTIPLIER = 1664525
LCG_INCREMENT  = 1013904223

# Formatting an IPv4 address is the single biggest per-IP cost, and joining three
# cached octet strings beats every other approach tested
OCTETS = [str(octet) for octet in range(256)]


def jump_params(m: int, n: int) -> tuple:
	'''
	Compute the (multiplier, increment) of the LCG advanced n steps, modulo m

	Composing x -> a1*x + c1 with x -> a2*x + c2 yields x -> (a1*a2)*x + (a2*c1 + c2),
	so n steps compose in O(log n) via binary exponentiation instead of O(n) iteration.
	This is what makes resuming a partially finished range instant.

	:param m: The LCG modulus
	:param n: Number of steps to compose
	'''

	result_a, result_c = 1, 0
	step_a,   step_c   = LCG_MULTIPLIER % m, LCG_INCREMENT % m

	while n:
		if n & 1:
			result_a, result_c = (step_a * result_a) % m, (step_a * result_c + step_c) % m

		step_a, step_c = (step_a * step_a) % m, (step_a * step_c + step_c) % m

		n >>= 1

	return result_a, result_c


class LCG:
	'''Linear Congruential Generator for deterministic random number generation'''

	def __init__(self, seed: int, m: int = 2**32):
		'''
		Initialize the LCG with a seed and modulus

		:param seed: The seed for the LCG
		:param m: The modulus for the LCG (must be a power of 2, otherwise the generator has no full period)
		'''

		# Hull-Dobell only holds for a power-of-2 modulus. With m=1000 the period
		# collapses to 9 values, so refuse rather than hand back a broken generator.
		if m < 1 or m & (m - 1):
			raise ValueError(f'LCG modulus must be a power of 2, got {m}')

		self.m       = m              # Modulus (should be >= total_valid)
		self.mask    = m - 1          # Modulus is a power of 2, so & replaces the slower %
		self.a       = LCG_MULTIPLIER # Multiplier (chosen for good period)
		self.c       = LCG_INCREMENT  # Increment (chosen for good period)
		self.current = seed & self.mask # Current state


	def jump(self, n: int):
		'''
		Advance the generator n steps in O(log n) rather than stepping n times

		:param n: Number of steps to advance
		'''

		jump_a, jump_c = jump_params(self.m, n)

		self.current = (jump_a * self.current + jump_c) & self.mask


	def next(self) -> int:
		'''Generate next random number'''

		self.current = (self.a * self.current + self.c) & self.mask

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

		self.excluded_ranges = []

		# ipaddress formatting dominates the per-IP cost, so IPv4 gets a direct
		# shift/mask instead. IPv6 keeps the general path.
		self._v4 = network.version == 4

		# Handle exclusions
		if exclude_list:
			# Clip exclusion ranges to target CIDR bounds so an exclusion that only
			# partially overlaps the target subtracts just the overlapping portion
			for range_start, range_end in optimize_ranges(parse_excludes(exclude_list)):
				clipped_start = max(range_start, self.start)
				clipped_end   = min(range_end,   self.end)

				if clipped_start <= clipped_end:
					self.excluded_ranges.append((clipped_start, clipped_end))

			self.total -= calculate_excluded_count(self.excluded_ranges)

		# Running count of valid IPs before each exclusion, and the IP it resumes at,
		# so get_ip_at_index can binary search instead of walking every exclusion
		self._offsets = []
		self._bases   = []

		cursor = self.start
		valid  = 0

		for range_start, range_end in self.excluded_ranges:
			valid += range_start - cursor
			cursor = range_end + 1

			self._offsets.append(valid)
			self._bases.append(cursor)


	@property
	def fingerprint(self) -> str:
		'''Short digest of the effective exclusions, used to detect a state file resumed against a different exclude list'''

		digest = hashlib.sha256(repr(self.excluded_ranges).encode()).hexdigest()

		return digest[:16]


	def get_ip_at_index(self, index: int) -> str:
		'''
		Get IP at specific index without generating previous IPs

		:param index: The index of the IP to get
		'''

		# Validate index
		if not 0 <= index < self.total:
			raise IndexError('IP index out of range')

		# Find how many exclusions sit before this index
		slot = bisect.bisect_right(self._offsets, index) if self._offsets else 0

		# Before the first exclusion it is a direct mapping, otherwise it sits after
		# exclusion `slot - 1`, offset by the valid IPs already skipped past
		value = self.start + index if not slot else self._bases[slot - 1] + index - self._offsets[slot - 1]

		if self._v4:
			return '.'.join((OCTETS[value >> 24], OCTETS[value >> 16 & 255], OCTETS[value >> 8 & 255], OCTETS[value & 255]))

		return str(ipaddress.ip_address(value))


def ip_stream(cidr: str, shard_num: int = 1, total_shards: int = 1, seed: int = None, exclude_list: list = None, resume_step: int = 0, state_file: str = None, state_interval: int = 1_000, write_state: bool = True, before_state_write=None):
	'''
	Stream random IPs from the CIDR range. Optionally supports sharding.
	Each IP in the range will be yielded exactly once in a pseudo-random order.

	:param cidr: Target IP range in CIDR format
	:param shard_num: Shard number (1-based, default: 1)
	:param total_shards: Total number of shards (default: 1)
	:param seed: Random seed for LCG (default: None for random)
	:param exclude_list: List of IPs/CIDRs to exclude (default: None)
	:param resume_step: Step within this shard to resume from (default: 0 for the beginning)
	:param state_file: Where to write the state file (default: None for ~/.pylcg)
	:param state_interval: Write the state file every N steps (default: 1000)
	:param write_state: Whether to write a state file at all (default: True)
	:param before_state_write: Callable run immediately before each state write, so a buffered consumer can flush and keep the recorded position behind what it has actually delivered (default: None)
	'''

	if total_shards < 1:
		raise ValueError('total_shards must be at least 1')

	if shard_num < 1 or shard_num > total_shards:
		raise ValueError(f'shard_num must be between 1 and {total_shards}')

	if resume_step < 0:
		raise ValueError('resume_step cannot be negative')

	if state_interval < 1:
		raise ValueError('state_interval must be at least 1')

	# Initialize IP range with exclusions
	ip_range    = IPRange(cidr, exclude_list)
	total_valid = ip_range.total

	if total_valid <= 0:
		return

	# Use random seed if none provided
	if seed is None:
		seed = random.randint(0, 2**32-1)

	# Smallest power of 2 >= total_valid, so the LCG covers every valid index
	modulus = 1 << max(0, (total_valid - 1).bit_length())

	# Convert to 0-based indexing internally
	shard_index = shard_num - 1

	# Shard i owns steps i+1, i+1+total_shards, i+1+2*total_shards ... which across
	# all shards is exactly the full period, so no gaps and no overlaps
	total_steps = -(-(modulus - shard_index) // total_shards)

	if resume_step >= total_steps:
		return

	# Multiplier/increment that advances the LCG by one whole shard stride. The
	# modulus is a power of 2, so the wraparound is a mask rather than a divide.
	stride_a, stride_c = jump_params(modulus, total_shards)
	stride_mask        = modulus - 1

	# Jump straight to this shard's resume position instead of replaying the sequence
	lcg = LCG(seed, modulus)
	lcg.jump(shard_index + 1 + resume_step * total_shards)

	step = resume_step

	# Everything below is deliberately inlined and bound to locals. A method call per
	# IP costs more than the arithmetic and the formatting combined, so the body of
	# get_ip_at_index is repeated here rather than called. The two must agree exactly,
	# which TestHotLoop.test_inlined_loop_matches_get_ip_at_index enforces.
	offsets, bases, start, is_v4 = ip_range._offsets, ip_range._bases, ip_range.start, ip_range._v4
	octets, join, seek           = OCTETS, '.'.join, bisect.bisect_right

	# Initialize state manager
	with StateManager(cidr, shard_num, total_shards, seed, ip_range.fingerprint, state_file, write_state, step, before_state_write) as state_mgr:
		# Generate IPs
		while step < total_steps:
			index = lcg.current

			# Indices at or above total_valid are the rejection-sampling slack between
			# total_valid and the next power of 2, so they map to no IP
			if index < total_valid:
				if offsets:
					slot  = seek(offsets, index)
					value = start + index if not slot else bases[slot - 1] + index - offsets[slot - 1]
				else:
					value = start + index

				if is_v4:
					yield join((octets[value >> 24], octets[value >> 16 & 255], octets[value >> 8 & 255], octets[value & 255]))
				else:
					yield str(ipaddress.ip_address(value))

			step        += 1
			lcg.current  = (stride_a * index + stride_c) & stride_mask

			# Recorded only after the consumer is done with the yielded IP, so an
			# interrupted run re-emits its last few IPs rather than skipping them
			state_mgr.step = step

			if not step % state_interval:
				state_mgr.write()
