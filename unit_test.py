#!/usr/bin/env python3
# PyLCG - Linear Congruential Generator for IP Sharding - Developed by acidvegas in Python (https://github.com/acidvegas/pylcg)
# unit_test.py

import gc
import ipaddress
import itertools
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from pylcg         import IPRange, ip_stream, jump_params, LCG
from pylcg.core    import LCG_MULTIPLIER, seed_increment
from pylcg.exclude import optimize_ranges, parse_excludes, read_exclude_file
from pylcg.state   import StateManager, STATE_DIR


class Colors:
	BLUE      = '\033[94m'
	GREEN     = '\033[92m'
	CYAN      = '\033[96m'
	PURPLE    = '\033[95m'
	ORANGE    = '\033[38;5;208m'
	BOLD      = '\033[1m'
	UNDERLINE = '\033[4m'
	ENDC      = '\033[0m'


def print_header(message: str):
	'''
	Print a header for the test suite

	:param message: The message to print
	'''

	print(f'\n\n{Colors.BLUE}{Colors.BOLD}{'='*80}')
	print(f'TEST SUITE: {message}')
	print(f'{'='*80}{Colors.ENDC}\n')


def print_subheader(message: str):
	'''
	Print a subheader for the test suite

	:param message: The message to print
	'''

	print(f'\n{Colors.PURPLE}{Colors.BOLD}▶ {message}{Colors.ENDC}')


def print_success(message: str):
	'''
	Print a success message for the test suite

	:param message: The message to print
	'''

	print(f'{Colors.GREEN}✓ {message}{Colors.ENDC}')


def print_info(message: str):
	'''
	Print an info message for the test suite

	:param message: The message to print
	'''

	print(f'{Colors.CYAN}ℹ INFO: {message}{Colors.ENDC}')


def print_detail(message: str):
	'''
	Print a detail message for the test suite

	:param message: The message to print
	'''

	print(f'{Colors.ORANGE}  → {message}{Colors.ENDC}')


def print_benchmark(operation: str, count: int, elapsed: float):
	'''
	Print a benchmark message for the test suite

	:param operation: The operation to benchmark
	:param count: The number of operations
	:param elapsed: The elapsed time in seconds
	'''

	ops_per_sec = count / elapsed if elapsed > 0 else 0
	print(f'{Colors.CYAN}⚡ BENCHMARK: {Colors.BOLD}{operation}{Colors.ENDC}')
	print(f'{Colors.CYAN}  → Operations: {count:,}')
	print(f'  → Time: {elapsed:.3f}s')
	print(f'  → Speed: {ops_per_sec:,.2f} ops/s{Colors.ENDC}')


def stream(cidr: str, shard_num: int = 1, total_shards: int = 1, seed: int = 0, exclude_list: list = None, resume_step: int = 0) -> list:
	'''
	Collect a whole shard with state writing disabled

	:param cidr: Target IP range in CIDR format
	:param shard_num: Shard number (1-based)
	:param total_shards: Total number of shards
	:param seed: Random seed for LCG
	:param exclude_list: List of IPs/CIDRs to exclude
	:param resume_step: Step within this shard to resume from
	'''

	return list(ip_stream(cidr, shard_num, total_shards, seed, exclude_list, resume_step, write_state=False))


class TestLCG(unittest.TestCase):
	'''Test Linear Congruential Generator functionality'''

	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing LCG Implementation')
		self.test_seed = 12345


	def test_lcg_initialization(self):
		'''Test LCG initialization with different parameters'''

		start_time = time.perf_counter()

		# Test default modulus
		lcg = LCG(self.test_seed)
		self.assertEqual(lcg.m, 2**32, 'Default modulus should be 2^32')
		self.assertEqual(lcg.current, self.test_seed, 'Current state should match seed')

		# Test custom modulus
		lcg = LCG(self.test_seed, 1024)
		self.assertEqual(lcg.m, 1024, 'Custom modulus not set correctly')

		# Test bounds
		for _ in range(1000):
			val = lcg.next()
			self.assertTrue(0 <= val < lcg.m, f'Generated value {val} outside bounds [0, {lcg.m})')

		elapsed = time.perf_counter() - start_time
		print_benchmark('LCG initialization tests', 1000, elapsed)


	def test_non_power_of_two_modulus_rejected(self):
		'''Hull-Dobell only holds for power-of-2 moduli, so anything else must be refused'''

		print_subheader('Testing Non-Power-Of-Two Modulus Rejection')

		for bad in (0, 3, 100, 1000, 1_000_000, 2**32 - 1):
			with self.assertRaises(ValueError, msg=f'Modulus {bad} should have been rejected'):
				LCG(1, bad)

		print_success('Non-power-of-2 moduli rejected (m=1000 would otherwise have a period of 9)')


	def test_full_period(self):
		'''The LCG must visit every value in [0, m) exactly once before repeating'''

		print_subheader('Testing Full Period (Hull-Dobell)')
		start_time = time.perf_counter()
		total      = 0

		for exponent in range(0, 17):
			modulus = 2**exponent
			lcg     = LCG(12345, modulus)
			seen    = {lcg.next() for _ in range(modulus)}
			total  += modulus

			self.assertEqual(len(seen), modulus, f'LCG with m=2^{exponent} did not have a full period')

		print_success('Full period confirmed for every modulus from 2^0 through 2^16')

		elapsed = time.perf_counter() - start_time
		print_benchmark('Full period verification', total, elapsed)


	def test_hull_dobell_conditions(self):
		'''Verify the three Hull-Dobell conditions actually hold for the chosen constants'''

		# c must be coprime to m, which for a power-of-2 modulus means c must be odd
		for seed in range(10_000):
			self.assertEqual(seed_increment(seed) % 2, 1, f'Increment for seed {seed} must be odd to be coprime with a power-of-2 modulus')

		# a-1 must be divisible by every prime factor of m (2 is the only one)
		self.assertEqual((LCG_MULTIPLIER - 1) % 2, 0, 'a-1 must be divisible by 2')

		# a-1 must be divisible by 4 when m is divisible by 4
		self.assertEqual((LCG_MULTIPLIER - 1) % 4, 0, 'a-1 must be divisible by 4')


	def test_seeds_give_distinct_orders(self):
		'''Different seeds must walk different cycles, not the same cycle from a different start'''

		modulus = 2**10
		orders  = set()

		for seed in range(64):
			lcg      = LCG(seed, modulus)
			sequence = [lcg.next() for _ in range(modulus)]
			start    = sequence.index(0)

			# Rotate so every cycle starts at 0, making a rotation of the same order compare equal
			orders.add(tuple(sequence[start:] + sequence[:start]))

		self.assertGreater(len(orders), 1, 'Every seed walked the same cycle')


	def test_lcg_sequence_properties(self):
		'''Test LCG sequence properties'''

		print_subheader('Testing LCG Sequence Properties')
		start_time = time.perf_counter()
		iterations = 100_000

		print_info('Testing deterministic behavior with identical seeds')
		lcg1 = LCG(self.test_seed)
		lcg2 = LCG(self.test_seed)

		sequence1 = [lcg1.next() for _ in range(iterations)]
		sequence2 = [lcg2.next() for _ in range(iterations)]
		self.assertEqual(sequence1, sequence2, 'LCG sequences not deterministic')
		print_success('Sequences are identical as expected')

		print_info('Testing different seeds produce different sequences')
		lcg3      = LCG(self.test_seed + 1)
		sequence3 = [lcg3.next() for _ in range(100)]
		self.assertNotEqual(sequence1[:100], sequence3, 'Different seeds produced identical sequences')
		print_success('Different seeds produced different sequences')

		print_info('Testing sequence distribution')
		values             = sequence1[:10000]
		unique_values      = len(set(values))
		distribution_ratio = unique_values / len(values)
		print_detail(f'Distribution ratio: {distribution_ratio:.2%}')

		self.assertGreater(distribution_ratio, 0.95, 'Poor distribution of values')
		print_success(f'Sequence shows good distribution ({distribution_ratio:.2%} unique values)')

		elapsed = time.perf_counter() - start_time
		print_benchmark('LCG sequence tests', iterations, elapsed)


class TestJumpAhead(unittest.TestCase):
	'''Test the O(log n) jump-ahead that makes instant resume possible'''

	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing Jump Ahead')


	def test_jump_matches_stepping(self):
		'''Jumping n steps must land exactly where stepping n times lands'''

		print_subheader('Testing Jump Equals Stepping')

		for modulus in (2**8, 2**16, 2**32):
			walker = LCG(12345, modulus)

			for step in range(1, 2001):
				walker.next()

				jumper = LCG(12345, modulus)
				jumper.jump(step)

				self.assertEqual(jumper.current, walker.current, f'Jump of {step} did not match stepping with m={modulus}')

		print_success('Jump matches brute-force stepping for every step 1..2000 across three moduli')


	def test_jump_algebra_at_scale(self):
		'''Jumping n1 then n2 must equal jumping n1+n2, even at billions of steps'''

		print_subheader('Testing Jump Algebra At Billion Scale')
		modulus = 2**32

		for n1, n2 in [(3_000_000_000, 1_000_000_000), (4_294_967_295, 1), (2**31, 2**31), (1, 4_294_967_295), (999_999_999, 3_294_967_297)]:
			split = LCG(12345, modulus)
			split.jump(n1)
			split.jump(n2)

			combined = LCG(12345, modulus)
			combined.jump(n1 + n2)

			self.assertEqual(split.current, combined.current, f'Jump algebra broke at n1={n1} n2={n2}')

		# A jump of the full period must return exactly to the starting point
		wrapped = LCG(12345, modulus)
		wrapped.jump(modulus)
		self.assertEqual(wrapped.current, 12345, 'A full-period jump did not return to the seed')

		print_success('Jump algebra holds at billion scale, and a full-period jump returns to the seed')


	def test_jump_speed(self):
		'''A deep jump must be effectively instant rather than replaying the sequence'''

		print_subheader('Testing Jump Speed')
		start_time = time.perf_counter()

		lcg = LCG(12345, 2**32)
		lcg.jump(3_000_000_000)

		elapsed = time.perf_counter() - start_time
		print_detail(f'Jumped 3,000,000,000 steps in {elapsed*1000:.3f} ms')
		self.assertLess(elapsed, 0.05, 'Jumping 3 billion steps should take microseconds, not seconds')
		print_success('Deep jump is instant')


	def test_jump_params_identity(self):
		'''A jump of zero steps must be the identity function'''

		lcg = LCG(12345)

		self.assertEqual(jump_params(2**32, 0, lcg.c), (1, 0), 'A zero-step jump should be the identity')
		self.assertEqual(jump_params(2**32, 1, lcg.c), (LCG_MULTIPLIER, lcg.c), 'A one-step jump should be the base parameters')


class TestExclusionFunctionality(unittest.TestCase):
	'''Test IP exclusion functionality'''

	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing Exclusion Functionality')


	def test_parse_excludes(self):
		'''Test parsing of exclusion lists'''

		start_time = time.perf_counter()
		test_cases = [
			{'input': ['192.168.0.1'],                                'expected_count': 1},
			{'input': ['192.168.0.0/16'],                             'expected_count': 65536},
			{'input': ['192.168.0.1', '10.0.0.0/16', '172.16.0.1'],   'expected_count': 65538}
		]

		for case in test_cases:
			ranges         = parse_excludes(case['input'])
			excluded_count = sum(end - start + 1 for start, end in ranges)
			self.assertEqual(excluded_count, case['expected_count'], f'Wrong exclusion count for {case['input']}')

		# Test invalid inputs
		invalid_inputs = ['256.256.256.256', 'not_an_ip', '192.168.0.0/33']

		for invalid in invalid_inputs:
			with self.assertRaises(ValueError):
				parse_excludes([invalid])

		elapsed = time.perf_counter() - start_time
		print_benchmark('Exclusion parsing tests', len(test_cases) + len(invalid_inputs), elapsed)


	def test_keyword_inside_list(self):
		'''The "private" keyword must expand when it appears as an entry in a list, not just as a bare string'''

		print_subheader('Testing Keyword Expansion Inside A List')

		bare    = IPRange('0.0.0.0/0', 'private').total
		in_list = IPRange('0.0.0.0/0', ['private']).total

		self.assertEqual(bare, in_list, 'The "private" keyword should expand the same way inside a list')

		# The README example mixes the keyword with entries that already sit inside
		# 192.168.0.0/16, so the total must match "private" on its own
		self.assertEqual(IPRange('0.0.0.0/0', ['192.168.1.1', '192.168.100.0/24', 'private']).total, bare, 'Mixing "private" with entries it already covers should not change the total')

		# An entry outside every reserved range must still subtract on top of it
		self.assertEqual(IPRange('0.0.0.0/0', ['8.8.8.0/24', 'private']).total, bare - 256, 'Mixing "private" with an unreserved entry should subtract that entry too')

		print_success('The "private" keyword expands inside lists (the README library example)')


	def test_empty_exclusion_set(self):
		'''An exclude source that resolves to nothing must not raise'''

		print_subheader('Testing Empty Exclusion Set')

		self.assertEqual(optimize_ranges(set()), [], 'An empty exclusion set should optimize to an empty list')

		comments_only = os.path.join(tempfile.gettempdir(), 'pylcg_test_comments_only.txt')

		with open(comments_only, 'w') as f:
			f.write('# nothing here\n\n# more comments\n')

		self.assertEqual(IPRange('10.0.0.0/24', comments_only).total, 256, 'A comment-only exclude file should exclude nothing')
		os.remove(comments_only)

		print_success('Comment-only and empty exclude sources no longer raise IndexError')


	def test_mistyped_file_name_says_so(self):
		'''A single exclude token that is neither an address nor a file must not be reported as a bad IP'''

		print_subheader('Testing Mistyped Exclude File')

		for typo in ('excludes.txt', 'bogus', '/no/such/list.txt'):
			with self.assertRaises(ValueError) as caught:
				parse_excludes(typo)

			self.assertIn('existing file', str(caught.exception), f'{typo} should be reported as a possible missing file')

		# Valid single tokens must still work rather than tripping the new check
		self.assertEqual(len(parse_excludes('10.0.0.0/8')), 1)
		self.assertEqual(len(parse_excludes('8.8.8.8')), 1)

		print_success('A mistyped exclude file is reported as a missing file, valid single entries still parse')


	def test_indented_comments(self):
		'''Comments with leading whitespace must be ignored like any other comment'''

		print_subheader('Testing Indented Comments')
		test_file = os.path.join(tempfile.gettempdir(), 'pylcg_test_indented.txt')

		with open(test_file, 'w') as f:
			f.write('  # indented comment\n\t# tab indented\n10.0.0.1\n   \n  10.0.0.2  \n')

		self.assertEqual(sorted(read_exclude_file(test_file)), ['10.0.0.1', '10.0.0.2'], 'Indented comments should be skipped and entries trimmed')
		self.assertEqual(IPRange('10.0.0.0/24', test_file).total, 254, 'Indented comment file should exclude exactly two IPs')
		os.remove(test_file)

		print_success('Indented comments are ignored and surrounding whitespace is trimmed')


	def test_optimize_ranges(self):
		'''Test IP range optimization'''

		start_time = time.perf_counter()
		test_cases = [
			# Non-overlapping IP ranges
			{
				'input':    {(int(ipaddress.ip_address('192.168.1.0')), int(ipaddress.ip_address('192.168.1.255'))),
				             (int(ipaddress.ip_address('192.168.3.0')), int(ipaddress.ip_address('192.168.3.255')))},
				'expected': [(int(ipaddress.ip_address('192.168.1.0')), int(ipaddress.ip_address('192.168.1.255'))),
				             (int(ipaddress.ip_address('192.168.3.0')), int(ipaddress.ip_address('192.168.3.255')))]
			},
			# Overlapping IP ranges
			{
				'input':    {(int(ipaddress.ip_address('192.168.1.0')),   int(ipaddress.ip_address('192.168.2.0'))),
				             (int(ipaddress.ip_address('192.168.1.128')), int(ipaddress.ip_address('192.168.2.128')))},
				'expected': [(int(ipaddress.ip_address('192.168.1.0')),   int(ipaddress.ip_address('192.168.2.128')))]
			},
			# Adjacent IP ranges
			{
				'input':    {(int(ipaddress.ip_address('192.168.1.0')), int(ipaddress.ip_address('192.168.1.255'))),
				             (int(ipaddress.ip_address('192.168.2.0')), int(ipaddress.ip_address('192.168.2.255')))},
				'expected': [(int(ipaddress.ip_address('192.168.1.0')), int(ipaddress.ip_address('192.168.2.255')))]
			}
		]

		for i, case in enumerate(test_cases, 1):
			self.assertEqual(optimize_ranges(case['input']), case['expected'], 'Range optimization produced incorrect result')
			print_success(f'Test case {i} passed')

		elapsed = time.perf_counter() - start_time
		print_benchmark('IP range optimization tests', len(test_cases), elapsed)


	def test_complex_exclusions(self):
		'''Test complex exclusion scenarios with mixed CIDR and single IP excludes'''

		start_time = time.perf_counter()
		test_range = '10.0.0.0/16'
		excludes   = [
			'10.0.0.0/24',   # Excludes first 256 IPs
			'10.0.255.0/24', # Excludes last 256 IPs
			'10.0.1.1',      # Single IP
			'10.0.1.2',      # Single IP
			'10.0.128.0/25'  # Excludes 128 IPs in the middle
		]

		ip_range       = IPRange(test_range, excludes)
		expected_total = 65536 - 256 - 256 - 2 - 128
		self.assertEqual(ip_range.total, expected_total, f'Expected {expected_total} IPs after exclusions, got {ip_range.total}')

		generated_ips      = set()
		excluded_networks  = [ipaddress.ip_network(ex if '/' in ex else ex + '/32') for ex in excludes]

		for i in range(ip_range.total):
			ip     = ip_range.get_ip_at_index(i)
			ip_obj = ipaddress.ip_address(ip)

			for excluded in excluded_networks:
				self.assertNotIn(ip_obj, excluded, f'Generated excluded IP: {ip}')

			self.assertNotIn(ip, generated_ips, f'Duplicate IP generated: {ip}')
			generated_ips.add(ip)

		self.assertEqual(len(generated_ips), expected_total, 'Generated IP count does not match expected total')

		elapsed = time.perf_counter() - start_time
		print_benchmark('Complex exclusion tests', len(generated_ips), elapsed)


	def test_index_lookup_matches_linear_walk(self):
		'''The binary-search index lookup must agree with a naive linear walk on every index'''

		print_subheader('Testing Indexed Lookup Against A Linear Walk')

		def linear_walk(ip_range: IPRange, index: int) -> str:
			'''
			Reference implementation of the index lookup, walking every exclusion in order

			:param ip_range: The IPRange to look up in
			:param index: The index of the IP to get
			'''

			target     = index
			current_ip = ip_range.start

			for range_start, range_end in ip_range.excluded_ranges:
				gap = range_start - current_ip

				if target < gap:
					return str(ipaddress.ip_address(current_ip + target))

				target     -= gap
				current_ip  = range_end + 1

			return str(ipaddress.ip_address(current_ip + target))

		for cidr, excludes in [('10.0.0.0/22', ['10.0.1.0/24', '10.0.2.5', '10.0.3.255']),
		                       ('10.0.0.0/24', ['10.0.0.0/28', '10.0.0.100/30', '10.0.0.255']),
		                       ('10.0.0.0/20', None),
		                       ('10.0.0.0/24', ['10.0.0.0/25'])]:
			ip_range = IPRange(cidr, excludes)

			for i in range(ip_range.total):
				self.assertEqual(ip_range.get_ip_at_index(i), linear_walk(ip_range, i), f'Indexed lookup disagreed with linear walk at index {i} of {cidr}')

		print_success('Indexed lookup is identical to the linear walk across every index of four ranges')


class TestIPRange(unittest.TestCase):
	'''Test IP range functionality'''

	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing IP Range Implementation')
		self.test_cidr = '192.168.0.0/24'


	def test_ip_range_initialization(self):
		'''Test IP range initialization'''

		start_time = time.perf_counter()
		test_cases = [
			('192.168.0.0/16', None,                              65536),
			('192.168.0.0/16', ['192.168.0.1'],                    65535),
			('192.168.0.0/16', ['192.168.0.0/17'],                 32768),
			('192.168.0.0/16', ['192.168.0.1', '192.168.0.2'],     65534),
			('10.0.0.0/16',    None,                               65536)
		]

		for cidr, excludes, expected_total in test_cases:
			self.assertEqual(IPRange(cidr, excludes).total, expected_total, f'Wrong total for {cidr} with excludes {excludes}')

		with self.assertRaises(ValueError):
			IPRange('invalid_cidr')

		elapsed = time.perf_counter() - start_time
		print_benchmark('IP range initialization tests', len(test_cases) + 1, elapsed)


	def test_ip_generation(self):
		'''Test IP generation functionality'''

		start_time = time.perf_counter()
		ip_range   = IPRange(self.test_cidr)

		for i in range(256):
			self.assertIn(ipaddress.ip_address(ip_range.get_ip_at_index(i)), ipaddress.ip_network(self.test_cidr))

		with self.assertRaises(IndexError):
			ip_range.get_ip_at_index(-1)

		with self.assertRaises(IndexError):
			ip_range.get_ip_at_index(256)

		elapsed = time.perf_counter() - start_time
		print_benchmark('IP generation tests', 256, elapsed)


	def test_fingerprint_tracks_effective_exclusions(self):
		'''The fingerprint must change with the exclusions and ignore how they were written'''

		print_subheader('Testing Exclusion Fingerprint')

		self.assertNotEqual(IPRange('10.0.0.0/24', ['10.0.0.1']).fingerprint, IPRange('10.0.0.0/24', ['10.0.0.2']).fingerprint, 'Different exclusions should fingerprint differently')
		self.assertEqual(IPRange('10.0.0.0/24', ['10.0.0.1', '10.0.0.2']).fingerprint, IPRange('10.0.0.0/24', ['10.0.0.2', '10.0.0.1']).fingerprint, 'Exclusion order should not change the fingerprint')
		self.assertEqual(IPRange('10.0.0.0/24', ['10.0.0.0/31']).fingerprint, IPRange('10.0.0.0/24', ['10.0.0.0', '10.0.0.1']).fingerprint, 'Equivalent exclusions should fingerprint identically')

		print_success('Fingerprint reflects the effective exclusions, not how they were spelled')


class TestSharding(unittest.TestCase):
	'''Test IP sharding functionality'''

	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing Sharding Implementation')
		self.test_cidr    = '192.168.0.0/16'
		self.test_seed    = 12345
		self.total_shards = 4
		self.excludes     = ['192.168.0.0/24', '192.168.1.1', '192.168.1.2', '192.168.255.0/24']


	def test_step_ownership_partitions_the_period(self):
		'''Every step of the sequence must belong to exactly one shard: no gaps, no overlaps'''

		print_subheader('Testing Step Ownership')

		for exponent in range(0, 14):
			modulus = 2**exponent

			for shards in range(1, 40):
				owned = sum(-(-(modulus - shard) // shards) for shard in range(shards))
				self.assertEqual(owned, modulus, f'Shard step counts summed to {owned}, expected {modulus} (m=2^{exponent}, {shards} shards)')

		print_success('Step ownership partitions the period exactly for m=2^0..2^13 across 1..39 shards')


	def test_shard_distribution(self):
		'''Test shard size distribution'''

		print_subheader('Testing IP Shard Distribution')
		start_time = time.perf_counter()

		shard_contents = {}
		all_ips        = set()
		expected_total = int(ipaddress.ip_network(self.test_cidr).num_addresses)

		for shard in range(1, self.total_shards + 1):
			shard_ips = set()

			for ip in ip_stream(self.test_cidr, shard, self.total_shards, self.test_seed, write_state=False):
				self.assertNotIn(ip, shard_ips, f'Duplicate IP in shard {shard}: {ip}')
				self.assertNotIn(ip, all_ips, f'IP {ip} appears in multiple shards')

				shard_ips.add(ip)
				all_ips.add(ip)

			shard_contents[shard] = shard_ips
			print_success(f'Shard {shard} generated: {len(shard_ips):,} IPs')

		self.assertEqual(len(all_ips), expected_total, 'Missing or extra IPs')

		# Without exclusions total_valid is exactly the modulus, so every step yields
		# an IP and the shards balance exactly
		shard_sizes = [len(ips) for ips in shard_contents.values()]
		print_detail(f'Shard sizes: {shard_sizes}')
		self.assertLessEqual(max(shard_sizes) - min(shard_sizes), 1, 'Shards are not balanced')
		print_success('Shards are evenly balanced')

		elapsed = time.perf_counter() - start_time
		print_benchmark('Shard distribution tests', len(all_ips), elapsed)


	def test_shard_determinism(self):
		'''Test shard generation determinism'''

		start_time = time.perf_counter()

		first_run  = stream(self.test_cidr, 1, self.total_shards, self.test_seed)
		second_run = stream(self.test_cidr, 1, self.total_shards, self.test_seed)
		self.assertEqual(first_run, second_run, 'Shard generation not deterministic')

		different_seed = stream(self.test_cidr, 1, self.total_shards, self.test_seed + 1)
		self.assertNotEqual(first_run, different_seed, 'Different seeds produced same sequence')

		elapsed = time.perf_counter() - start_time
		print_benchmark('Shard determinism tests', len(first_run) * 3, elapsed)


	def test_sharding_with_exclusions(self):
		'''Test sharding with exclusions to verify complete coverage'''

		start_time        = time.perf_counter()
		expected_total    = IPRange(self.test_cidr, self.excludes).total
		shard_contents    = {}
		all_ips           = set()
		excluded_networks = [ipaddress.ip_network(ex if '/' in ex else ex + '/32') for ex in self.excludes]

		for shard in range(1, self.total_shards + 1):
			shard_ips = set()

			for ip in ip_stream(self.test_cidr, shard, self.total_shards, self.test_seed, self.excludes, write_state=False):
				self.assertNotIn(ip, shard_ips, f'Duplicate IP in shard {shard}: {ip}')
				self.assertNotIn(ip, all_ips, f'IP {ip} appears in multiple shards')

				ip_obj = ipaddress.ip_address(ip)

				for excluded in excluded_networks:
					self.assertNotIn(ip_obj, excluded, f'Generated excluded IP: {ip}')

				shard_ips.add(ip)
				all_ips.add(ip)

			shard_contents[shard] = shard_ips
			print_success(f'Shard {shard}: {len(shard_ips):,} IPs')

		self.assertEqual(len(all_ips), expected_total, f'Expected {expected_total} IPs, got {len(all_ips)}')

		# With exclusions, total_valid is below the modulus so some steps map to no IP.
		# Shards therefore balance statistically rather than exactly. Coverage stays
		# exact; only the split moves. Allow a generous multiple of the expected spread.
		shard_sizes = [len(ips) for ips in shard_contents.values()]
		spread      = max(shard_sizes) - min(shard_sizes)
		tolerance   = max(16, int(8 * (expected_total / self.total_shards) ** 0.5))
		print_detail(f'Shard sizes: {shard_sizes} (spread {spread}, tolerance {tolerance})')
		self.assertLessEqual(spread, tolerance, f'Shards are unreasonably unbalanced. Sizes: {shard_sizes}')

		elapsed = time.perf_counter() - start_time
		print_benchmark('Sharding with exclusions tests', len(all_ips), elapsed)


	def test_coverage_across_many_shard_counts(self):
		'''Full coverage with no duplicates across many range shapes and shard counts'''

		print_subheader('Testing Coverage Across Shard Counts')
		start_time = time.perf_counter()
		cases      = [
			('10.0.0.0/24',    None),
			('10.0.0.0/24',    ['10.0.0.5', '10.0.0.9/31']),
			('10.0.0.0/22',    ['10.0.1.0/24']),
			('172.16.0.0/20',  ['172.16.4.0/22', '172.16.15.255']),
			('10.0.0.0/28',    ['10.0.0.0/29'])
		]
		checked = 0

		for cidr, excludes in cases:
			expected = IPRange(cidr, excludes).total

			for shards in (1, 2, 3, 4, 5, 7, 8, 13, 16, 31, 64, 257):
				collected = [ip for shard in range(1, shards + 1) for ip in stream(cidr, shard, shards, 7, excludes)]

				self.assertEqual(len(collected), expected, f'{cidr} with {shards} shards yielded {len(collected)} IPs, expected {expected}')
				self.assertEqual(len(set(collected)), expected, f'{cidr} with {shards} shards produced duplicates')
				checked += 1

		print_success(f'{len(cases)} range shapes x 12 shard counts: complete coverage, zero duplicates')

		elapsed = time.perf_counter() - start_time
		print_benchmark('Shard coverage combinations', checked, elapsed)


	def test_shard_count_above_range_size(self):
		'''More shards than IPs must still cover the range exactly once'''

		collected = [ip for shard in range(1, 65) for ip in stream('10.0.0.0/30', shard, 64, 3)]

		self.assertEqual(sorted(collected), ['10.0.0.0', '10.0.0.1', '10.0.0.2', '10.0.0.3'], 'Oversized shard count lost or duplicated IPs')


class TestResume(unittest.TestCase):
	'''Test resuming a partially finished shard'''

	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing Resume')


	def test_exhaustive_resume(self):
		'''Resuming at every possible step must reproduce the remaining sequence exactly'''

		print_subheader('Testing Every Possible Resume Point')
		start_time = time.perf_counter()
		checked    = 0

		for cidr, excludes in [('10.0.0.0/25', None),
		                       ('10.0.0.0/25', ['10.0.0.5', '10.0.0.9/31']),
		                       ('10.0.0.0/26', ['10.0.0.0/28']),
		                       ('10.0.0.0/28', ['10.0.0.0/29'])]:
			total_valid = IPRange(cidr, excludes).total
			modulus     = 1 << max(0, (total_valid - 1).bit_length())

			for shards in (1, 3, 4, 8):
				for shard in range(1, min(shards, 3) + 1):
					full        = stream(cidr, shard, shards, 7, excludes)
					total_steps = -(-(modulus - (shard - 1)) // shards)
					consumed    = 0

					for step in range(total_steps + 1):
						tail = stream(cidr, shard, shards, 7, excludes, resume_step=step)

						self.assertEqual(tail, full[consumed:], f'Resume at step {step} of {cidr} shard {shard}/{shards} did not match')
						checked += 1

						# The next step consumes one more IP only if it was not rejected
						if step < total_steps and len(tail) > len(stream(cidr, shard, shards, 7, excludes, resume_step=step + 1)):
							consumed += 1

		print_success(f'{checked:,} resume points reproduced the remaining sequence byte for byte')

		elapsed = time.perf_counter() - start_time
		print_benchmark('Exhaustive resume verification', checked, elapsed)


	def test_resume_past_the_end(self):
		'''Resuming beyond the last step must yield nothing rather than wrapping around'''

		full = stream('10.0.0.0/28', 1, 1, 5)

		self.assertEqual(stream('10.0.0.0/28', 1, 1, 5, resume_step=16), [], 'Resuming at the end should yield nothing')
		self.assertEqual(stream('10.0.0.0/28', 1, 1, 5, resume_step=99999), [], 'Resuming past the end should yield nothing')
		self.assertEqual(len(full), 16, 'Sanity check on range size failed')


	def test_negative_resume_rejected(self):
		'''A negative resume step must raise rather than silently misbehave'''

		with self.assertRaises(ValueError):
			stream('10.0.0.0/28', 1, 1, 5, resume_step=-1)


	def test_crash_and_resume_never_loses_ips(self):
		'''Repeatedly interrupting and resuming a full run must still cover the range exactly'''

		print_subheader('Testing Repeated Crash And Resume')
		start_time = time.perf_counter()
		random.seed(1)
		path = os.path.join(tempfile.gettempdir(), 'pylcg_test_crash_resume.state')

		for cidr, excludes, shards in [('172.16.0.0/20', ['172.16.4.0/22', '172.16.15.255'], 17),
		                               ('10.0.0.0/18',   ['10.0.1.0/24', '10.0.32.0/19'],    64),
		                               ('10.0.0.0/16',   None,                                7),
		                               ('10.0.0.0/19',   ['10.0.0.0/22', '10.0.16.0/21'],   256)]:
			expected  = IPRange(cidr, excludes).total
			collected = []
			crashes   = 0

			for shard in range(1, shards + 1):
				step = 0

				while True:
					# Run a random burst, then pretend the process died. Recording every
					# step means the state file is the only thing carried across the
					# "crash", exactly as a real interrupted run would resume.
					generator = ip_stream(cidr, shard, shards, 999, excludes, step, path, 1)
					chunk     = list(itertools.islice(generator, random.randint(1, 40)))
					generator.close()

					if not chunk:
						break

					collected += chunk
					crashes   += 1
					step       = StateManager.load(path)['step']

			# Coverage must be exact: every IP present, none missing
			self.assertEqual(len(set(collected)), expected, f'{cidr} left gaps across resumes: covered {len(set(collected))}, expected {expected}')

			# Each interruption re-scans at most the one IP that was in flight
			repeats = len(collected) - expected
			self.assertLessEqual(repeats, crashes, f'{cidr} repeated {repeats} IPs across {crashes} interruptions, expected at most one each')
			print_success(f'{cidr} with {shards} shards survived {crashes} interruptions: {expected:,} IPs, no gaps, {repeats} re-scanned')

		os.remove(path)

		elapsed = time.perf_counter() - start_time
		print_benchmark('Crash and resume verification', 4, elapsed)


class TestStateManagement(unittest.TestCase):
	'''Test state management functionality'''

	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing State Management')
		self.test_cidr = '192.168.0.0/16'
		self.test_seed = 12345


	def tearDown(self):
		'''Leave no state directory behind when the tests exercised the default location'''

		if os.path.isdir(STATE_DIR) and not os.listdir(STATE_DIR):
			os.rmdir(STATE_DIR)


	def test_state_round_trip(self):
		'''A written state file must load back with every field intact'''

		print_subheader('Testing State Round Trip')

		with StateManager(self.test_cidr, 2, 4, self.test_seed, 'deadbeefdeadbeef') as manager:
			manager.step = 4242
			manager.write()
			path = manager.state_file

		state = StateManager.load(path)
		self.assertEqual(state['seed'],     self.test_seed)
		self.assertEqual(state['cidr'],     self.test_cidr)
		self.assertEqual(state['shard'],    2)
		self.assertEqual(state['total'],    4)
		self.assertEqual(state['excludes'], 'deadbeefdeadbeef')
		self.assertEqual(state['step'],     4242)

		os.remove(path)
		print_success('State file round trips cleanly')


	def test_state_written_on_close(self):
		'''The final position must be flushed when the stream ends, not left at the last interval'''

		print_subheader('Testing Final Flush')
		path = os.path.join(tempfile.gettempdir(), 'pylcg_test_final_flush.state')

		ips = list(ip_stream('10.0.0.0/24', seed=7, state_file=path, state_interval=10_000))

		self.assertEqual(len(ips), 256, 'Sanity check on range size failed')
		self.assertEqual(StateManager.load(path)['step'], 256, 'Final step was not flushed on close')

		os.remove(path)
		print_success('Final position is flushed even though it lands between write intervals')


	def test_state_file_survives_interruption(self):
		'''An interrupted stream must leave a resumable state file'''

		print_subheader('Testing Interrupted Stream State')
		path = os.path.join(tempfile.gettempdir(), 'pylcg_test_interrupt.state')

		generator = ip_stream('10.0.0.0/22', seed=7, state_file=path, state_interval=1)
		first     = list(itertools.islice(generator, 500))
		generator.close()

		# The last delivered IP is only marked done once the consumer comes back for
		# the next one, so an interrupted run always rewinds by exactly one IP. That
		# is deliberate: a crash mid-processing must re-scan, never skip.
		step = StateManager.load(path)['step']
		self.assertEqual(step, 499, 'An interrupted run should rewind by exactly one IP')

		rest = list(ip_stream('10.0.0.0/22', seed=7, resume_step=step, write_state=False))
		self.assertEqual(len(set(first + rest)), 1024, 'Interrupted run plus resumed run left gaps')
		self.assertEqual(len(first) + len(rest), 1025, 'Interrupted run plus resumed run should repeat exactly one IP')

		os.remove(path)
		print_success('Interrupted stream resumes with no gaps, re-scanning exactly one IP')


	def test_batched_state_never_skips(self):
		'''Batched writes may repeat IPs on resume but must never skip any'''

		print_subheader('Testing Batched State Direction')
		path = os.path.join(tempfile.gettempdir(), 'pylcg_test_batched.state')

		generator = ip_stream('10.0.0.0/22', seed=11, state_file=path, state_interval=64)
		first     = list(itertools.islice(generator, 500))
		generator.close()

		step = StateManager.load(path)['step']
		self.assertLessEqual(step, 500, 'Batched state should lag behind, never lead')

		rest      = list(ip_stream('10.0.0.0/22', seed=11, resume_step=step, write_state=False))
		combined  = set(first) | set(rest)
		overlap   = len(first) + len(rest) - len(combined)

		self.assertEqual(len(combined), 1024, 'Batched resume lost IPs')
		self.assertLessEqual(overlap, 64, 'Batched resume repeated more than one interval of IPs')
		print_detail(f'State lagged to step {step}, {overlap} IPs re-emitted, 0 skipped')

		os.remove(path)
		print_success('Batched state re-emits at most one interval and never skips')


	def test_state_never_records_undelivered_output(self):
		'''The position must be flushed to the consumer before it is recorded, and must not advance if that flush fails'''

		print_subheader('Testing State Never Runs Ahead Of Delivery')
		path      = os.path.join(tempfile.gettempdir(), 'pylcg_test_delivery.state')
		delivered = []
		produced  = []

		def flush():
			'''Stand in for a consumer flushing its buffer, then dying part way through'''

			delivered.append(len(produced))

			# Pretend the consumer went away after the second flush
			if len(delivered) > 2:
				raise BrokenPipeError('consumer is gone')

		with self.assertRaises(BrokenPipeError):
			for ip in ip_stream('10.0.0.0/20', seed=3, state_file=path, state_interval=10, before_state_write=flush):
				produced.append(ip)

		recorded = StateManager.load(path)['step']

		# The recorded position must correspond to a flush that actually succeeded
		self.assertIn(recorded, delivered, f'Recorded step {recorded} was never confirmed as delivered ({delivered[-4:]})')
		self.assertLess(recorded, len(produced), 'Recorded step should lag the IPs generated after the consumer died')

		# Resuming from it must not skip anything the consumer had already been given
		rest = list(ip_stream('10.0.0.0/20', seed=3, resume_step=recorded, write_state=False))
		self.assertEqual(len(set(produced[:recorded]) | set(rest)), 4096, 'Resuming from the recorded position left a gap')

		os.remove(path)
		print_success(f'Recorded step {recorded} never ran ahead of a successful flush')


	def test_legacy_state_file_rejected(self):
		'''A pre-2.0 state file must be refused with a clear error, not misread'''

		print_subheader('Testing Legacy State Rejection')
		path = os.path.join(tempfile.gettempdir(), 'pylcg_test_legacy.state')

		with open(path, 'w') as f:
			f.write('987654321,5000')

		with self.assertRaises(ValueError):
			StateManager.load(path)

		with open(path, 'w') as f:
			f.write('987654321')

		with self.assertRaises(ValueError):
			StateManager.load(path)

		os.remove(path)
		print_success('Legacy state files are rejected instead of silently misinterpreted')


	def test_state_file_parent_directory_is_created(self):
		'''An explicit --state-file must behave like the default and create its directory'''

		print_subheader('Testing State File Directory Creation')
		root = os.path.join(tempfile.gettempdir(), 'pylcg_test_newdir')
		path = os.path.join(root, 'nested', 'shard1.state')

		if os.path.isdir(root):
			shutil.rmtree(root)

		try:
			ips = list(ip_stream('10.0.0.0/24', seed=12345, state_file=path))

			self.assertEqual(len(ips), 256, 'Sanity check on range size failed')
			self.assertTrue(os.path.isfile(path), 'State file was not created in a directory that did not exist')
			self.assertEqual(StateManager.load(path)['step'], 256, 'State was not written to the explicit path')
		finally:
			shutil.rmtree(root, ignore_errors=True)

		print_success('An explicit state file path creates its parent directories')


	def test_state_file_missing_a_field_is_rejected(self):
		'''A state file missing any field must be refused rather than half loaded'''

		path = os.path.join(tempfile.gettempdir(), 'pylcg_test_missing_field.state')

		with open(path, 'w') as f:
			f.write('pylcg seed=1 cidr=10.0.0.0/24 shard=1 total=1 step=5')

		with self.assertRaises(ValueError) as caught:
			StateManager.load(path)

		self.assertIn('excludes', str(caught.exception), 'The error should name the missing field')

		os.remove(path)


	def test_state_can_be_disabled(self):
		'''Disabling state must skip file creation entirely'''

		with StateManager('10.0.0.0/24', 1, 1, 1, 'abc', enabled=False) as manager:
			self.assertIsNone(manager.state_file, 'No state file should be created when disabled')
			self.assertIsNone(manager.handle, 'No file handle should be opened when disabled')
			manager.step = 10
			manager.write()


	def test_state_filename_includes_exclusions(self):
		'''Two runs differing only by exclude list must not share a state file'''

		with StateManager('10.0.0.0/24', 1, 1, 1, 'aaaaaaaaaaaaaaaa') as a:
			with StateManager('10.0.0.0/24', 1, 1, 1, 'bbbbbbbbbbbbbbbb') as b:
				self.assertNotEqual(a.state_file, b.state_file, 'State files must not collide across different exclude lists')
				paths = (a.state_file, b.state_file)

		for path in paths:
			os.remove(path)


class TestExclusionClipping(unittest.TestCase):
	'''Test that exclusion ranges are correctly clipped to target CIDR bounds'''

	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing Exclusion Clipping')


	def test_exclusion_larger_than_target(self):
		'''Exclusion that fully contains the target range'''

		self.assertEqual(IPRange('10.0.0.0/24', ['10.0.0.0/16']).total, 0, 'Exclusion larger than target should leave 0 IPs')


	def test_exclusion_partially_overlapping(self):
		'''Exclusion that partially overlaps the target'''

		self.assertEqual(IPRange('10.0.1.0/24', ['10.0.0.0/23']).total, 0, 'Exclusion covering all of target should leave 0 IPs')


	def test_exclusion_entirely_outside_target(self):
		'''Exclusion that has no overlap with target'''

		self.assertEqual(IPRange('192.168.1.0/24', ['10.0.0.0/8']).total, 256, 'Non-overlapping exclusion should not reduce total')


	def test_exclusion_spanning_target_boundary(self):
		'''Exclusion that starts before and ends within the target'''

		self.assertEqual(IPRange('10.0.0.128/25', ['10.0.0.0/24']).total, 0)


	def test_mixed_inside_and_outside_exclusions(self):
		'''Mix of exclusions inside, outside, and overlapping the target'''

		ip_range = IPRange('10.0.0.0/24', ['9.0.0.0/8', '10.0.0.50/32', '10.0.0.100/31', '11.0.0.0/8'])
		self.assertEqual(ip_range.total, 253)

		generated = set()

		for i in range(ip_range.total):
			ip = ip_range.get_ip_at_index(i)
			self.assertIn(ipaddress.ip_address(ip), ipaddress.ip_network('10.0.0.0/24'))
			self.assertNotIn(ip, generated, f'Duplicate: {ip}')
			generated.add(ip)

		for excluded_ip in ['10.0.0.50', '10.0.0.100', '10.0.0.101']:
			self.assertNotIn(excluded_ip, generated, f'Excluded IP appeared: {excluded_ip}')


	def test_all_excluded_yields_nothing(self):
		'''ip_stream yields nothing when all IPs are excluded'''

		self.assertEqual(stream('10.0.0.0/24', 1, 1, 12345, ['10.0.0.0/16']), [])


class TestInputValidation(unittest.TestCase):
	'''Test that invalid inputs raise clear errors'''

	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing Input Validation')


	def test_invalid_cidr(self):
		'''Invalid CIDR string raises ValueError'''

		with self.assertRaises(ValueError):
			IPRange('not_a_cidr')

		with self.assertRaises(ValueError):
			stream('not_a_cidr')


	def test_host_bits_set(self):
		'''CIDR with host bits set raises ValueError (strict=True)'''

		with self.assertRaises(ValueError):
			IPRange('192.168.1.1/24')


	def test_total_shards_zero(self):
		'''total_shards=0 raises ValueError'''

		with self.assertRaises(ValueError):
			stream('10.0.0.0/24', 1, 0, 1)


	def test_shard_num_out_of_range(self):
		'''shard_num outside [1, total_shards] raises ValueError'''

		with self.assertRaises(ValueError):
			stream('10.0.0.0/24', 0, 1, 1)

		with self.assertRaises(ValueError):
			stream('10.0.0.0/24', 5, 4, 1)


	def test_state_interval_must_be_positive(self):
		'''A state interval below 1 raises ValueError'''

		with self.assertRaises(ValueError):
			list(ip_stream('10.0.0.0/24', seed=1, state_interval=0, write_state=False))


	def test_invalid_exclude(self):
		'''Invalid exclusion entries raise ValueError'''

		with self.assertRaises(ValueError):
			IPRange('10.0.0.0/24', ['not_an_ip'])

		with self.assertRaises(ValueError):
			IPRange('10.0.0.0/24', ['256.0.0.0'])


class TestSeedZero(unittest.TestCase):
	'''Test that seed=0 is a valid deterministic seed'''

	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing Seed Zero')


	def test_seed_zero_deterministic(self):
		'''seed=0 produces identical sequences across runs'''

		ips_a = stream('10.0.0.0/28', 1, 1, 0)
		ips_b = stream('10.0.0.0/28', 1, 1, 0)

		self.assertEqual(ips_a, ips_b, 'seed=0 should be deterministic')
		self.assertEqual(len(ips_a), 16)


	def test_seed_zero_differs_from_seed_one(self):
		'''seed=0 and seed=1 produce different sequences'''

		self.assertNotEqual(stream('10.0.0.0/28', 1, 1, 0), stream('10.0.0.0/28', 1, 1, 1))


class TestEdgeCaseCIDR(unittest.TestCase):
	'''Test edge case CIDR sizes'''

	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing Edge Case CIDR Sizes')


	def test_single_ip(self):
		'''/32 range returns exactly one IP'''

		self.assertEqual(stream('192.168.1.1/32', 1, 1, 0), ['192.168.1.1'])


	def test_two_ips(self):
		'''/31 range returns exactly two IPs'''

		ips = stream('192.168.1.0/31', 1, 1, 0)
		self.assertEqual(len(ips), 2)
		self.assertEqual(set(ips), {'192.168.1.0', '192.168.1.1'})


	def test_small_range_complete_coverage(self):
		'''/28 range (16 IPs) has complete coverage with no duplicates'''

		ips = stream('172.16.0.0/28', 1, 1, 42)
		self.assertEqual(len(ips), 16)
		self.assertEqual(len(set(ips)), 16)

		for ip in ips:
			self.assertIn(ipaddress.ip_address(ip), ipaddress.ip_network('172.16.0.0/28'))


	def test_non_power_of_two_valid_count(self):
		'''A range whose valid count is not a power of 2 still covers exactly once'''

		# 256 addresses minus 3 exclusions leaves 253, so the modulus is 256 and three
		# steps map to no IP at all
		ips = stream('10.0.0.0/24', 1, 1, 3, ['10.0.0.7', '10.0.0.40/31'])
		self.assertEqual(len(ips), 253)
		self.assertEqual(len(set(ips)), 253)


class TestParseExcludesVariants(unittest.TestCase):
	'''Test parse_excludes with different input types'''

	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing Parse Excludes Variants')


	def test_private_keyword(self):
		'''The "private" keyword loads all RFC reserved ranges'''

		from pylcg.exclude import RFC_RANGES

		ranges = parse_excludes('private')
		self.assertGreater(len(ranges), 0)
		self.assertEqual(len(ranges), len(RFC_RANGES['private']))


	def test_comma_separated_string(self):
		'''Comma-separated string is parsed correctly'''

		ranges = parse_excludes('10.0.0.1,10.0.0.2,10.0.0.3')
		self.assertEqual(len(ranges), 3)
		self.assertEqual(sum(end - start + 1 for start, end in ranges), 3)


	def test_file_input(self):
		'''File path input is parsed correctly'''

		test_file = os.path.join(tempfile.gettempdir(), 'pylcg_test_excludes.txt')

		with open(test_file, 'w') as f:
			f.write('# comment line\n10.0.0.0/24\n192.168.1.1\n')

		ranges = parse_excludes(test_file)
		self.assertEqual(sum(end - start + 1 for start, end in ranges), 257)

		os.remove(test_file)


	def test_file_path_inside_a_list(self):
		'''A file path given as one entry of a list is expanded like a bare path'''

		print_subheader('Testing File Path As A List Entry')
		test_file = os.path.join(tempfile.gettempdir(), 'pylcg_test_nested.txt')

		# 9.0.0.0/24 sits outside every reserved range, so it must add to the total
		with open(test_file, 'w') as f:
			f.write('9.0.0.0/24\n')

		# File, keyword, blank entry and a literal address all mixed together
		ranges  = parse_excludes([test_file, '', 'private', '8.8.8.8'])
		total   = sum(end - start + 1 for start, end in optimize_ranges(ranges))
		private = sum(end - start + 1 for start, end in optimize_ranges(parse_excludes('private')))

		self.assertEqual(total, private + 256 + 1, 'Nested file, keyword and address should all be expanded')

		os.remove(test_file)
		print_success('A file path, a keyword, a blank and an address all expand inside one list')


	def test_blank_entries_are_skipped(self):
		'''Empty strings in an exclude list are ignored rather than raising'''

		self.assertEqual(parse_excludes(['', '  ', '10.0.0.1']), {(int(ipaddress.ip_address('10.0.0.1')),) * 2})


	def test_random_seed_when_none_given(self):
		'''Omitting the seed still produces a complete, valid range'''

		ips = list(ip_stream('10.0.0.0/28', write_state=False))

		self.assertEqual(len(ips), 16)
		self.assertEqual(len(set(ips)), 16)


class TestHotLoop(unittest.TestCase):
	'''ip_stream inlines the body of get_ip_at_index for speed, so the two copies must never drift apart'''

	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing Hot Loop Against get_ip_at_index')


	def test_inlined_loop_matches_get_ip_at_index(self):
		'''Every IP the loop emits must equal what get_ip_at_index would have returned for that index'''

		print_subheader('Testing Inlined Translation')
		checked = 0

		for cidr, excludes in [('10.0.0.0/22', None),
		                       ('10.0.0.0/22', ['10.0.1.0/24', '10.0.2.5', '10.0.3.255']),
		                       ('10.0.0.0/24', ['10.0.0.0/28', '10.0.0.100/30']),
		                       ('172.16.0.0/20', ['172.16.4.0/22', '172.16.15.255']),
		                       ('2001:db8::/122', None),
		                       ('2001:db8::/122', ['2001:db8::/124'])]:
			ip_range = IPRange(cidr, excludes)

			# What the loop actually produces, in order
			produced = stream(cidr, 1, 1, 7, excludes)

			# What get_ip_at_index says the same run should produce, derived independently
			expected = [ip_range.get_ip_at_index(i) for i in range(ip_range.total)]

			self.assertEqual(sorted(produced), sorted(expected), f'Inlined loop disagreed with get_ip_at_index for {cidr}')
			self.assertEqual(len(produced), ip_range.total, f'Inlined loop emitted the wrong count for {cidr}')
			checked += len(produced)

		print_success(f'{checked:,} IPs from the inlined loop match get_ip_at_index exactly, IPv4 and IPv6')


	def test_inlined_loop_matches_across_shards_and_resume(self):
		'''The inlined translation must hold under sharding and resuming too'''

		cidr, excludes = '10.0.0.0/22', ['10.0.1.0/24', '10.0.2.5']
		ip_range       = IPRange(cidr, excludes)
		expected       = {ip_range.get_ip_at_index(i) for i in range(ip_range.total)}

		for shards in (1, 3, 4, 7, 16):
			collected = {ip for shard in range(1, shards + 1) for ip in stream(cidr, shard, shards, 7, excludes)}
			self.assertEqual(collected, expected, f'Inlined loop diverged with {shards} shards')

		for resume in (0, 1, 5, 50, 300):
			self.assertTrue(set(stream(cidr, 1, 4, 7, excludes, resume)) <= expected, f'Inlined loop produced an IP outside the range when resuming at {resume}')


class TestCLI(unittest.TestCase):
	'''Test that the command line tool emits nothing but IPs on stdout'''

	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing CLI Output Stream')


	def test_stdout_contains_only_ips(self):
		'''Anything piping pylcg into a scanner must never receive a non-address line'''

		print_subheader('Testing Clean Stdout')

		for args in (['10.0.0.0/29', '--no-state'], ['10.0.0.0/29', '--no-state', '--total-shards', '2', '--shard-num', '1']):
			result = subprocess.run([sys.executable, '-m', 'pylcg.cli'] + args, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))

			self.assertEqual(result.returncode, 0, f'pylcg {" ".join(args)} failed: {result.stderr}')

			for line in result.stdout.splitlines():
				ipaddress.ip_address(line)

			# The seed notice still has to reach the user, just not on stdout
			self.assertIn('Using random seed', result.stderr, 'The random seed should be reported on stderr')

		print_success('stdout carries only parseable IP addresses, seed notice goes to stderr')


	def test_errors_are_one_clean_line(self):
		'''A user mistake must produce a single readable message, never a traceback'''

		print_subheader('Testing Clean Error Messages')
		cases = [
			['10.0.0.0/24', '--total-shards', '0'],
			['10.0.0.0/24', '--shard-num', '5', '--total-shards', '4'],
			['10.0.0.0/24', '--shard-num', '0'],
			['10.0.0.0/24', '--state-interval', '0'],
			['not-a-cidr', '--no-state'],
			['192.168.1.1/24', '--no-state'],
			['10.0.0.0/24', '--exclude', 'bogus', '--no-state'],
			['10.0.0.0/24', '--state', '500', '--no-state'],
			['10.0.0.0/24', '--state', '/nonexistent/missing.state', '--no-state']
		]

		for args in cases:
			result = subprocess.run([sys.executable, '-m', 'pylcg.cli'] + args, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))

			self.assertEqual(result.returncode, 2, f'{" ".join(args)} should exit 2, got {result.returncode}')
			self.assertNotIn('Traceback', result.stderr, f'{" ".join(args)} produced a traceback')
			self.assertEqual(result.stdout, '', f'{" ".join(args)} wrote to stdout on failure')
			# argparse prints its usage block first, so check the final line carries the message
			self.assertTrue(result.stderr.strip().splitlines()[-1].startswith('pylcg'), f'{" ".join(args)} message should be prefixed: {result.stderr!r}')

		print_success(f'{len(cases)} user errors each give one prefixed line, exit 2, nothing on stdout')


	def test_unwritable_state_file_is_reported_cleanly(self):
		'''An OS level failure must also avoid dumping a traceback'''

		directory = tempfile.mkdtemp()
		os.chmod(directory, 0o500)

		try:
			result = subprocess.run([sys.executable, '-m', 'pylcg.cli', '10.0.0.0/24', '--seed', '1', '--state-file', os.path.join(directory, 'x.state')],
			                        capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))

			self.assertEqual(result.returncode, 1, 'An unwritable state file should exit 1')
			self.assertNotIn('Traceback', result.stderr, 'An unwritable state file produced a traceback')
		finally:
			os.chmod(directory, 0o700)
			os.rmdir(directory)


	def test_broken_pipe_exits_zero(self):
		'''Piping into something that stops reading is normal, not an error'''

		result = subprocess.run(f'{sys.executable} -m pylcg.cli 0.0.0.0/0 --seed 7 --no-state 2>/dev/null | head -3',
		                        shell=True, capture_output=True, text=True, executable='/bin/bash', cwd=os.path.dirname(os.path.abspath(__file__)))

		self.assertEqual(len(result.stdout.splitlines()), 3, 'head should have received exactly three IPs')

		for line in result.stdout.splitlines():
			ipaddress.ip_address(line)


class TestPerformance(unittest.TestCase):
	'''Guard the properties that make large ranges practical'''

	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing Performance Characteristics')


	def best_rate(self, work, count: int, rounds: int = 5) -> float:
		'''
		Best-of-N rate for a piece of work, so a stray GC pause cannot fail the run

		:param work: Callable performing the work once
		:param count: Number of operations the callable performs
		:param rounds: How many times to repeat before taking the best
		'''

		best = 0

		for _ in range(rounds):
			gc.collect()
			gc.disable()

			start = time.perf_counter()
			work()
			elapsed = time.perf_counter() - start

			gc.enable()

			best = max(best, count / elapsed)

		return best


	def test_no_wasted_steps_at_any_shard_count(self):
		'''A shard must advance the generator once per IP it emits, no matter how many shards there are

		This is the property that keeps throughput flat, asserted by counting steps rather
		than by timing, so a busy machine cannot make it flake. The old design walked the
		whole sequence in every shard and discarded the other shards' work, which on a /0
		split 1024 ways meant roughly 1024 steps per useful IP.
		'''

		print_subheader('Testing Steps Per Emitted IP')
		path   = os.path.join(tempfile.gettempdir(), 'pylcg_test_steps.state')
		wanted = 5_000

		for shards in (1, 64, 1024, 4096):
			# 0.0.0.0/0 has no rejection slack, so every step must produce exactly one IP
			generator = ip_stream('0.0.0.0/0', 1, shards, 1, None, 0, path, 1)
			produced  = len(list(itertools.islice(generator, wanted)))
			generator.close()

			# The last delivered IP is only marked done once the consumer asks for the
			# next one, so an interrupted count always trails by exactly one
			steps = StateManager.load(path)['step']
			print_detail(f'{shards:>5} shards: {steps:,} steps for {produced:,} IPs')

			self.assertEqual(produced, wanted, f'Expected {wanted} IPs from shard 1 of {shards}')
			self.assertEqual(steps, wanted - 1, f'Shard 1 of {shards} took {steps} steps to emit {wanted} IPs, expected {wanted - 1}')

		os.remove(path)
		print_success('Exactly one step per emitted IP from 1 shard to 4096, no wasted work')


	def test_deep_resume_is_instant(self):
		'''Resuming billions of steps into a range must take milliseconds'''

		print_subheader('Testing Deep Resume Speed')
		start = time.perf_counter()
		first = next(ip_stream('0.0.0.0/0', 1, 1, 424242, resume_step=3_000_000_000, write_state=False))
		taken = time.perf_counter() - start

		print_detail(f'Resumed 3,000,000,000 steps into a /0 -> {first} in {taken*1000:.1f} ms')
		self.assertLess(taken, 1.0, 'Deep resume should be effectively instant')
		print_success('Deep resume is instant')


	def test_large_exclude_list_stays_fast(self):
		'''Index lookups must not degrade linearly with the exclusion count'''

		print_subheader('Testing Large Exclude List')
		rates = {}

		for count in (16, 20_000):
			step     = (2**32) // (count + 1)
			excludes = [f'{(x >> 24) & 255}.{(x >> 16) & 255}.{(x >> 8) & 255}.{x & 255}/32' for x in (i * step for i in range(count))]
			ip_range = IPRange('0.0.0.0/0', excludes)
			indexes  = [random.randrange(ip_range.total) for _ in range(20_000)]

			def lookups(ip_range=ip_range, indexes=indexes):
				'''Perform every lookup once'''

				for index in indexes:
					ip_range.get_ip_at_index(index)

			rates[count] = self.best_rate(lookups, len(indexes))
			print_detail(f'{count:>6} exclusions: {rates[count]:>12,.0f} lookups/s')

		# The linear walk fell ~436x over this same jump while the binary search only
		# pays a few extra comparisons, so a loose bound still catches a regression by
		# a wide margin without flaking when the machine is busy
		self.assertGreater(rates[20_000], rates[16] * 0.2, f'Lookup rate collapsed with a large exclude list: {rates}')
		print_success('Lookup speed holds up with 20,000 exclusion ranges')



if __name__ == '__main__':
	print(f'\n{Colors.CYAN}{Colors.BOLD}{'='*80}')
	print('PyLCG Comprehensive Test Suite')
	print(f'{'='*80}')
	print(f'{Colors.PURPLE}Test Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}')
	print(f'Python Version: {sys.version.split()[0]}')
	print(f'{'='*80}{Colors.ENDC}\n')
	unittest.main(verbosity=2)
