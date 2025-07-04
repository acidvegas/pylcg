#!/usr/bin/env python
# PyLCG - Linear Congruential Generator for IP Sharding - Developed by acidvegas in Python (https://github.com/acidvegas/pylcg)
# unit_test.py

import ipaddress
import os
import sys
import tempfile
import time
import unittest

from pylcg         import IPRange, ip_stream, LCG
from pylcg.exclude import parse_excludes, optimize_ranges
from pylcg.state   import save_state, StateManager


class Colors:
	BLUE      = '\033[94m'
	GREEN     = '\033[92m'
	YELLOW    = '\033[93m'
	CYAN      = '\033[96m'
	RED       = '\033[91m'
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


def print_warning(message: str):
	'''
	Print a warning message for the test suite
	
	:param message: The message to print
	'''

	print(f'{Colors.YELLOW}⚠ WARNING: {message}{Colors.ENDC}')


def print_error(message: str):
	'''
	Print an error message for the test suite
	
	:param message: The message to print
	'''

	print(f'{Colors.RED}✗ ERROR: {message}{Colors.ENDC}')


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
		custom_modulus = 1000
		lcg = LCG(self.test_seed, custom_modulus)
		self.assertEqual(lcg.m, custom_modulus, 'Custom modulus not set correctly')
		
		# Test bounds
		for _ in range(1000):
			val = lcg.next()
			self.assertTrue(0 <= val < lcg.m, f'Generated value {val} outside bounds [0, {lcg.m})')
		
		elapsed = time.perf_counter() - start_time
		print_benchmark('LCG initialization tests', 1000, elapsed)
		

	def test_lcg_sequence_properties(self):
		'''Test LCG sequence properties'''
		print_subheader('Testing LCG Sequence Properties')
		start_time = time.perf_counter()
		iterations = 1_000_000

		print_info('Testing deterministic behavior with identical seeds')
		print_detail(f'Creating two LCGs with seed {self.test_seed}')
		lcg1 = LCG(self.test_seed)
		lcg2 = LCG(self.test_seed)
		
		print_detail(f'Generating {iterations:,} numbers for each sequence')
		sequence1 = [lcg1.next() for _ in range(iterations)]
		sequence2 = [lcg2.next() for _ in range(iterations)]
		self.assertEqual(sequence1, sequence2, 'LCG sequences not deterministic')
		print_success('Sequences are identical as expected')

		print_info('Testing different seeds produce different sequences')
		print_detail(f'Creating new LCG with seed {self.test_seed + 1}')
		lcg3 = LCG(self.test_seed + 1)
		sequence3 = [lcg3.next() for _ in range(100)]
		self.assertNotEqual(sequence1[:100], sequence3, 'Different seeds produced identical sequences')
		print_success('Different seeds produced different sequences')

		print_info('Testing sequence distribution')
		values = sequence1[:10000]
		unique_values = len(set(values))
		distribution_ratio = unique_values / len(values)
		print_detail(f'Analyzed {len(values):,} values')
		print_detail(f'Found {unique_values:,} unique values')
		print_detail(f'Distribution ratio: {distribution_ratio:.2%}')
		
		self.assertGreater(distribution_ratio, 0.95, 'Poor distribution of values')
		print_success(f'Sequence shows good distribution ({distribution_ratio:.2%} unique values)')

		elapsed = time.perf_counter() - start_time
		print_benchmark('LCG sequence tests', iterations, elapsed)


class TestExclusionFunctionality(unittest.TestCase):
	'''Test IP exclusion functionality'''
	
	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing Exclusion Functionality')
		
	def test_parse_excludes(self):
		'''Test parsing of exclusion lists'''

		start_time = time.perf_counter()
		test_cases = [
			# Single IP
			{
				'input': ['192.168.0.1'],
				'expected_count': 1
			},
			# CIDR range
			{
				'input': ['192.168.0.0/16'],
				'expected_count': 65536
			},
			# Mixed input
			{
				'input': ['192.168.0.1', '10.0.0.0/16', '172.16.0.1'],
				'expected_count': 65538
			}
		]
		
		for case in test_cases:
			ranges = parse_excludes(case['input'])
			excluded_count = sum(end - start + 1 for start, end in ranges)
			self.assertEqual(excluded_count, case['expected_count'],
						   f'Wrong exclusion count for {case['input']}')
			
		# Test invalid inputs
		invalid_inputs = ['256.256.256.256', 'not_an_ip', '192.168.0.0/33']
		for invalid in invalid_inputs:
			with self.assertRaises(ValueError):
				parse_excludes([invalid])
				
		elapsed = time.perf_counter() - start_time
		print_benchmark('Exclusion parsing tests', len(test_cases) + len(invalid_inputs), elapsed)
		

	def test_optimize_ranges(self):
		'''Test IP range optimization'''
		start_time = time.perf_counter()
		test_cases = [
			# Non-overlapping IP ranges
			{
				'input': {
					(int(ipaddress.ip_address('192.168.1.0')), int(ipaddress.ip_address('192.168.1.255'))),
					(int(ipaddress.ip_address('192.168.3.0')), int(ipaddress.ip_address('192.168.3.255')))
				},
				'expected': [
					(int(ipaddress.ip_address('192.168.1.0')), int(ipaddress.ip_address('192.168.1.255'))),
					(int(ipaddress.ip_address('192.168.3.0')), int(ipaddress.ip_address('192.168.3.255')))
				]
			},
			# Overlapping IP ranges
			{
				'input': {
					(int(ipaddress.ip_address('192.168.1.0')), int(ipaddress.ip_address('192.168.2.0'))),
					(int(ipaddress.ip_address('192.168.1.128')), int(ipaddress.ip_address('192.168.2.128')))
				},
				'expected': [
					(int(ipaddress.ip_address('192.168.1.0')), int(ipaddress.ip_address('192.168.2.128')))
				]
			},
			# Adjacent IP ranges
			{
				'input': {
					(int(ipaddress.ip_address('192.168.1.0')), int(ipaddress.ip_address('192.168.1.255'))),
					(int(ipaddress.ip_address('192.168.2.0')), int(ipaddress.ip_address('192.168.2.255')))
				},
				'expected': [
					(int(ipaddress.ip_address('192.168.1.0')), int(ipaddress.ip_address('192.168.2.255')))
				]
			}
		]
		
		print_info('Testing IP range optimization')
		for i, case in enumerate(test_cases, 1):
			print_detail(f'Test case {i}:')
			for start, end in case['input']:
				print_detail(f'  Input range: {ipaddress.ip_address(start)} - {ipaddress.ip_address(end)}')
			
			result = optimize_ranges(case['input'])
			
			print_detail('  Result:')
			for start, end in result:
				print_detail(f'    {ipaddress.ip_address(start)} - {ipaddress.ip_address(end)}')
			
			self.assertEqual(result, case['expected'], 
							'Range optimization produced incorrect result')
			print_success(f'Test case {i} passed')
		
		elapsed = time.perf_counter() - start_time
		print_benchmark('IP range optimization tests', len(test_cases), elapsed)
		
		
	def test_complex_exclusions(self):
		'''Test complex exclusion scenarios with mixed CIDR and single IP excludes'''

		start_time = time.perf_counter()
		
		# Test case with a /16 range and various excludes
		test_range = '10.0.0.0/16'
		excludes = [
			'10.0.0.0/24',      # Excludes first 256 IPs
			'10.0.255.0/24',    # Excludes last 256 IPs
			'10.0.1.1',         # Single IP
			'10.0.1.2',         # Single IP
			'10.0.128.0/25',    # Excludes 128 IPs in the middle
		]
		
		ip_range = IPRange(test_range, excludes)
		
		# Calculate expected total
		expected_total = 65536 - 256 - 256 - 2 - 128
		self.assertEqual(ip_range.total, expected_total, f'Expected {expected_total} IPs after exclusions, got {ip_range.total}')
		
		# Generate all IPs and verify exclusions
		generated_ips = set()
		excluded_networks = [ipaddress.ip_network(ex) if '/' in ex else ipaddress.ip_network(ex + '/32') 
						   for ex in excludes]
		
		for i in range(ip_range.total):
			ip = ip_range.get_ip_at_index(i)
			ip_obj = ipaddress.ip_address(ip)
			
			# Verify IP is not in excluded ranges
			for excluded in excluded_networks:
				self.assertNotIn(ip_obj, excluded, f'Generated excluded IP: {ip}')
			
			# Check for duplicates
			self.assertNotIn(ip, generated_ips, f'Duplicate IP generated: {ip}')
			generated_ips.add(ip)

		# Verify we generated exactly the expected number of IPs
		self.assertEqual(len(generated_ips), expected_total, 'Generated IP count doesn\'t match expected total')
		
		elapsed = time.perf_counter() - start_time
		print_benchmark('Complex exclusion tests', len(generated_ips), elapsed)


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
			# Basic initialization
			('192.168.0.0/16', None, 65536),
			# Single IP exclusion
			('192.168.0.0/16', ['192.168.0.1'], 65535),
			# CIDR exclusion
			('192.168.0.0/16', ['192.168.0.0/17'], 32768),
			# Multiple exclusions
			('192.168.0.0/16', ['192.168.0.1', '192.168.0.2'], 65534),
			# Large range
			('10.0.0.0/16', None, 65536)
		]
		
		for cidr, excludes, expected_total in test_cases:
			ip_range = IPRange(cidr, excludes)
			self.assertEqual(ip_range.total, expected_total,
						   f'Wrong total for {cidr} with excludes {excludes}')
			
		# Test invalid CIDR
		with self.assertRaises(ValueError):
			IPRange('invalid_cidr')
			
		elapsed = time.perf_counter() - start_time
		print_benchmark('IP range initialization tests', len(test_cases) + 1, elapsed)
		

	def test_ip_generation(self):
		'''Test IP generation functionality'''

		start_time = time.perf_counter()
		ip_range = IPRange(self.test_cidr)
		
		# Test sequential generation
		for i in range(256):
			ip = ip_range.get_ip_at_index(i)
			self.assertTrue(ipaddress.ip_address(ip) in ipaddress.ip_network(self.test_cidr))
			
		# Test bounds
		with self.assertRaises(IndexError):
			ip_range.get_ip_at_index(-1)
		with self.assertRaises(IndexError):
			ip_range.get_ip_at_index(256)
			
		elapsed = time.perf_counter() - start_time
		print_benchmark('IP generation tests', 256, elapsed)


class TestSharding(unittest.TestCase):
	'''Test IP sharding functionality'''
	
	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing Sharding Implementation')
		self.test_cidr = '192.168.0.0/16'
		self.test_seed = 12345
		self.total_shards = 4
		self.excludes = [
			'192.168.0.0/24',    # Exclude first 256 IPs
			'192.168.1.1',       # Single IP
			'192.168.1.2',       # Single IP
			'192.168.255.0/24'   # Exclude last 256 IPs
		]
	

	def test_shard_distribution(self):
		'''Test shard size distribution'''

		print_subheader('Testing IP Shard Distribution')
		start_time = time.perf_counter()
		
		print_info('Generating shards for CIDR range: ' + self.test_cidr)
		print_detail(f'Total Shards: {self.total_shards}')
		print_detail(f'Using seed: {self.test_seed}')
		
		# Generate IPs for each shard
		shard_contents = {}
		all_ips = set()
		
		# Calculate expected total IPs
		network = ipaddress.ip_network(self.test_cidr)
		expected_total = int(network.num_addresses)
		print_detail(f'Expected total IPs: {expected_total:,}')
		
		for shard in range(1, self.total_shards + 1):
			print_info(f'Generating Shard {shard}/{self.total_shards}')
			shard_ips = set()
			for ip in ip_stream(self.test_cidr, shard, self.total_shards, self.test_seed):
				# Verify no duplicates
				self.assertNotIn(ip, shard_ips, f'Duplicate IP in shard {shard}: {ip}')
				self.assertNotIn(ip, all_ips, f'IP {ip} appears in multiple shards')
				
				shard_ips.add(ip)
				all_ips.add(ip)
				
			shard_contents[shard] = shard_ips
			print_success(f'Shard {shard} generated: {len(shard_ips):,} IPs')
		
		# Verify total IPs
		print_info('Verifying shard distribution')
		print_detail(f'Total IPs generated: {len(all_ips):,}')
		self.assertEqual(len(all_ips), expected_total, 'Missing or extra IPs')
		
		# Verify shard sizes are balanced
		shard_sizes = [len(ips) for ips in shard_contents.values()]
		size_difference = max(shard_sizes) - min(shard_sizes)
		print_detail(f'Shard sizes: {shard_sizes}')
		print_detail(f'Size difference between largest and smallest shard: {size_difference}')
		self.assertLessEqual(size_difference, 1, 'Shards are not balanced')
		print_success('Shards are evenly balanced')
		
		elapsed = time.perf_counter() - start_time
		print_benchmark('Shard distribution tests', len(all_ips), elapsed)
		

	def test_shard_determinism(self):
		'''Test shard generation determinism'''

		start_time = time.perf_counter()
		
		# Generate same shard twice
		shard_num = 1
		ips_first_run = list(ip_stream(self.test_cidr, shard_num, self.total_shards, self.test_seed))
		ips_second_run = list(ip_stream(self.test_cidr, shard_num, self.total_shards, self.test_seed))
		
		self.assertEqual(ips_first_run, ips_second_run, 'Shard generation not deterministic')
		
		# Different seeds should produce different sequences
		ips_different_seed = list(ip_stream(self.test_cidr, shard_num, self.total_shards, self.test_seed + 1))
		self.assertNotEqual(ips_first_run, ips_different_seed, 'Different seeds produced same sequence')
		
		elapsed = time.perf_counter() - start_time
		print_benchmark('Shard determinism tests', len(ips_first_run) * 3, elapsed)
		

	def test_sharding_with_exclusions(self):
		'''Test sharding with exclusions to verify complete coverage'''

		start_time = time.perf_counter()
		
		# Calculate expected total IPs after exclusions
		ip_range = IPRange(self.test_cidr, self.excludes)
		expected_total = ip_range.total
		
		# Generate IPs for each shard
		shard_contents = {}
		all_ips = set()
		excluded_networks = [ipaddress.ip_network(ex) if '/' in ex else ipaddress.ip_network(ex + '/32') 
						   for ex in self.excludes]
		
		for shard in range(1, self.total_shards + 1):
			shard_ips = set()
			for ip in ip_stream(self.test_cidr, shard, self.total_shards, 
								self.test_seed, None, self.excludes):
				# Verify no duplicates
				self.assertNotIn(ip, shard_ips, f'Duplicate IP in shard {shard}: {ip}')
				self.assertNotIn(ip, all_ips, f'IP {ip} appears in multiple shards')
				
				# Verify IP is not in excluded ranges
				ip_obj = ipaddress.ip_address(ip)
				for excluded in excluded_networks:
					self.assertNotIn(ip_obj, excluded, f'Generated excluded IP: {ip}')
				
				shard_ips.add(ip)
				all_ips.add(ip)
			
			shard_contents[shard] = shard_ips
			print_success(f'Shard {shard}: {len(shard_ips):,} IPs')
		
		# Verify total IPs
		self.assertEqual(len(all_ips), expected_total, 
						f'Expected {expected_total} IPs, got {len(all_ips)}')
		
		# Verify shard sizes are balanced
		shard_sizes = [len(ips) for ips in shard_contents.values()]
		size_difference = max(shard_sizes) - min(shard_sizes)
		self.assertLessEqual(size_difference, 1, 
							f'Shards are not balanced. Sizes: {shard_sizes}')
		
		elapsed = time.perf_counter() - start_time
		print_benchmark('Sharding with exclusions tests', len(all_ips), elapsed)


class TestStateManagement(unittest.TestCase):
	'''Test state management functionality'''
	
	def setUp(self):
		'''Set up the test suite'''

		print_header('Testing State Management')
		self.test_cidr = '192.168.0.0/16'
		self.test_seed = 12345
		self.excludes = ['192.168.0.0/24', '192.168.1.1']
	
	def test_state_manager(self):
		'''Test StateManager functionality'''
		
		print_subheader('Testing StateManager')
		start_time = time.perf_counter()
		
		# Test context manager and file handling
		with StateManager(self.test_seed, self.test_cidr, 1, 1) as manager:
			# Test initial state write
			manager.update(12345)
			
			# Test file exists and is writable
			self.assertTrue(os.path.exists(manager.state_file))
			
			# Test multiple rapid updates
			for state in range(1000):
				manager.update(state)
			
			# Verify final state
			with open(manager.state_file, 'r') as f:
				final_state = int(f.read().strip())
			self.assertEqual(final_state, 999)
		
		# Test file handle is properly closed
		self.assertFalse(manager.handle.closed)
		
		elapsed = time.perf_counter() - start_time
		print_benchmark('StateManager rapid update tests', 1000, elapsed)
	
	def test_state_saving(self):
		'''Test state file creation and format'''

		start_time = time.perf_counter()
		
		# Test state file creation
		with StateManager(self.test_seed, self.test_cidr, 1, 1) as manager:
			manager.update(12345)
		
		# Verify file exists
		state_file = os.path.join(tempfile.gettempdir(), f'pylcg_{self.test_seed}_{self.test_cidr.replace('/', '_')}_1_1.state')
		self.assertTrue(os.path.exists(state_file), 'State file not created')
		
		# Verify file content
		with open(state_file, 'r') as f:
			state = int(f.read().strip())
		self.assertEqual(state, 12345, 'State not saved correctly')
		
		elapsed = time.perf_counter() - start_time
		print_benchmark('State saving tests', 1, elapsed)
		

	def test_state_resumption(self):
		'''Test IP generation resumption from saved state'''

		start_time = time.perf_counter()
		test_size = 1000
		midpoint = test_size // 2
		
		# Generate initial sequence
		initial_ips = []
		initial_generator = ip_stream(self.test_cidr, seed=self.test_seed)
		lcg = None
		
		# Get first half and save LCG state
		for i in range(midpoint):
			try:
				ip = next(initial_generator)
				initial_ips.append(ip)
				if i == 0:  # Get LCG on first iteration
					lcg = initial_generator.gi_frame.f_locals['lcg']
			except StopIteration:
				break
				
		# Save state at midpoint
		midpoint_state = lcg.current
		
		# Complete the sequence
		try:
			while len(initial_ips) < test_size:
				initial_ips.append(next(initial_generator))
		except StopIteration:
			pass
		
		# Resume from midpoint
		resumed_ips = []
		resumed_generator = ip_stream(self.test_cidr, seed=self.test_seed, state=midpoint_state)
		
		try:
			while len(resumed_ips) < len(initial_ips[midpoint:]):
				resumed_ips.append(next(resumed_generator))
		except StopIteration:
			pass
		
		# Verify resumed sequence matches
		self.assertEqual(initial_ips[midpoint:], resumed_ips, 'Resumed sequence doesn\'t match original')
		
		elapsed = time.perf_counter() - start_time
		print_benchmark('State resumption tests', len(initial_ips), elapsed)
		

	def test_comprehensive_state_resumption(self):
		'''Test comprehensive state resumption with exclusions'''
		
		start_time = time.perf_counter()
		
		# Generate first batch of IPs
		first_batch = []
		batch_size = 1000
		generator = ip_stream(self.test_cidr, 1, 1, self.test_seed, None, self.excludes)
		
		# Get the first batch and save state
		for _ in range(batch_size):
			ip = next(generator)
			first_batch.append(ip)
			if len(first_batch) == batch_size // 2:
				# Get LCG state at midpoint
				lcg = generator.gi_frame.f_locals['lcg']
				saved_state = lcg.current
		
		# Resume from saved state
		resumed_generator = ip_stream(self.test_cidr, 1, 1, self.test_seed, saved_state, self.excludes)
		resumed_batch = []
		
		# Generate remaining IPs
		while len(resumed_batch) < batch_size // 2:
			resumed_batch.append(next(resumed_generator))
		
		# Verify resumed sequence
		self.assertEqual(first_batch[batch_size//2:], resumed_batch, 'Resumed sequence doesn\'t match original')
		
		# Verify no excluded IPs in either batch
		excluded_networks = [ipaddress.ip_network(ex) if '/' in ex else ipaddress.ip_network(ex + '/32') 
						   for ex in self.excludes]
		
		for ip in first_batch + resumed_batch:
			ip_obj = ipaddress.ip_address(ip)
			for excluded in excluded_networks:
				self.assertNotIn(ip_obj, excluded, f'Generated excluded IP: {ip}')
		
		elapsed = time.perf_counter() - start_time
		print_benchmark('Comprehensive state resumption tests', len(first_batch) + len(resumed_batch), elapsed)


if __name__ == '__main__':
	print(f'\n{Colors.CYAN}{Colors.BOLD}{'='*80}')
	print(f'PyLCG Comprehensive Test Suite')
	print(f'{'='*80}')
	print(f'{Colors.PURPLE}Test Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}')
	print(f'Python Version: {sys.version.split()[0]}')
	print(f'{'='*80}{Colors.ENDC}\n')
	unittest.main(verbosity=2)
