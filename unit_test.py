#!/usr/bin/env python3
import unittest
import ipaddress
import time
from pylcg import IPRange, ip_stream, LCG

class Colors:
	BLUE   = '\033[94m'
	GREEN  = '\033[92m'
	YELLOW = '\033[93m'
	CYAN   = '\033[96m'
	RED    = '\033[91m'
	ENDC   = '\033[0m'

def print_header(message: str) -> None:
	print(f'\n\n{Colors.BLUE}{"="*80}')
	print(f'TEST: {message}')
	print(f'{"="*80}{Colors.ENDC}\n')

def print_success(message: str) -> None:
	print(f'{Colors.GREEN}✓ {message}{Colors.ENDC}')

def print_info(message: str) -> None:
	print(f"{Colors.CYAN}ℹ {message}{Colors.ENDC}")

def print_warning(message: str) -> None:
	print(f"{Colors.YELLOW}! {message}{Colors.ENDC}")

class TestIPSharder(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		print_header('Setting up test environment')
		cls.test_cidr = '192.0.0.0/16'  # 65,536 IPs
		cls.test_seed = 12345
		cls.total_shards = 4

		# Calculate expected IPs
		network = ipaddress.ip_network(cls.test_cidr)
		cls.all_ips = {str(ip) for ip in network}
		print_success(f"Initialized test environment with {len(cls.all_ips):,} IPs")

	def test_ip_range_initialization(self):
		print_header('Testing IPRange initialization')
		start_time = time.perf_counter()

		ip_range = IPRange(self.test_cidr)
		self.assertEqual(ip_range.total, 65536)

		first_ip = ip_range.get_ip_at_index(0)
		last_ip = ip_range.get_ip_at_index(ip_range.total - 1)

		elapsed = time.perf_counter() - start_time
		print_success(f'IP range initialization completed in {elapsed:.6f}s')
		print_info(f'IP range spans from {first_ip} to {last_ip}')
		print_info(f'Total IPs in range: {ip_range.total:,}')

	def test_lcg_sequence(self):
		print_header('Testing LCG sequence generation')

		# Test sequence generation speed
		lcg = LCG(seed=self.test_seed)
		iterations = 1_000_000

		start_time = time.perf_counter()
		for _ in range(iterations):
			lcg.next()
		elapsed = time.perf_counter() - start_time

		print_success(f'Generated {iterations:,} random numbers in {elapsed:.6f}s')
		print_info(f'Average time per number: {(elapsed/iterations)*1000000:.2f} microseconds')

		# Test deterministic behavior
		lcg1 = LCG(seed=self.test_seed)
		lcg2 = LCG(seed=self.test_seed)

		start_time = time.perf_counter()
		for _ in range(1000):
			self.assertEqual(lcg1.next(), lcg2.next())
		elapsed = time.perf_counter() - start_time

		print_success(f'Verified LCG determinism in {elapsed:.6f}s')

	def test_shard_distribution(self):
		print_header('Testing shard distribution and randomness')

		# Test distribution across shards
		sample_size = 65_536  # Full size for /16
		shard_counts = {i: 0 for i in range(1, self.total_shards + 1)}  # 1-based sharding
		unique_ips = set()
		duplicate_count = 0

		start_time = time.perf_counter()

		# Collect IPs from each shard
		for shard in range(1, self.total_shards + 1):  # 1-based sharding
			ip_gen = ip_stream(self.test_cidr, shard, self.total_shards, self.test_seed)
			shard_unique = set()

			# Get all IPs from this shard
			for ip in ip_gen:
				if ip in unique_ips:
					duplicate_count += 1
				else:
					unique_ips.add(ip)
					shard_unique.add(ip)

			shard_counts[shard] = len(shard_unique)

		elapsed = time.perf_counter() - start_time

		# Print distribution statistics
		print_success(f'Generated {len(unique_ips):,} IPs in {elapsed:.6f}s')
		print_info(f'Average time per IP: {(elapsed/len(unique_ips))*1000000:.2f} microseconds')
		print_info(f'Unique IPs generated: {len(unique_ips):,}')

		if duplicate_count > 0:
			print_warning(f'Duplicates found: {duplicate_count:,} ({(duplicate_count/len(unique_ips))*100:.2f}%)')

		expected_per_shard = sample_size // self.total_shards
		for shard, count in shard_counts.items():
			deviation = abs(count - expected_per_shard) / expected_per_shard * 100
			print_info(f'Shard {shard}: {count:,} unique IPs ({deviation:.2f}% deviation from expected)')

		# Test randomness by checking sequential patterns
		ips_list = sorted([int(ipaddress.ip_address(ip)) for ip in list(unique_ips)[:1000]])
		sequential_count = sum(1 for i in range(len(ips_list)-1) if ips_list[i] + 1 == ips_list[i+1])
		sequential_percentage = (sequential_count / (len(ips_list)-1)) * 100

		print_info(f'Sequential IP pairs in first 1000: {sequential_percentage:.2f}% (lower is more random)')

if __name__ == '__main__':
	print(f"\n{Colors.CYAN}{'='*80}")
	print(f"Starting IP Sharder Tests - Testing with 65,536 IPs (/16 network)")
	print(f"{'='*80}{Colors.ENDC}\n")
	unittest.main(verbosity=2)
