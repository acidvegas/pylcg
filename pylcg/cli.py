#!/usr/bin/env python
# PyLCG - Linear Congruential Generator for IP Sharding - Developed by acidvegas in Python (https://github.com/acidvegas/pylcg)
# pylcg/cli.py

import argparse
import os
import random

from .core  import ip_stream
from .state import StateManager


def main():
	parser = argparse.ArgumentParser(description='Ultra-fast random IP address generator with optional sharding')
	parser.add_argument('cidr', help='Target IP range in CIDR format')
	parser.add_argument('--shard-num', type=int, default=1, help='Shard number (1-based)')
	parser.add_argument('--total-shards', type=int, default=1, help='Total number of shards (default: 1, no sharding)')
	parser.add_argument('--seed', type=int, help='Random seed for LCG (optional, random if not provided)')
	parser.add_argument('--state', type=str, help='Resume from state file path or LCG state integer (requires --seed)')
	parser.add_argument('--exclude', help='IPs/CIDRs to exclude. Comma-separated list, file path, or "private" for all private & reserved ranges')

	args = parser.parse_args()

	if args.total_shards < 1:
		raise ValueError('Total shards must be at least 1')

	if args.shard_num > args.total_shards:
		raise ValueError('Shard number must be less than or equal to total shards')

	if args.shard_num < 1:
		raise ValueError('Shard number must be at least 1')

	# Parse state: either a file path or a raw integer
	state_val = None
	resume_yielded = None

	if args.state is not None:
		if os.path.isfile(args.state):
			state_val, resume_yielded = StateManager.load(args.state)
		else:
			try:
				state_val = int(args.state)
			except ValueError:
				raise ValueError(f'--state must be a valid file path or integer, got: {args.state}')

	if state_val is not None and args.seed is None:
		raise ValueError('When using --state, you must provide the --seed that was used originally')

	# Generate random seed if not provided
	if args.seed is None:
		args.seed = random.randint(0, 2**32-1)
		print(f'Using random seed: {args.seed}')

	# Convert exclude argument to list if provided
	exclude_list = args.exclude if args.exclude else None

	for ip in ip_stream(args.cidr, args.shard_num, args.total_shards, args.seed, state_val, exclude_list, resume_yielded):
		print(ip)



if __name__ == '__main__':
	main()
