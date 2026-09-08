#!/usr/bin/env python3
# PyLCG - Linear Congruential Generator for IP Sharding - Developed by acidvegas in Python (https://github.com/acidvegas/pylcg)
# pylcg/cli.py

import argparse
import os
import random
import sys

from .core  import ip_stream, IPRange
from .state import StateManager


def main():
	'''Parse arguments and run, turning expected failures into one clear line instead of a traceback'''

	args = parse_args()

	try:
		run(args)
	except BrokenPipeError:
		# Downstream closed the pipe, which is normal for things like `pylcg ... | head`
		os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())

		return 0
	except KeyboardInterrupt:
		print('pylcg: interrupted', file=sys.stderr)

		return 130
	except ValueError as e:
		print(f'pylcg: {e}', file=sys.stderr)

		return 2
	except OSError as e:
		print(f'pylcg: {e.strerror or e}: {e.filename or ""}'.rstrip(': '), file=sys.stderr)

		return 1

	return 0


def parse_args() -> argparse.Namespace:
	'''Build the argument parser and validate the shard options'''

	parser = argparse.ArgumentParser(prog='pylcg', description='Ultra-fast random IP address generator with optional sharding')
	parser.add_argument('cidr', help='Target IP range in CIDR format')
	parser.add_argument('--shard-num', type=int, default=1, help='Shard number (1-based)')
	parser.add_argument('--total-shards', type=int, default=1, help='Total number of shards (default: 1, no sharding)')
	parser.add_argument('--seed', type=int, help='Random seed for LCG (optional, random if not provided)')
	parser.add_argument('--state', help='Resume from a state file path, or from a raw step number (a raw step number requires --seed)')
	parser.add_argument('--state-file', help='Where to write the state file (default: ~/.pylcg)')
	parser.add_argument('--state-interval', type=int, default=1_000, help='Write the state file every N steps (default: 1000)')
	parser.add_argument('--no-state', action='store_true', help='Do not write a state file at all')
	parser.add_argument('--exclude', help='IPs/CIDRs to exclude. Comma-separated list, file path, or "private" for all private & reserved ranges')

	args = parser.parse_args()

	# argparse prints usage and exits 2 for these, which is the right shape for a bad flag
	if args.total_shards < 1:
		parser.error('--total-shards must be at least 1')

	if args.shard_num < 1:
		parser.error('--shard-num must be at least 1')

	if args.shard_num > args.total_shards:
		parser.error(f'--shard-num {args.shard_num} is greater than --total-shards {args.total_shards}')

	if args.state_interval < 1:
		parser.error('--state-interval must be at least 1')

	return args


def run(args: argparse.Namespace):
	'''
	Stream the shard to stdout

	:param args: Parsed command line arguments
	'''

	# Parse state: either a file path or a raw step number
	resume_step = 0

	if args.state is not None:
		if os.path.isfile(args.state):
			state = StateManager.load(args.state)

			# Anything also passed on the command line has to agree with the file, or
			# the resumed run would silently walk a different sequence
			if args.seed is not None and args.seed != state['seed']:
				raise ValueError(f'--seed {args.seed} does not match the seed in {args.state} ({state["seed"]})')

			if args.cidr != state['cidr']:
				raise ValueError(f'CIDR {args.cidr} does not match the CIDR in {args.state} ({state["cidr"]})')

			if (args.shard_num, args.total_shards) != (state['shard'], state['total']):
				raise ValueError(f'Shard {args.shard_num}/{args.total_shards} does not match the shard in {args.state} ({state["shard"]}/{state["total"]})')

			args.seed   = state['seed']
			resume_step = state['step']
			expected    = state['excludes']
		else:
			try:
				resume_step = int(args.state)
			except ValueError:
				raise ValueError(f'--state: {args.state!r} is neither an existing state file nor a step number')

			if resume_step < 0:
				raise ValueError('--state step number cannot be negative')

			expected = None

			if args.seed is None:
				raise ValueError('When resuming from a raw step number, you must provide the --seed that was used originally')
	else:
		expected = None

	# Generate random seed if not provided. The notice goes to stderr so it cannot be
	# mistaken for an address by whatever is reading the IP stream.
	if args.seed is None:
		args.seed = random.randint(0, 2**32-1)
		print(f'Using random seed: {args.seed}', file=sys.stderr)

	# Convert exclude argument to list if provided
	exclude_list = args.exclude if args.exclude else None

	# Resuming against a different exclude list would renumber every index, so refuse
	if expected is not None:
		if IPRange(args.cidr, exclude_list).fingerprint != expected:
			raise ValueError(f'--exclude does not match the exclude list used when {args.state} was written')

	# One large write beats thousands of small ones, so IPs are collected into a batch.
	# The batch is drained and flushed immediately before every state write, so the
	# recorded position can still never run ahead of what has left the process.
	batch = []

	def drain():
		'''Write any pending IPs and push them out of the process'''

		if batch:
			sys.stdout.write('\n'.join(batch) + '\n')
			batch.clear()

		sys.stdout.flush()

	stream = ip_stream(args.cidr, args.shard_num, args.total_shards, args.seed, exclude_list, resume_step, args.state_file, args.state_interval, not args.no_state, drain)

	# Closing the stream explicitly rather than leaving it to the collector means an
	# interrupt still drains, flushes and records the position in that order
	try:
		for ip in stream:
			batch.append(ip)

			# Bound the memory the batch can take, which also covers --no-state where
			# nothing else would ever drain it
			if len(batch) >= args.state_interval:
				sys.stdout.write('\n'.join(batch) + '\n')
				batch.clear()

		drain()
	finally:
		stream.close()



if __name__ == '__main__':
	sys.exit(main())
