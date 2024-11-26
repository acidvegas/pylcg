import argparse
from .core import ip_stream

def main():
	parser = argparse.ArgumentParser(description='Ultra-fast random IP address generator with optional sharding')
	parser.add_argument('cidr', help='Target IP range in CIDR format')
	parser.add_argument('--shard-num', type=int, default=1, help='Shard number (1-based)')
	parser.add_argument('--total-shards', type=int, default=1, help='Total number of shards (default: 1, no sharding)')
	parser.add_argument('--seed', type=int, default=0, help='Random seed for LCG')

	args = parser.parse_args()

	if args.total_shards < 1:
		raise ValueError('Total shards must be at least 1')

	if args.shard_num > args.total_shards:
		raise ValueError('Shard number must be less than or equal to total shards')

	if args.shard_num < 1:
		raise ValueError('Shard number must be at least 1')

	for ip in ip_stream(args.cidr, args.shard_num, args.total_shards, args.seed):
		print(ip)

if __name__ == '__main__':
	main()
