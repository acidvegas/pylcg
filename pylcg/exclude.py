#!/usr/bin/env python3
# PyLCG - Linear Congruential Generator for IP Sharding - Developed by acidvegas in Python (https://github.com/acidvegas/pylcg)
# pylcg/exclude.py

import ipaddress
import os


# Common reserved ranges
RFC_RANGES = {
	'private': [
		'0.0.0.0/8',         # Current network
		'10.0.0.0/8',        # Private network (Class A)
		'100.64.0.0/10',     # Carrier-grade NAT
		'127.0.0.0/8',       # Loopback
		'169.254.0.0/16',    # Link-local
		'172.16.0.0/12',     # Private network (Class B)
		'192.0.0.0/24',      # IETF Protocol Assignments
		'192.0.2.0/24',      # TEST-NET-1
		'192.88.99.0/24',    # IPv6 to IPv4 relay
		'192.168.0.0/16',    # Private network (Class C)
		'198.18.0.0/15',     # Network benchmark tests
		'198.51.100.0/24',   # TEST-NET-2
		'203.0.113.0/24',    # TEST-NET-3
		'224.0.0.0/4',       # Multicast
		'240.0.0.0/4',       # Reserved for future use
		'255.255.255.255/32' # Broadcast
	]
}


def parse_excludes(exclude_list: list, version: int = None) -> set:
	'''
	Convert a list of IPs/CIDRs to a set of (start, end) integer ranges

	:param exclude_list: List of IP addresses or CIDR ranges to exclude, can be:
	                     - List of IPs/CIDRs (individual entries may also be a keyword like "private")
	                     - Comma-separated string of IPs/CIDRs
	                     - File path containing IPs/CIDRs (one per line)
	                     - "private" for all private & reserved ranges
	:param version: Only keep entries of this IP version, skipping the rest (default: None for all)
	'''

	# Initialize the set of ranges
	ranges = set()

	# Handle different input types
	if isinstance(exclude_list, str):
		# Check if it's a special keyword
		if exclude_list.lower() in RFC_RANGES:
			exclude_list = RFC_RANGES[exclude_list.lower()]
		# Check if it's a file
		elif os.path.isfile(exclude_list):
			exclude_list = read_exclude_file(exclude_list)
		# Treat as comma-separated list
		else:
			exclude_list = [x.strip() for x in exclude_list.split(',')]

			# A lone token that is not a keyword, not on disk and not an address is
			# almost always a mistyped file name, so say so rather than complaining
			# that a file name is not an IP address
			if len(exclude_list) == 1:
				try:
					ipaddress.ip_network(exclude_list[0], strict=False)
				except ValueError:
					raise ValueError(f'--exclude: {exclude_list[0]!r} is not an IP, a CIDR, the keyword "private", or an existing file')

	# Expand any keyword or file path appearing as an individual entry, so a list
	# like ['192.168.1.1', 'private'] works the same as the bare 'private' string
	expanded = []

	for item in exclude_list:
		item = item.strip()

		if not item:
			continue

		if item.lower() in RFC_RANGES:
			expanded.extend(RFC_RANGES[item.lower()])
		elif os.path.isfile(item):
			expanded.extend(read_exclude_file(item))
		else:
			expanded.append(item)

	# Parse each item in the exclude list
	for item in expanded:
		try:
			# A bare IP parses as a single-address network
			network = ipaddress.ip_network(item, strict=False)
		except ValueError:
			raise ValueError(f'Invalid IP or CIDR: {item}')

		# IPv4 and IPv6 share the same integer space, so an entry from the other
		# family would exclude unrelated addresses from the target
		if version is None or network.version == version:
			ranges.add((int(network.network_address), int(network.broadcast_address)))

	return ranges


def read_exclude_file(filepath: str) -> list:
	'''
	Read exclusion entries from a file, one per line, ignoring blanks & comments

	:param filepath: Path to the file containing IPs/CIDRs
	'''

	entries = []

	with open(filepath, 'r') as f:
		for line in f:
			# Strip before testing for a comment so indented comments are ignored too
			line = line.strip()

			if line and not line.startswith('#'):
				entries.append(line)

	return entries


def optimize_ranges(ranges: set) -> list:
	'''
	Merge overlapping ranges and sort them

	:param ranges: Set of (start, end) IP ranges
	'''

	# Convert to list and sort by start address
	sorted_ranges = sorted(ranges)

	# Nothing to merge (an exclude file of only comments lands here)
	if not sorted_ranges:
		return []

	# Initialize the optimized list
	optimized = []

	# Initialize the current range
	current_start, current_end = sorted_ranges[0]

	# Merge overlapping ranges
	for start, end in sorted_ranges[1:]:
		if start <= current_end + 1:
			# If the current range overlaps with the next, merge them
			current_end = max(current_end, end)
		else:
			# Otherwise, add the current range to the optimized list
			optimized.append((current_start, current_end))
			current_start, current_end = start, end

	# Add the last range to the optimized list
	optimized.append((current_start, current_end))

	return optimized


def calculate_excluded_count(ranges: list) -> int:
	'''
	Calculate total number of IPs in excluded ranges

	:param ranges: List of (start, end) IP ranges
	'''

	return sum(end - start + 1 for start, end in ranges)
