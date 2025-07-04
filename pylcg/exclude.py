#!/usr/bin/env python
# PyLCG - Linear Congruential Generator for IP Sharding - Developed by acidvegas in Python (https://github.com/acidvegas/pylcg)
# pylcg/exclude.py

import ipaddress
import os

# Common reserved ranges
RFC_RANGES = {
    'private': [  # All private & reserved ranges
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

def parse_excludes(exclude_list: list) -> set:
    '''
    Convert a list of IPs/CIDRs to a set of (start, end) integer ranges
    
    :param exclude_list: List of IP addresses or CIDR ranges to exclude, can be:
                        - List of IPs/CIDRs
                        - Comma-separated string of IPs/CIDRs
                        - File path containing IPs/CIDRs (one per line)
                        - "private" for all private & reserved ranges
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
            with open(exclude_list, 'r') as f:
                exclude_list = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        # Treat as comma-separated list
        else:
            exclude_list = [x.strip() for x in exclude_list.split(',')]
    
    # Parse each item in the exclude list
    for item in exclude_list:
        try:
            # Handle single IP
            if '/' not in item:
                ip_int = int(ipaddress.ip_address(item))
                ranges.add((ip_int, ip_int))
            # Handle CIDR
            else:
                network = ipaddress.ip_network(item, strict=False)
                ranges.add((int(network.network_address), int(network.broadcast_address)))
        except ValueError:
            raise ValueError(f'Invalid IP or CIDR: {item}')
    
    return ranges


def optimize_ranges(ranges: set) -> list:
    '''
    Merge overlapping ranges and sort them
    
    :param ranges: Set of (start, end) IP ranges
    '''

    # Convert to list and sort by start address
    sorted_ranges = sorted(ranges)

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