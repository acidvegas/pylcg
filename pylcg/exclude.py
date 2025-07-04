#!/usr/bin/env python
# PyLCG - Linear Congruential Generator for IP Sharding - Developed by acidvegas in Python (https://github.com/acidvegas/pylcg)
# pylcg/exclude.py

import ipaddress


def parse_excludes(exclude_list: list) -> set:
    '''
    Convert a list of IPs/CIDRs to a set of (start, end) integer ranges
    
    :param exclude_list: List of IP addresses or CIDR ranges to exclude
    '''
    
    # Initialize the set of ranges
    ranges = set()
    
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