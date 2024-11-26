# PyLCG
> Ultra-fast Linear Congruential Generator for IP Sharding

PyLCG is a high-performance Python implementation of a memory-efficient IP address sharding system using Linear Congruential Generators (LCG) for deterministic random number generation. This tool enables distributed scanning & network reconnaissance by efficiently dividing IP ranges across multiple machines while maintaining pseudo-random ordering.

## Features

- Memory-efficient IP range processing
- Deterministic pseudo-random IP generation
- High-performance LCG implementation
- Support for sharding across multiple machines
- Zero dependencies beyond Python standard library
- Simple command-line interface

## Installation

### From PyPI
```bash
pip install pylcg
```

### From Source
```bash
git clone https://github.com/acidvegas/pylcg
cd pylcg
chmod +x pylcg.py
```

## Usage

### Command Line

```bash
./pylcg.py 192.168.0.0/16 --shard-num 1 --total-shards 4 --seed 12345
```

### As a Library

```python
from pylcg import ip_stream

# Generate IPs for the first shard of 4 total shards
for ip in ip_stream('192.168.0.0/16', shard_num=1, total_shards=4, seed=12345):
    print(ip)
```

## How It Works

### Linear Congruential Generator

PyLCG uses an optimized LCG implementation with carefully chosen parameters:
| Name       | Variable | Value        |
|------------|----------|--------------|
| Multiplier | `a`      | `1664525`    |
| Increment  | `c`      | `1013904223` |
| Modulus    | `m`      | `2^32`       |

This generates a deterministic sequence of pseudo-random numbers using the formula:
```
next = (a * current + c) mod m
```

### Memory-Efficient IP Processing

Instead of loading entire IP ranges into memory, PyLCG:
1. Converts CIDR ranges to start/end integers
2. Uses generator functions for lazy evaluation
3. Calculates IPs on-demand using index mapping
4. Maintains constant memory usage regardless of range size

### Sharding Algorithm

The sharding system uses an interleaved approach:
1. Each shard is assigned a subset of indices based on modulo arithmetic
2. The LCG randomizes the order within each shard
3. Work is distributed evenly across shards
4. No sequential scanning patterns

## Performance

PyLCG is designed for maximum performance:
- Generates millions of IPs per second
- Constant memory usage (~100KB)
- Minimal CPU overhead
- No disk I/O required

Benchmark results on a typical system:
- IP Generation: ~5-10 million IPs/second
- Memory Usage: < 1MB for any range size
- LCG Operations: < 1 microsecond per number

## Contributing

### Performance Optimization

We welcome contributions that improve PyLCG's performance. When submitting optimizations:

1. Run the included benchmark suite:
```bash
python3 unit_test.py
```

2. Include before/after benchmark results for:
- IP generation speed
- Memory usage
- LCG sequence generation
- Shard distribution metrics

3. Consider optimizing:
- Number generation algorithms
- Memory access patterns
- CPU cache utilization
- Python-specific optimizations

4. Document any tradeoffs between:
- Speed vs memory usage
- Randomness vs performance
- Complexity vs maintainability

### Benchmark Guidelines

When running benchmarks:
1. Use consistent hardware/environment
2. Run multiple iterations
3. Test with various CIDR ranges
4. Measure both average and worst-case performance
5. Profile memory usage patterns
6. Test shard distribution uniformity

## Roadmap

- [ ] IPv6 support
- [ ] Custom LCG parameters
- [ ] Configurable chunk sizes
- [ ] State persistence
- [ ] Resume capability
- [ ] S3/URL input support
- [ ] Extended benchmark suite

---

###### Mirrors: [acid.vegas](https://git.acid.vegas/pylcg) • [SuperNETs](https://git.supernets.org/acidvegas/pylcg) • [GitHub](https://github.com/acidvegas/pylcg) • [GitLab](https://gitlab.com/acidvegas/pylcg) • [Codeberg](https://codeberg.org/acidvegas/pylcg)
