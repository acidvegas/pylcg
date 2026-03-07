# PyLCG
> Ultra-fast Linear Congruential Generator for IP Sharding

PyLCG is a high-performance Python implementation of a memory-efficient IP address sharding system using Linear Congruential Generators (LCG) for deterministic random number generation. This tool enables distributed scanning & network reconnaissance by efficiently dividing IP ranges across multiple machines while maintaining pseudo-random ordering.

###### A GoLang version of this library is also available [here](https://github.com/acidvegas/golcg)

## Features

- Memory-efficient IP range processing
- Deterministic pseudo-random IP generation
- High-performance LCG implementation
- Support for sharding across multiple machines
- Zero dependencies beyond Python standard library
- Simple command-line interface and library usage

## Installation

```bash
pip install pylcg
```

## Usage

### Command Line Arguments

| Argument       | Required | Default | Description                                                                                                |
|---------------|----------|---------|----------------------------------------------------------------------------------------------------------|
| cidr          | Yes      | -       | Target IP range in CIDR format                                                                            |
| --seed        | No       | Random  | Random seed for LCG (use when you need reproducible results)                                              |
| --shard-num   | No       | 1       | Shard number (1-based)                                                                                    |
| --total-shards| No       | 1       | Total number of shards                                                                                    |
| --state       | No       | None    | Resume from state file path or LCG state integer (requires --seed to be set)                             |
| --exclude     | No       | None    | IPs/CIDRs to exclude (comma-separated list, file path, or 'private' for all private & reserved ranges)   |

### Command Line Examples

```bash
# Basic usage (random seed each time)
pylcg 192.168.0.0/16

# Use specific seed for reproducible results
pylcg 192.168.0.0/16 --seed 12345

# Sharding with 4 total shards (random seed)
pylcg 192.168.0.0/16 --shard-num 1 --total-shards 4

# Exclude private & reserved ranges
pylcg 0.0.0.0/0 --exclude private

# Exclude specific IPs and ranges (comma-separated)
pylcg 10.0.0.0/8 --exclude "10.0.0.1,10.0.0.2,10.0.1.0/24"

# Exclude IPs/ranges from a file
pylcg 0.0.0.0/0 --exclude excludes.txt

# Resume from state file (requires original seed)
pylcg 192.168.0.0/16 --seed 12345 --state /tmp/pylcg_12345_192.168.0.0_16_1_1.state

# Resume from raw LCG state integer (legacy)
pylcg 192.168.0.0/16 --seed 12345 --state 987654321

# Pipe to dig for PTR record lookups
pylcg 192.168.0.0/16 | while read ip; do
    echo -n "$ip -> "
    dig +short -x $ip
done

# One-liner for PTR lookups
pylcg 198.150.0.0/16 | xargs -I {} dig +short -x {}

# Parallel PTR lookups
pylcg 198.150.0.0/16 | parallel "dig +short -x {} | sed 's/^/{} -> /'"
```

### Exclude File Format
```text
# Comments are supported
# Individual IPs
8.8.8.8
1.1.1.1

# CIDR ranges
10.0.0.0/8
172.16.0.0/12
192.168.0.0/16

# Mix of both
169.254.0.0/16
203.0.113.37
```

### As a Library

```python
from pylcg import ip_stream

# Basic usage (random seed)
for ip in ip_stream('192.168.0.0/16'):
    print(ip)

# With specific seed
for ip in ip_stream('192.168.0.0/16', seed=12345):
    print(ip)

# With sharding
for ip in ip_stream('192.168.0.0/16', shard_num=1, total_shards=4, seed=12345):
    print(ip)

# With exclusions
excludes = [
    '192.168.1.1',          # Single IP
    '192.168.100.0/24',     # CIDR range
    'private'               # All private & reserved ranges
]
for ip in ip_stream('0.0.0.0/0', exclude_list=excludes):
    print(ip)

# Resume from previous state (requires original seed and yielded count)
for ip in ip_stream('192.168.0.0/16', seed=12345, state=987654321, resume_yielded=5000):
    print(ip)
```

## State Management & Resume Capability

PyLCG automatically saves its state after every IP yielded to enable resume functionality in case of interruption. The state file is written to your system's temp directory (usually `/tmp` on Unix systems or `%TEMP%` on Windows) using a line-buffered file handle for efficiency.

The state file follows the naming pattern:
```
pylcg_[seed]_[cidr]_[shard]_[total].state
```

For example:
```
pylcg_12345_192.168.0.0_16_1_4.state
```

The state file contains two comma-separated values: the LCG's internal state and the number of IPs yielded so far (e.g. `987654321,5000`). This allows instant resumption without replaying the sequence.

To resume, pass the state file path directly to `--state`:
```bash
pylcg 192.168.0.0/16 --shard-num 1 --total-shards 4 --seed 12345 --state /tmp/pylcg_12345_192.168.0.0_16_1_4.state
```

You can also pass a raw LCG state integer for backwards compatibility, though this triggers a slower replay to reconstruct progress:
```bash
pylcg 192.168.0.0/16 --seed 12345 --state 987654321
```

Note: When using the `--state` parameter, you must provide the same `--seed` that was used in the original run.

## How It Works

### IP Address Integer Representation

Every IPv4 address is fundamentally a 32-bit number. For example, the IP address "192.168.1.1" can be broken down into its octets (192, 168, 1, 1) and converted to a single integer:
```
192.168.1.1 = (192 × 256³) + (168 × 256²) + (1 × 256¹) + (1 × 256⁰)
             = 3232235777
```

This integer representation allows us to treat IP ranges as simple number sequences. A CIDR block like "192.168.0.0/16" becomes a continuous range of integers:
- Start: 192.168.0.0   → 3232235520
- End:   192.168.255.255 → 3232301055

By working with these integer representations, we can perform efficient mathematical operations on IP addresses without the overhead of string manipulation or complex data structures. This is where the Linear Congruential Generator comes into play.

### Linear Congruential Generator

PyLCG uses an LCG with the formula `X_{n+1} = (a * X_n + c) mod m` and three carefully chosen parameters:

| Name       | Variable | Value        |
|------------|----------|--------------|
| Multiplier | `a`      | `1664525`    |
| Increment  | `c`      | `1013904223` |
| Modulus    | `m`      | Power of 2   |

The modulus is not a fixed value — it is set dynamically to the smallest power of 2 that is ≥ the number of valid IPs in the target range. For any CIDR `/N`, the range contains exactly `2^(32-N)` addresses (always a power of 2), so the modulus equals the range size exactly when no exclusions are applied.

These constants satisfy the **Hull-Dobell theorem**, which guarantees the LCG visits every integer in `[0, m-1]` exactly once before repeating (a "full period"). The three conditions are:

1. `c` and `m` share no common factors — `c` is odd, `m` is a power of 2, so `gcd(c, m) = 1`
2. `a - 1` is divisible by all prime factors of `m` — `a - 1 = 1664524` is divisible by 2 (the only prime factor of any power of 2)
3. `a - 1` is divisible by 4 (required when `m` is divisible by 4) — `1664524 / 4 = 416131`

The multiplier and increment values originate from the Numerical Recipes library and were selected for strong spectral test performance, ensuring good distribution across the sequence.

### Applying LCG to IP Addresses

Once we have our IP addresses as integers, the LCG generates indices that map directly to IPs in the range:

1. For a given IP range, calculate the number of valid IPs: `total_valid = end_ip - start_ip + 1` (minus any exclusions)

2. Set the LCG modulus to the smallest power of 2 ≥ `total_valid`

3. The LCG generates values in `[0, modulus-1]`. Each value is used as follows:
   - If `idx >= total_valid`: **skip** it (rejection sampling — this value falls outside the range)
   - If `idx < total_valid`: map it to an IP via `get_ip_at_index(idx)`, which translates the index to `start_ip + idx` (adjusting for any excluded ranges)

Because the LCG has a full period, it visits every integer in `[0, modulus-1]` exactly once. Since `[0, total_valid-1]` is a subset, every valid index appears exactly once. This ensures:
- Every IP in the range is visited exactly once, with no duplicates
- The sequence appears random but is deterministic
- Memory usage is constant regardless of range size
- The same seed always produces the same sequence

### Sharding Algorithm

All shards use the same seed and the same LCG sequence. Each generated index `idx` is assigned to exactly one shard by `idx % total_shards == shard_index`. Since the LCG visits every index exactly once, each shard receives its fair share with no overlap and no gaps.

Shard sizes are balanced to within 1 IP: each shard gets `total_valid // total_shards` IPs, with the first `total_valid % total_shards` shards receiving one extra. Because the indices are pseudo-randomly distributed, each shard's IPs are spread across the entire range rather than clustered in sequential blocks.

### Exclusion System

Exclusion ranges are converted to integer `(start, end)` tuples, merged if overlapping or adjacent, and then clipped to the target CIDR bounds (so excluding `10.0.0.0/8` from a `10.0.0.0/24` target only subtracts the 256 IPs that actually overlap).

The LCG operates on a "virtual" index space of `[0, total_valid-1]` where `total_valid` is the CIDR size minus excluded IPs. The `get_ip_at_index` method translates a virtual index to a real IP by walking through the sorted exclusion ranges and skipping over gaps:

```python
target     = index
current_ip = self.start

for range_start, range_end in self.excluded_ranges:
    gap = range_start - current_ip
    if target < gap:
        return str(ipaddress.ip_address(current_ip + target))
    target     -= gap
    current_ip  = range_end + 1

return str(ipaddress.ip_address(current_ip + target))
```

For example, with range `10.0.0.0/24` excluding `10.0.0.5` and `10.0.0.10-12`:
- Index 4 → `10.0.0.4` (before first exclusion)
- Index 5 → `10.0.0.6` (skips `10.0.0.5`)
- Index 9 → `10.0.0.13` (skips `10.0.0.10-12`)

### State Management

The LCG is fully deterministic: given a state value, all future outputs are fixed. The state file saves two values after every yielded IP:

1. `lcg.current` — the LCG's internal state, which determines all future outputs
2. `yielded` — how many IPs have been generated so far, which determines where the shard is in its work

To resume, both values are restored and generation continues from exactly where it stopped. No replay or recalculation is needed.

### Contributing

We welcome contributions that improve PyLCG's performance. When submitting optimizations:

1. Run the included benchmark suite:
```bash
python3 unit_test.py
```

---

###### Mirrors: [SuperNETs](https://git.supernets.org/acidvegas/pylcg) • [GitHub](https://github.com/acidvegas/pylcg) • [GitLab](https://gitlab.com/acidvegas/pylcg) • [Codeberg](https://codeberg.org/acidvegas/pylcg)
