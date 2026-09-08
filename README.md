# PyLCG
> Ultra-fast Linear Congruential Generator for IP Sharding

PyLCG is a high-performance Python implementation of a memory-efficient IP address sharding system using Linear Congruential Generators *(LCG)* for deterministic random number generation. This tool enables distributed scanning & network reconnaissance by efficiently dividing IP ranges across multiple machines while maintaining pseudo-random ordering.

## Features

- Memory-efficient IP range processing
- Deterministic pseudo-random IP generation
- Sharding that costs the same whether you run 1 machine or 4,096
- Resume from any point in a range in a few milliseconds, no replay
- Zero dependencies beyond Python standard library
- Simple command-line interface and library usage

## Installation

```bash
pip install pylcg
```

## Usage

### Command Line Arguments

| Argument         | Required | Default  | Description                                                                                       |
|------------------|----------|----------|---------------------------------------------------------------------------------------------------|
| cidr             | Yes      | -        | Target IP range in CIDR format                                                                    |
| --seed           | No       | Random   | Random seed for LCG *(use when you need reproducible results)*                                    |
| --shard-num      | No       | 1        | Shard number *(1-based)*                                                                          |
| --total-shards   | No       | 1        | Total number of shards                                                                            |
| --state          | No       | None     | Resume from a state file path, or from a raw step number *(a raw step number requires --seed)*    |
| --state-file     | No       | ~/.pylcg | Where to write the state file                                                                     |
| --state-interval | No       | 1000     | Write the state file every N steps                                                                |
| --no-state       | No       | Off      | Do not write a state file at all                                                                  |
| --exclude        | No       | None     | IPs/CIDRs to exclude *(comma-separated list, file path, or 'private' for private & reserved)*      |

### Exit Codes

| Code | Meaning                                                       |
|------|---------------------------------------------------------------|
| 0    | Finished, or the reader closed the pipe *(`pylcg ... \| head`)* |
| 1    | Could not write the state file                                |
| 2    | Bad arguments, CIDR, exclude list or state file               |
| 130  | Interrupted with Ctrl-C, position saved                       |

Errors print one line to stderr. Nothing but IP addresses ever reaches stdout, including the random seed notice.

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

# Resume exactly where it stopped
pylcg 0.0.0.0/0 --seed 12345 --state ~/.pylcg/pylcg2_12345_0.0.0.0_0_1_1_4f53cda18c2baa0c.state

# Put the state file somewhere specific instead. Missing directories are created.
pylcg 0.0.0.0/0 --seed 12345 --state-file ~/scans/shard1.state

# Skip the state file entirely when you do not need to resume
pylcg 192.168.0.0/16 --no-state

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

Blank lines and comments are ignored, including indented ones. Surrounding whitespace is trimmed.

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

# With exclusions. The "private" keyword can be mixed in with explicit entries.
excludes = [
    '192.168.1.1',          # Single IP
    '192.168.100.0/24',     # CIDR range
    'private'               # All private & reserved ranges
]
for ip in ip_stream('0.0.0.0/0', exclude_list=excludes):
    print(ip)

# Resume from a known position (requires the original seed)
for ip in ip_stream('192.168.0.0/16', seed=12345, resume_step=5000):
    print(ip)

# No state file
for ip in ip_stream('192.168.0.0/16', seed=12345, write_state=False):
    print(ip)
```

## State Management & Resume Capability

Resuming needs exactly two things: the **seed** and the **step count**. Nothing else. The step count is the number of steps the shard has already taken, and the generator can jump straight to that position without replaying anything.

The state file is written to `~/.pylcg/` unless you pass `--state-file`:

```
pylcg2_[seed]_[cidr]_[shard]_[total]_[exclusions].state
```

Its contents are a single self-describing line:

```
pylcg2 seed=12345 cidr=0.0.0.0/0 shard=1 total=4 excludes=4f53cda18c2baa0c step=1830000000
```

Because the file carries the seed, CIDR, shard layout and a digest of the exclusion list, resuming validates all of it. If any of them disagree with what you pass on the command line, pylcg refuses to start instead of quietly walking a different sequence.

### Resume Guarantees

**No gaps.** A resumed run never skips an IP. The position is only recorded once the consumer has come back for the next IP, and when pylcg is writing to a pipe the output is flushed before the position is recorded. If that flush fails the position is deliberately left where it was.

**Duplicates are possible, and that is on purpose.** A run interrupted between state writes rewinds to the last recorded position, so up to one `--state-interval` of IPs get re-scanned. Re-scanning a few IPs is harmless. Skipping them is not, so the tradeoff always leans that way.

**One caveat when piping.** If the process reading pylcg's output exits early or is killed with data still sitting in the pipe, those IPs were delivered as far as pylcg can tell. Nothing can detect that from the generator side. `--state-interval 1` narrows the window to a single IP at roughly a tenth of the throughput.

### Choosing a state interval

Measured on `0.0.0.0/0`, best of 5 runs:

| --state-interval | Throughput      | At risk on an abrupt kill |
|------------------|-----------------|---------------------------|
| 1                | 66,830 ips/s    | 1 IP                      |
| 10               | 476,352 ips/s   | 10 IPs                    |
| 100              | 1,318,081 ips/s | 100 IPs                   |
| 1000 *(default)* | 1,546,499 ips/s | 1,000 IPs                 |
| 10000            | 1,591,130 ips/s | 10,000 IPs                |
| --no-state       | 1,579,282 ips/s | everything                |

Anything at 100 or above costs essentially nothing, so the default sits at 1000 to keep the exposure small for free.

## How It Works

### IP Address Integer Representation

Every IPv4 address is fundamentally a 32-bit number. For example, the IP address "192.168.1.1" can be broken down into its octets *(192, 168, 1, 1)* and converted to a single integer:
```
192.168.1.1 = (192 x 256^3) + (168 x 256^2) + (1 x 256^1) + (1 x 256^0)
            = 3232235777
```

This integer representation allows us to treat IP ranges as simple number sequences. A CIDR block like "192.168.0.0/16" becomes a continuous range of integers:
- Start: 192.168.0.0     -> 3232235520
- End:   192.168.255.255 -> 3232301055

By working with these integer representations, we can perform efficient mathematical operations on IP addresses without the overhead of string manipulation or complex data structures. This is where the Linear Congruential Generator comes into play.

### Linear Congruential Generator

PyLCG uses an LCG with the formula `X_{n+1} = (a * X_n + c) mod m` and three parameters:

| Name       | Variable | Value        |
|------------|----------|--------------|
| Multiplier | `a`      | `1664525`    |
| Increment  | `c`      | `1013904223` |
| Modulus    | `m`      | Power of 2   |

The modulus is not a fixed value. It is set dynamically to the smallest power of 2 that is >= the number of valid IPs in the target range. For any CIDR `/N`, the range contains exactly `2^(32-N)` addresses *(always a power of 2)*, so the modulus equals the range size exactly when no exclusions are applied.

These constants satisfy the **Hull-Dobell theorem**, which guarantees the LCG visits every integer in `[0, m-1]` exactly once before repeating *(a "full period")*. The three conditions are:

1. `c` and `m` share no common factors. `c` is odd and `m` is a power of 2, so `gcd(c, m) = 1`
2. `a - 1` is divisible by all prime factors of `m`. `a - 1 = 1664524` is divisible by 2, the only prime factor of any power of 2
3. `a - 1` is divisible by 4, required when `m` is divisible by 4. `1664524 / 4 = 416131`

The multiplier and increment come from the Numerical Recipes library. Their published spectral testing was done for `m = 2^32`; PyLCG uses `m = 2^k` for whatever `k` the range needs, so the full-period guarantee carries over exactly but the spectral result does not automatically transfer to smaller moduli.

Because those conditions only hold for a power-of-2 modulus, `LCG` rejects anything else. With `m = 1000` the period would collapse to 9 values, so it raises rather than silently handing back a generator that repeats almost immediately.

### Applying LCG to IP Addresses

Once we have our IP addresses as integers, the LCG generates indices that map directly to IPs in the range:

1. For a given IP range, calculate the number of valid IPs: `total_valid = end_ip - start_ip + 1` *(minus any exclusions)*

2. Set the LCG modulus to the smallest power of 2 >= `total_valid`

3. The LCG generates values in `[0, modulus-1]`. Each value is used as follows:
   - If `idx >= total_valid`: **skip** it *(rejection sampling, this value falls outside the range)*
   - If `idx < total_valid`: map it to an IP via `get_ip_at_index(idx)`, which translates the index to `start_ip + idx` *(adjusting for any excluded ranges)*

Because the LCG has a full period, it visits every integer in `[0, modulus-1]` exactly once. Since `[0, total_valid-1]` is a subset, every valid index appears exactly once. This ensures:
- Every IP in the range is visited exactly once, with no duplicates
- The sequence appears random but is deterministic
- Memory usage is constant regardless of range size
- The same seed always produces the same sequence

### Jumping Ahead

Composing `x -> a1*x + c1` with `x -> a2*x + c2` gives `x -> (a1*a2)*x + (a2*c1 + c2)`, which is another function of the same shape. Composing the step function with itself `n` times therefore takes `O(log n)` by binary exponentiation instead of `n` iterations.

This is what makes two things possible. Resuming a range jumps straight to the saved step rather than replaying the sequence, and each shard walks only its own steps instead of everyone else's.

Jumping 3 billion steps into a `/0` takes a few milliseconds.

### Sharding Algorithm

Every step of the sequence belongs to exactly one shard. Shard `i` owns steps `i+1`, `i+1+total_shards`, `i+1+2*total_shards` and so on. Summed across all shards that is exactly the full period, so there are no gaps and no overlaps.

A shard jumps directly from one of its own steps to the next using a precomputed `total_shards`-step jump, so it never walks the work belonging to other shards. Throughput is therefore flat in the shard count:

| Shards | Throughput      |
|--------|-----------------|
| 1      | 1,619,870 ips/s |
| 16     | 1,584,541 ips/s |
| 64     | 1,611,294 ips/s |
| 256    | 1,587,859 ips/s |
| 1024   | 1,541,978 ips/s |
| 4096   | 1,519,679 ips/s |

Coverage is always exact: every IP appears once, in exactly one shard. Shard *sizes* depend on whether any steps get rejected:

| Case                                              | Shard size spread |
|---------------------------------------------------|-------------------|
| No exclusions                                     | within 1          |
| Exclusions, shard count is a power of 2           | within 1          |
| Exclusions, shard count is not a power of 2       | varies slightly   |

Without exclusions `total_valid` equals the modulus, so every step produces an IP and the split is even by construction. With exclusions some steps map to no IP; when the shard count is a power of 2 those rejected steps still divide evenly across shards, so the split stays exact. Otherwise sizes scatter around the average by roughly `sqrt(n*p*(1-p))`. Measured on `10.0.0.0/16` with a `/18` excluded *(a harsh 25% rejection rate)*, 3 shards spread by 184 out of ~16,400 and 100 shards spread by 47 out of ~490. For `0.0.0.0/0 --exclude private` split 1000 ways it works out to about 0.02%.

### Exclusion System

Exclusion ranges are converted to integer `(start, end)` tuples, merged if overlapping or adjacent, and then clipped to the target CIDR bounds *(so excluding `10.0.0.0/8` from a `10.0.0.0/24` target only subtracts the 256 IPs that actually overlap)*.

The LCG operates on a "virtual" index space of `[0, total_valid-1]` where `total_valid` is the CIDR size minus excluded IPs. `get_ip_at_index` translates a virtual index to a real IP.

IPv4 addresses are formatted by joining cached octet strings rather than going through `ipaddress`, which is where most of the per-IP cost used to sit. The modulus is always a power of 2, so the wraparound is a mask rather than a divide.

The body of `get_ip_at_index` is also repeated inline inside `ip_stream`, because a method call per IP costs more than the arithmetic and the formatting combined. The two copies must agree, which the test suite enforces directly by comparing every IP the loop emits against what `get_ip_at_index` returns for the same index.

Rather than walking every exclusion on every lookup, PyLCG precomputes a running count of the valid IPs preceding each exclusion and binary searches it. Lookups stay fast no matter how big the exclude list gets, which matters if you feed it a full bogon or blocklist:

| Exclusion ranges | Linear walk | Binary search |
|------------------|-------------|---------------|
| 16               | 356,985/s   | 490,983/s     |
| 1,000            | 16,704/s    | 485,353/s     |
| 20,000           | 818/s       | 457,549/s     |

For example, with range `10.0.0.0/24` excluding `10.0.0.5` and `10.0.0.10-12`:
- Index 4 -> `10.0.0.4` *(before first exclusion)*
- Index 5 -> `10.0.0.6` *(skips `10.0.0.5`)*
- Index 9 -> `10.0.0.13` *(skips `10.0.0.10-12`)*

## IPv6

IPv6 CIDRs are accepted and work correctly for small ranges, but this is untested territory and not what the library is built for. The multiplier and increment are 32-bit constants; against a 2^128 modulus the full-period guarantee still holds but the distribution quality does not. Anything larger than a small prefix will not finish in any case.

## Contributing

Run the test suite:
```bash
python3 unit_test.py
```

It covers the full-period property, jump-ahead correctness at billion-step scale, complete coverage across shard counts, every possible resume point on small ranges, repeated crash-and-resume over full runs, and the throughput characteristics above.

###### A GoLang version of this library is also available [here](https://github.com/acidvegas/golcg). It has not been updated to match version 2.0 and will produce a different ordering.

---

###### Mirrors: [SuperNETs](https://git.supernets.org/acidvegas/pylcg) - [GitHub](https://github.com/acidvegas/pylcg) - [GitLab](https://gitlab.com/acidvegas/pylcg) - [Codeberg](https://codeberg.org/acidvegas/pylcg)

---

###### Mirrors: [SuperNETs](https://git.supernets.org/acidvegas/) • [GitHub](https://github.com/acidvegas/) • [GitLab](https://gitlab.com/acidvegas/) • [Codeberg](https://codeberg.org/acidvegas/)
