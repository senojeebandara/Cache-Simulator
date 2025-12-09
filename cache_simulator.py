"""
==============================================
  CACHE SIMULATOR 
==============================================

Description:
------------
This Python program simulates cache performance
for different configurations. It supports:
 - Direct-mapped, N-way set-associative, and fully associative caches.
 - Replacement policies: LRU, LFU, FIFO, Random.
 - Single-level and multi-level extensions (optional).
It computes:
 - Hit ratio, miss ratio, and AMAT (Average Memory Access Time).
"""

# ----------------------------------------------------------
# 1. Import required libraries
# ----------------------------------------------------------
from collections import defaultdict, deque
import random
import matplotlib.pyplot as plt
import pandas as pd


# ----------------------------------------------------------
# 2. Cache Simulator Class
# ----------------------------------------------------------
class CacheSimulator:
    def __init__(self, cache_size_bytes=8192, block_size=64, associativity=1,
                 replacement='LRU', hit_latency=1, miss_latency=100, name="L1"):
        """
        cache_size_bytes : int
            Total cache size in bytes
        block_size : int
            Block (line) size in bytes
        associativity : int
            Number of lines per set (1 = direct-mapped)
        replacement : str
            Replacement policy ('LRU', 'LFU', 'FIFO', 'Random')
        hit_latency, miss_latency : int
            Latency cycles for AMAT calculation
        name : str
            Cache level name (L1, L2, etc.)
        """
        self.name = name
        self.cache_size = cache_size_bytes
        self.block_size = block_size
        self.assoc = associativity
        self.replacement = replacement
        self.hit_latency = hit_latency
        self.miss_latency = miss_latency

        # Derived attributes
        self.num_lines = self.cache_size // self.block_size
        self.num_sets = self.num_lines // self.assoc

        self.reset()
        self._init_sets()

    def _init_sets(self):
        """Initialize cache structures."""
        self.sets = [deque() for _ in range(self.num_sets)]
        self.freq = [defaultdict(int) for _ in range(self.num_sets)]
        self.members = [set() for _ in range(self.num_sets)]

    def reset(self):
        """Reset counters."""
        self.hits = 0
        self.misses = 0
        self.accesses = 0

    def _index_tag(self, address):
        """Split address into index and tag."""
        block_addr = address // self.block_size
        index = block_addr % self.num_sets
        tag = block_addr // self.num_sets
        return index, tag

    def access(self, address):
        """Access a memory address (read)."""
        self.accesses += 1
        index, tag = self._index_tag(address)
        dq = self.sets[index]
        members = self.members[index]
        freq = self.freq[index]

        # Cache hit
        if tag in members:
            self.hits += 1
            if self.replacement == 'LRU':
                dq.remove(tag)
                dq.append(tag)
            elif self.replacement == 'LFU':
                freq[tag] += 1
            return True

        # Cache miss
        self.misses += 1
        if len(dq) < self.assoc:
            dq.append(tag)
            members.add(tag)
            freq[tag] = 1
        else:
            if self.replacement == 'FIFO':
                evict = dq.popleft()
            elif self.replacement == 'LRU':
                evict = dq.popleft()
            elif self.replacement == 'LFU':
                evict = min(dq, key=lambda t: freq[t])
                dq.remove(evict)
                freq.pop(evict, None)
            elif self.replacement == 'Random':
                evict = random.choice(list(dq))
                dq.remove(evict)
            else:
                evict = dq.popleft()

            members.remove(evict)
            dq.append(tag)
            members.add(tag)
            freq[tag] = 1

        return False

    def stats(self):
        """Return hit ratio, miss ratio, and AMAT."""
        hit_ratio = self.hits / self.accesses if self.accesses else 0
        miss_ratio = 1 - hit_ratio
        amat = (hit_ratio * self.hit_latency) + \
               (miss_ratio * (self.hit_latency + self.miss_latency))
        return {
            "Cache": self.name,
            "Accesses": self.accesses,
            "Hits": self.hits,
            "Misses": self.misses,
            "Hit Ratio": hit_ratio,
            "Miss Ratio": miss_ratio,
            "AMAT": amat
        }


# ----------------------------------------------------------
# 3. Trace Generation (simulate different workloads)
# ----------------------------------------------------------
def generate_locality_trace(length=10000, working_set=1024, randomness=0.05):
    """Simulates a realistic trace with locality."""
    trace = []
    for i in range(length):
        if random.random() < randomness:
            addr = random.randint(0, working_set * 10)
        else:
            addr = (i % working_set) * 4
        trace.append(addr)
    return trace


# ----------------------------------------------------------
# 4. Experiment Runner
# ----------------------------------------------------------
def run_experiment(trace, cache_configs):
    """Run multiple cache configurations and return results."""
    results = []
    for cfg in cache_configs:
        sim = CacheSimulator(**cfg)
        for addr in trace:
            sim.access(addr)
        stats = sim.stats()
        stats.update({
            "Cache Size (KB)": cfg["cache_size_bytes"] // 1024,
            "Assoc": cfg["associativity"],
            "Policy": cfg["replacement"]
        })
        results.append(stats)
    return pd.DataFrame(results)


# ----------------------------------------------------------
# 5. Example Experiments
# ----------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    trace = generate_locality_trace(length=20000)

    # Define experiment configs
    cache_configs = []
    for size in [4, 8, 16, 32]:  # in KB
        for assoc in [1, 2, 4, 8]:
            for policy in ['LRU', 'FIFO', 'LFU', 'Random']:
                cache_configs.append({
                    "cache_size_bytes": size * 1024,
                    "block_size": 64,
                    "associativity": assoc,
                    "replacement": policy,
                    "hit_latency": 1,
                    "miss_latency": 100,
                    "name": f"{size}KB-{assoc}way-{policy}"
                })

    df = run_experiment(trace, cache_configs)

    # Display table
    print(df[["Cache", "Hit Ratio", "Miss Ratio", "AMAT"]])

    # Plot: Hit ratio vs cache size
    plt.figure(figsize=(9, 5))
    for policy in df["Policy"].unique():
        subset = df[df["Policy"] == policy]
        plt.plot(subset["Cache Size (KB)"], subset["Hit Ratio"],
                 label=policy, marker='o')
    plt.title("Hit Ratio vs Cache Size for Different Policies")
    plt.xlabel("Cache Size (KB)")
    plt.ylabel("Hit Ratio")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot: AMAT vs cache size
    plt.figure(figsize=(9, 5))
    for policy in df["Policy"].unique():
        subset = df[df["Policy"] == policy]
        plt.plot(subset["Cache Size (KB)"], subset["AMAT"],
                 label=policy, marker='x')
    plt.title("AMAT vs Cache Size for Different Policies")
    plt.xlabel("Cache Size (KB)")
    plt.ylabel("AMAT (cycles)")
    plt.legend()
    plt.grid(True)
    plt.show()
