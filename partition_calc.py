#!/usr/bin/env python3
"""
Partition calculation module for SFH framework.
Provides the PartitionCalculator class with methods for
enumerating integer partitions, verifying combinatorial identities,
and running Monte Carlo style partition analyses.
"""

import itertools
import random
from collections import defaultdict
import numpy as np

class PartitionCalculator:
    def __init__(self, max_n: int = 50, seed: int = 42):
        self.max_n = max_n
        self.rng = random.Random(seed)

    # ---------------------------
    # Partition generation
    # ---------------------------
    def generate_all_partitions(self, n: int):
        """Generate all integer partitions of n using recursion."""
        if n == 0:
            return [[]]
        result = []
        self._partitions_recursive(n, n, [], result)
        return result

    def _partitions_recursive(self, n, max_val, prefix, result):
        if n == 0:
            result.append(prefix)
            return
        for i in range(min(n, max_val), 0, -1):
            self._partitions_recursive(n - i, i, prefix + [i], result)

    # ---------------------------
    # Classification
    # ---------------------------
    def classify_partitions(self, partitions):
        """Classify partitions into distinct parts, odd parts, etc."""
        distinct_parts = [p for p in partitions if len(p) == len(set(p))]
        odd_parts_only = [p for p in partitions if all(x % 2 == 1 for x in p)]

        return {
            "distinct_parts": distinct_parts,
            "odd_parts_only": odd_parts_only,
            "distinct_parts_count": len(distinct_parts),
            "odd_parts_only_count": len(odd_parts_only),
            "euler_identity_verified": len(distinct_parts) == len(odd_parts_only)
        }

    # ---------------------------
    # Partition function value
    # ---------------------------
    def partition_function_value(self, n: int) -> int:
        """Return number of partitions of n using Euler’s pentagonal recurrence."""
        if n < 0:
            return 0
        if n == 0:
            return 1

        total = 0
        k = 1
        while True:
            pent1 = k * (3 * k - 1) // 2
            pent2 = k * (3 * k + 1) // 2
            if pent1 > n and pent2 > n:
                break

            sign = -1 if (k % 2 == 0) else 1
            if pent1 <= n:
                total += sign * self.partition_function_value(n - pent1)
            if pent2 <= n:
                total += sign * self.partition_function_value(n - pent2)

            k += 1
        return total

    # ---------------------------
    # Monte Carlo Partition Analysis
    # ---------------------------
    def run_partition_analysis(self, num_samples=1000):
        """
        Run Monte Carlo sampling of partitions up to self.max_n.
        Returns a dict containing sampled coherence/fertility distributions.
        """
        results = {"n": [], "lengths": [], "max_parts": [], "min_parts": []}

        for _ in range(num_samples):
            n = self.rng.randint(1, self.max_n)
            partitions = self.generate_all_partitions(n)
            if not partitions:
                continue
            chosen = self.rng.choice(partitions)
            results["n"].append(n)
            results["lengths"].append(len(chosen))
            results["max_parts"].append(max(chosen) if chosen else 0)
            results["min_parts"].append(min(chosen) if chosen else 0)

        return results
