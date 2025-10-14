#!/usr/bin/env python3
import argparse
import math
import random
from typing import List, Tuple
import numpy as np
import os

def f(x: float) -> float:
    """Fitness: f(x) = x * sin(10πx) + 1, domain [0, 1]."""
    return x * math.sin(10 * math.pi * x) + 1.0

def decode(bits: List[int]) -> float:
    """Map a binary chromosome (bits) to x in [0, 1]."""
    l = len(bits)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    maxv = (1 << l) - 1
    return val / maxv if maxv > 0 else 0.0

def random_chromosome(l: int) -> List[int]:
    return [random.randint(0, 1) for _ in range(l)]

def init_population(K: int, l: int) -> List[List[int]]:
    return [random_chromosome(l) for _ in range(K)]

def evaluate_population(pop: List[List[int]]) -> Tuple[np.ndarray, np.ndarray, float]:
    xs = np.array([decode(c) for c in pop], dtype=float)
    fitness = np.array([f(x) for x in xs], dtype=float)
    total = float(np.sum(fitness))
    return xs, fitness, total

def roulette_select(pop: List[List[int]], fitness: np.ndarray, total: float) -> List[List[int]]:
    if total <= 1e-12:
        # Avoid degeneracy: uniform selection if fitness sums to ~0
        return random.sample(pop, k=len(pop))
    probs = fitness / total
    cum = np.cumsum(probs)
    selected = []
    for _ in range(len(pop)):
        r = random.random()
        idx = int(np.searchsorted(cum, r, side="left"))
        selected.append(pop[idx].copy())
    return selected

def one_point_crossover(pop: List[List[int]], pc: float) -> List[List[int]]:
    random.shuffle(pop)
    offspring = []
    for i in range(0, len(pop), 2):
        p1 = pop[i]
        p2 = pop[(i+1) % len(pop)]
        if random.random() < pc and len(p1) > 1:
            pt = random.randint(1, len(p1)-1)
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        else:
            c1, c2 = p1.copy(), p2.copy()
        offspring.extend([c1, c2])
    return offspring[:len(pop)]

def bitflip_mutation(pop: List[List[int]], pm: float) -> List[List[int]]:
    for c in pop:
        for j in range(len(c)):
            if random.random() < pm:
                c[j] = 1 - c[j]
    return pop

def run_ga(l=16, K=80, M=200, pc=0.9, seed=42, plot=False):
    random.seed(seed)
    np.random.seed(seed)

    pm = 1.0 / l
    pop = init_population(K, l)

    best_hist = []
    best = (-1.0, None, None)  # (fitness, x, chromosome)

    for gen in range(M):
        xs, fit, total = evaluate_population(pop)

        # Track best
        idx = int(np.argmax(fit))
        if fit[idx] > best[0]:
            best = (float(fit[idx]), float(xs[idx]), pop[idx].copy())
        best_hist.append(float(np.max(fit)))

        # Selection → Crossover → Mutation
        parents = roulette_select(pop, fit, total)
        children = one_point_crossover(parents, pc)
        pop = bitflip_mutation(children, pm)

    if plot:
        import matplotlib.pyplot as plt
        os.makedirs("outputs", exist_ok=True)
        plt.figure()
        plt.plot(best_hist, linewidth=2)
        plt.xlabel("Generation")
        plt.ylabel("Best fitness")
        plt.title("GA on f(x) = x sin(10πx) + 1")
        plt.tight_layout()
        plt.savefig("outputs/best_fitness.png", dpi=180)

    return best[1], best[0], best_hist

def main():
    parser = argparse.ArgumentParser(description="GA to maximize f(x) = x sin(10πx) + 1 on [0,1]")
    parser.add_argument("--l", type=int, default=16, help="chromosome length (bits)")
    parser.add_argument("--K", type=int, default=80, help="population size")
    parser.add_argument("--M", type=int, default=200, help="generations")
    parser.add_argument("--pc", type=float, default=0.9, help="crossover probability")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--plot", action="store_true", help="save best fitness plot")
    args = parser.parse_args()

    x_best, f_best, _ = run_ga(l=args.l, K=args.K, M=args.M, pc=args.pc, seed=args.seed, plot=args.plot)
    print(f"Best x ≈ {x_best:.6f} | f(x) ≈ {f_best:.6f}")

if __name__ == "__main__":
    main()
