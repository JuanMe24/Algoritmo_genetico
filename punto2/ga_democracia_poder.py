import argparse, random, os
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

# ---------- Utilidades de datos ----------
def gen_curules(P:int=5, total:int=50, rng:random.Random=None) -> List[int]:
    rng = rng or random
    parts = [rng.randint(1, 10) for _ in range(P)]
    s = sum(parts)
    base = [int(round(total * p / s)) for p in parts]
    diff = total - sum(base)
    for i in range(abs(diff)):
        j = i % P
        base[j] += 1 if diff > 0 else -1
    # corregir potenciales negativos
    while any(b < 0 for b in base):
        j = base.index(min(base))
        k = base.index(max(base))
        base[j] += 1; base[k] -= 1
    return base

def gen_pesos(N:int=50, rng:random.Random=None) -> List[int]:
    rng = rng or random
    return [rng.randint(1, 100) for _ in range(N)]

# ---------- Modelo ----------
@dataclass
class Instance:
    P: int
    N: int
    curules: List[int]
    pesos: List[int]

    @property
    def Wtot(self): return int(np.sum(self.pesos))

    def targets(self) -> np.ndarray:
        shares = np.array(self.curules, dtype=float) / float(sum(self.curules))
        return shares * float(self.Wtot)

def party_weights(assign: np.ndarray, pesos: np.ndarray, P: int) -> np.ndarray:
    Wp = np.zeros(P, dtype=float)
    for i,p in enumerate(assign):
        Wp[int(p)] += pesos[i]
    return Wp

def penalty(assign: np.ndarray, inst: Instance) -> float:
    target = inst.targets()
    Wp = party_weights(assign, np.array(inst.pesos, dtype=float), inst.P)
    dev = np.sum(np.abs(Wp - target))
    counts = np.bincount(assign, minlength=inst.P).astype(float)
    avg = np.mean(counts)
    balance = np.sum(np.abs(counts - avg)) * 0.1
    return float(dev + balance)

def fitness(assign: np.ndarray, inst: Instance) -> float:
    pen = penalty(assign, inst)
    return 1.0 / (1.0 + pen)

# ---------- AG ----------
def init_pop(K:int, N:int, P:int, rng:random.Random) -> List[np.ndarray]:
    return [np.array([rng.randrange(P) for _ in range(N)], dtype=int) for __ in range(K)]

def tournament_select(pop: List[np.ndarray], fit: List[float], t:int, rng:random.Random) -> np.ndarray:
    idxs = [rng.randrange(len(pop)) for _ in range(t)]
    best = max(idxs, key=lambda i: fit[i])
    return pop[best].copy()

def uniform_crossover(a: np.ndarray, b: np.ndarray, rng:random.Random) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.array([rng.random() < 0.5 for _ in range(len(a))], dtype=bool)
    c1 = a.copy(); c2 = b.copy()
    c1[mask] = b[mask]
    c2[mask] = a[mask]
    return c1, c2

def mutate(assign: np.ndarray, pm: float, P:int, rng:random.Random) -> np.ndarray:
    for i in range(len(assign)):
        if rng.random() < pm:
            old = assign[i]
            new = rng.randrange(P-1)
            if new >= old: new += 1
            assign[i] = new
    return assign

def run_ga(inst: Instance, K=120, M=500, pm=0.02, t=3, seed=123, plot=False):
    rng = random.Random(seed)
    pop = init_pop(K, inst.N, inst.P, rng)
    fit = [fitness(x, inst) for x in pop]
    best = max(range(K), key=lambda i: fit[i])
    best_sol = pop[best].copy()
    best_fit = fit[best]
    hist = [best_fit]

    for _ in range(M):
        new_pop = []
        while len(new_pop) < K:
            a = tournament_select(pop, fit, t, rng)
            b = tournament_select(pop, fit, t, rng)
            c1, c2 = uniform_crossover(a, b, rng)
            c1 = mutate(c1, pm, inst.P, rng)
            c2 = mutate(c2, pm, inst.P, rng)
            new_pop.extend([c1, c2])
        pop = new_pop[:K]
        fit = [fitness(x, inst) for x in pop]
        i = max(range(K), key=lambda i: fit[i])
        if fit[i] > best_fit:
            best_fit = fit[i]; best_sol = pop[i].copy()
        hist.append(best_fit)

    target = inst.targets()
    Wp = party_weights(best_sol, np.array(inst.pesos, dtype=float), inst.P)
    dev = np.sum(np.abs(Wp - target))

    if plot:
        import matplotlib.pyplot as plt
        from pathlib import Path
        outdir = Path(__file__).resolve().parent / "outputs"
        outdir.mkdir(parents=True, exist_ok=True)

        # objetivo vs asignado
        plt.figure()
        idx = np.arange(inst.P)
        plt.bar(idx-0.15, target, width=0.3, label="Objetivo")
        plt.bar(idx+0.15, Wp, width=0.3, label="Asignado")
        plt.xticks(idx, [f"P{p}" for p in range(inst.P)])
        plt.xlabel("Partidos")
        plt.ylabel("Poder (peso total)")
        plt.title("Objetivo vs Asignado")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "objetivo_vs_asignado.png", dpi=160)

        # fitness
        plt.figure()
        plt.plot(hist, linewidth=2)
        plt.xlabel("Generación")
        plt.ylabel("Mejor fitness")
        plt.title("Evolución de fitness")
        plt.tight_layout()
        plt.savefig(outdir / "fitness.png", dpi=160)

    return best_sol, best_fit, hist, target, Wp, dev

def main():
    parser = argparse.ArgumentParser(description="AG para distribución proporcional de poder (Ejercicio 2)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--M", type=int, default=500)
    parser.add_argument("--K", type=int, default=120)
    parser.add_argument("--pm", type=float, default=0.02)
    parser.add_argument("--pt", type=int, default=3, help="tamaño del torneo")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    P = 5; N = 50
    cur = gen_curules(P=P, total=50, rng=rng)
    pes = gen_pesos(N=N, rng=rng)
    inst = Instance(P=P, N=N, curules=cur, pesos=pes)

    best_sol, best_fit, hist, target, Wp, dev = run_ga(inst, K=args.K, M=args.M, pm=args.pm, t=args.pt, seed=args.seed, plot=args.plot)

    print("Curules por partido:", cur)
    print("Suma de pesos (Wtot):", inst.Wtot)
    print("Objetivo por partido:", [round(x,2) for x in target.tolist()])
    print("Asignado por partido:", [round(x,2) for x in Wp.tolist()])
    print("Desviación total:", round(float(dev), 2))
    print("Fitness:", round(float(best_fit), 6))
    print("Asignación (partido para cada entidad 0..4):")
    print(best_sol.tolist())

if __name__ == "__main__":
    main()
