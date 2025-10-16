import argparse, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_target(path: str, H: int, W: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)

def init_population(K: int, H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, size=(K, H, W, 3), dtype=np.uint8)

def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))

def fitness(pop: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    errs = np.array([mse(ind, target) for ind in pop], dtype=np.float64)
    return 1.0 / (1.0 + errs), errs

def tournament_select(pop: np.ndarray, fit: np.ndarray, t: int, rng: np.random.Generator) -> np.ndarray:
    idxs = rng.integers(0, len(pop), size=t)
    best = idxs[np.argmax(fit[idxs])]
    return pop[best].copy()

def crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    H, W, _ = a.shape
    h = rng.integers(max(1, H//4), max(2, 3*H//4))
    w = rng.integers(max(1, W//4), max(2, 3*W//4))
    y = rng.integers(0, H - h + 1)
    x = rng.integers(0, W - w + 1)
    c1, c2 = a.copy(), b.copy()
    c1[y:y+h, x:x+w, :] = b[y:y+h, x:x+w, :]
    c2[y:y+h, x:x+w, :] = a[y:y+h, x:x+w, :]
    return c1, c2

def mutate(ind: np.ndarray, pm: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    H, W, C = ind.shape
    mask = rng.random((H, W, 1)) < pm
    noise = rng.normal(0.0, sigma, size=(H, W, C))
    out = ind.astype(np.int32)
    out = np.where(mask, np.clip(out + noise, 0, 255), out)
    return out.astype(np.uint8)

def run_ga(target: np.ndarray, K=50, M=300, pm=0.01, sigma=12.0, t=3, seed=123, plot=False, save_every=0, outdir="outputs"):
    rng = np.random.default_rng(seed)
    H, W, _ = target.shape
    pop = init_population(K, H, W, rng)
    fit, errs = fitness(pop, target)
    best_idx = int(np.argmax(fit))
    best = pop[best_idx].copy()
    best_fit = float(fit[best_idx])
    history = [best_fit]

    os.makedirs(outdir, exist_ok=True)

    for gen in range(1, M+1):
        new_pop = []
        while len(new_pop) < K:
            a = tournament_select(pop, fit, t, rng)
            b = tournament_select(pop, fit, t, rng)
            c1, c2 = crossover(a, b, rng)
            c1 = mutate(c1, pm, sigma, rng)
            c2 = mutate(c2, pm, sigma, rng)
            new_pop.extend([c1, c2])
        pop = np.array(new_pop[:K], dtype=np.uint8)
        fit, errs = fitness(pop, target)
        i = int(np.argmax(fit))
        if fit[i] > best_fit:
            best_fit = float(fit[i])
            best = pop[i].copy()
        history.append(best_fit)

        if save_every and gen % save_every == 0:
            Image.fromarray(best).save(os.path.join(outdir, f"best_gen_{gen:04d}.png"))

    if plot:
        plt.figure()
        plt.plot(history, linewidth=2)
        plt.xlabel("Generation")
        plt.ylabel("Best fitness")
        plt.title("GA — Image Evolution")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fitness.png"), dpi=160)

    Image.fromarray(best).save(os.path.join(outdir, "best_final.png"))
    return best, history

def main():
    import argparse
    ap = argparse.ArgumentParser(description="GA — Evolución de imágenes RGB (Ejercicio 4)")
    ap.add_argument("--target", type=str, required=True, help="Ruta a la imagen objetivo")
    ap.add_argument("--H", type=int, default=96, help="Alto")
    ap.add_argument("--W", type=int, default=96, help="Ancho")
    ap.add_argument("--K", type=int, default=50, help="Población")
    ap.add_argument("--M", type=int, default=300, help="Generaciones")
    ap.add_argument("--pm", type=float, default=0.01, help="Prob. de mutación por píxel")
    ap.add_argument("--sigma", type=float, default=12.0, help="Sigma mutación")
    ap.add_argument("--pt", type=int, default=3, help="Tamaño torneo (no usado explícitamente aquí)")
    ap.add_argument("--plot", action="store_true", help="Guardar fitness.png")
    ap.add_argument("--save-every", type=int, default=0, help="Guardar mejor cada N generaciones")
    ap.add_argument("--seed", type=int, default=123, help="Semilla RNG")
    args = ap.parse_args()

    target = load_target(args.target, args.H, args.W)
    best, history = run_ga(target, K=args.K, M=args.M, pm=args.pm, sigma=args.sigma, t=args.pt,
                           seed=args.seed, plot=args.plot, save_every=args.save_every, outdir="outputs")
    print("Guardado mejor individuo en outputs/best_final.png")

if __name__ == "__main__":
    main()
