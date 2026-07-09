#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cocoex
import cma

# =========================
# CONFIGURATION
# =========================
D = 40
FUNCTION = 24  # BBOB f24
BUDGET = 1000_000
PROGRESS_STEP = 100_000
SEED = 20250306

def uniform_sampling(problem, budget):
    """Simple uniform random sampling baseline"""
    best = np.inf
    batch = 4096
    done = 0
    while done < budget:
        n = min(batch, budget - done)
        X = np.random.uniform(-5.0, 5.0, (n, D))
        vals = [float(problem(x)) for x in X]
        b = min(vals)
        if b < best:
            best = b
        done += n
    return best

def main():
    np.random.seed(SEED)
    
    # BBOB problem via cocoex
    problem = cocoex.BareProblem(
        suite_name="bbob",
        function=FUNCTION,
        dimension=D,
        instance=1,
    )
    
    print(f"Problem: {problem.id}")
    print(f"Dimension: {D}")
    print(f"Budget: {BUDGET}")
    fopt = float(problem.best_value())
    print(f"fopt = {fopt:.12e}")
    
    print("\nRunning uniform random sampling baseline...")
    # Use smaller budget for baseline to keep runtime reasonable
    random_best = uniform_sampling(problem, 20_000)
    print(f"Uniform sampling best : {random_best:.12e}")
    print(f"Uniform sampling error: {abs(random_best - fopt):.12e}")
    
    print("\nStarting Hansen BIPOP-CMA-ES...\n")
    
    lower = -5.0
    upper = 5.0
    evals = 0
    next_progress = PROGRESS_STEP
    best_f = np.inf
    eval_history = []
    err_history = []
    
    def objective(x):
        nonlocal evals, best_f, next_progress
        # cocoex expects a 1D vector
        f = float(problem(np.asarray(x).flatten()))
        evals += 1
        if f < best_f:
            best_f = f
        
        while evals >= next_progress:
            err = abs(best_f - fopt)
            print(
                f"evals={next_progress:10d} "
                f"best_f={best_f:.12e} "
                f"error={err:.12e}"
            )
            eval_history.append(next_progress)
            err_history.append(max(err, 1e-300))
            next_progress += PROGRESS_STEP
        return f
    
    sigma0 = 0.2 * (upper - lower) / 2.0  # reasonable starting sigma
    
    xbest, es = cma.fmin2(
        objective,
        np.zeros(D),          # start at origin (common for BBOB)
        sigma0,
        options={
            "bounds": [lower, upper],
            "maxfevals": BUDGET,
            "seed": SEED,
            "verb_disp": 0,
            "verb_log": 0,
            "verb_filenameprefix": None,
        },
        restarts=np.inf,
        bipop=True,
    )
    
    print("\n==============================")
    print("Optimization finished")
    print("==============================")
    print("Best f :", best_f)
    print("fopt   :", fopt)
    print("Error  :", abs(best_f - fopt))
    
    print("\nComparison")
    print("------------------------------")
    print(f"Uniform sampling : {random_best:.12e}")
    print(f"BIPOP-CMA-ES     : {best_f:.12e}")
    print(f"Improvement      : {random_best - best_f:.12e}")
    
    if eval_history:
        plt.figure(figsize=(8, 5))
        plt.semilogy(
            np.log10(np.asarray(eval_history) / D),
            err_history,
        )
        plt.xlabel(r"$\log_{10}(\#\mathrm{evals}/D)$")
        plt.ylabel(r"$|f - f_{\mathrm{opt}}|$")
        plt.title(f"BBOB f{FUNCTION}, D={D}, BIPOP-CMA-ES (pycma)")
        plt.grid(True, which="both")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
