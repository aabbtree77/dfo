#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import minionpy as mpy

# =========================
# CONFIGURATION
# =========================
D = 20
FUNCTION = 12
BUDGET = 100_000_000
PROGRESS_STEP = 1000_000
SEED = 20250306

# =========================
# CORRECT CEC2022 BIASES
# =========================
CEC2022_BIASES = {
    1: 300.0,
    2: 400.0,
    3: 600.0,
    4: 800.0,
    5: 900.0,
    6: 1800.0,
    7: 2000.0,
    8: 2200.0,
    9: 2300.0,
    10: 2400.0,
    11: 2600.0,
    12: 2700.0,
}

def sanity_checks(problem):
    print("\n--- Sanity checks ---")
    for _ in range(5):
        x = np.random.uniform(-100, 100, (1, D))
        print("rand:", problem(x))
    print("zero:", problem(np.zeros((1, D))))
    print("---------------------\n")


def sweep_functions():
    print("\n--- Function landscape probe ---")
    for f in range(1, 13):
        p = mpy.CEC2022Functions(function_number=f, dimension=D)
        x = np.zeros((1, D))
        print(f"F{f} @ 0:", p(x)[0])
    print("---------------------\n")


def random_search(problem, n=100000):
    best = np.inf
    for _ in range(n):
        x = np.random.uniform(-100, 100, (1, D))
        best = min(best, problem(x)[0])
    print(f"Random search ({n} evals):", best)


def main():
    np.random.seed(SEED)

    problem = mpy.CEC2022Functions(function_number=FUNCTION, dimension=D)

    print(f"Problem: CEC2022_F{FUNCTION}")
    print(f"Dimension: {D}")
    print(f"Budget: {BUDGET}")

    fopt = CEC2022_BIASES[FUNCTION]
    print(f"fopt = {fopt:.12e}")

    #sanity_checks(problem)
    #sweep_functions()
    #random_search(problem, 100000)

    # =========================
    # CMA-ES OPTIMIZATION
    # =========================
    lower, upper = -100.0, 100.0
    bounds = [(lower, upper)] * D

    evals = 0
    next_progress = PROGRESS_STEP
    best_f = np.inf

    eval_history = []
    err_history = []

    def tracked_batch(X):
        nonlocal evals, best_f, next_progress

        fs = problem(X)
        evals += len(X)

        gen_best = np.min(fs)
        if gen_best < best_f:
            best_f = gen_best

        while evals >= next_progress:
            err = abs(best_f - fopt)
            print(f"evals={next_progress:10d}  best_f={best_f:.12e}  error={err:.12e}")
            eval_history.append(next_progress)
            err_history.append(max(err, 1e-300))
            next_progress += PROGRESS_STEP

        return fs

    initial_step = 0.3

    options = {
        "initial_step": initial_step,
        "bound_strategy": "reflect-random",
        #"population_size": 3000,
        # CMA-ES left at defaults intentionally (important for correctness)
    }

    x0 = [[0.0] * D]

    optimizer = mpy.Minimizer(
        func=tracked_batch,
        x0=x0,
        bounds=bounds,
        algo="CMAES",
        maxevals=BUDGET,
        seed=SEED,
        options=options,
    )

    result = optimizer.optimize()

    print("\nMessage:", result.message)
    print("Best f :", best_f)
    print("fopt   :", fopt)
    print("error  :", abs(best_f - fopt))

    # =========================
    # PLOT
    # =========================
    if eval_history:
        xvals = np.log10(np.asarray(eval_history) / D)

        plt.figure(figsize=(8, 5))
        plt.semilogy(xvals, err_history)
        plt.xlabel("log10(evals / D)")
        plt.ylabel("|f - fopt|")
        plt.title(f"CEC2022 F{FUNCTION}, D={D}")
        plt.grid(True, which="both")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
