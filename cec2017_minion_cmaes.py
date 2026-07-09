#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import minionpy as mpy

# =========================
# CONFIGURATION
# =========================
D = 50
FUNCTION = 12
BUDGET = 100_000_000
PROGRESS_STEP = 1000_000
SEED = 20250306

# =========================
# CEC2017 GLOBAL OPTIMUMS
# =========================
CEC2017_BIASES = {
    1: 100.0,
    2: 200.0,
    3: 300.0,
    4: 400.0,
    5: 500.0,
    6: 600.0,
    7: 700.0,
    8: 800.0,
    9: 900.0,
    10: 1000.0,
    11: 1100.0,
    12: 1200.0,
    13: 1300.0,
    14: 1400.0,
    15: 1500.0,
    16: 1600.0,
    17: 1700.0,
    18: 1800.0,
    19: 1900.0,
    20: 2000.0,
    21: 2100.0,
    22: 2200.0,
    23: 2300.0,
    24: 2400.0,
    25: 2500.0,
    26: 2600.0,
    27: 2700.0,
    28: 2800.0,
    29: 2900.0,
    30: 3000.0,
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
    for f in range(1, 31):
        p = mpy.CEC2017Functions(function_number=f, dimension=D)
        x = np.zeros((1, D))
        print(f"F{f:02d} @ 0:", p(x)[0])
    print("---------------------\n")


def uniform_sampling(problem, budget):
    """
    Uniform random sampling using exactly the same evaluation
    budget as the optimizer.
    """
    best = np.inf

    batch = 4096
    done = 0

    while done < budget:
        n = min(batch, budget - done)
        X = np.random.uniform(-100.0, 100.0, (n, D))
        fs = problem(X)
        b = np.min(fs)
        if b < best:
            best = b
        done += n

    return best


def main():
    np.random.seed(SEED)

    problem = mpy.CEC2017Functions(
        function_number=FUNCTION,
        dimension=D,
    )

    print(f"Problem: CEC2017_F{FUNCTION}")
    print(f"Dimension: {D}")
    print(f"Budget: {BUDGET}")

    fopt = CEC2017_BIASES[FUNCTION]
    print(f"fopt = {fopt:.12e}")

    # sanity_checks(problem)
    # sweep_functions()

    print("\nRunning uniform random sampling baseline...")
    random_best = uniform_sampling(problem, 10000)

    print(f"Uniform sampling best : {random_best:.12e}")
    print(f"Uniform sampling error: {abs(random_best - fopt):.12e}")

    print("\nStarting CMA-ES...\n")

    # =========================
    # CMA-ES
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

            print(
                f"evals={next_progress:10d} "
                f"best_f={best_f:.12e} "
                f"error={err:.12e}"
            )

            eval_history.append(next_progress)
            err_history.append(max(err, 1e-300))

            next_progress += PROGRESS_STEP

        return fs

    
    # --- CMA-ES configuration (same as original) ---
    options = {
        "population_size": 0,     # default 4 + 3 log(D)
        "mu": 0,                  # default lambda / 2
        "initial_step": 0.3,
        "cc": 0.0,                # defaults internally
        "cs": 0.0,
        "c1": 0.0,
        "cmu": 0.0,
        "damps": 0.0,
        "convergence_tol": 1e-300,
        "bound_strategy": "reflect-random",
    }

    optimizer = mpy.Minimizer(
        func=tracked_batch,
        x0=[[0.0] * D],
        bounds=bounds,
        algo="CMAES",
        maxevals=BUDGET,
        seed=20250305,
        options=options,
    )

    result = optimizer.optimize()
    

    print("\n==============================")
    print("Optimization finished")
    print("==============================")
    print("Message :", result.message)
    print("Best f  :", best_f)
    print("fopt    :", fopt)
    print("Error   :", abs(best_f - fopt))

    print("\nComparison")
    print("------------------------------")
    print(f"Uniform sampling : {random_best:.12e}")
    print(f"CMA-ES           : {best_f:.12e}")


    # =========================
    # Plot
    # =========================

    if eval_history:
        xvals = np.log10(np.asarray(eval_history) / D)

        plt.figure(figsize=(8, 5))
        plt.semilogy(xvals, err_history)

        plt.xlabel("log10(evals / D)")
        plt.ylabel("|f - fopt|")
        plt.title(f"CEC2017 F{FUNCTION}, D={D}")

        plt.grid(True, which="both")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
