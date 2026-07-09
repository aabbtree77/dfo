#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import minionpy as mpy
import cma

# =========================
# CONFIGURATION
# =========================
D = 20
FUNCTION = 12
BUDGET = 100_000_000
PROGRESS_STEP = 100_000
SEED = 20250306

# =========================
# CEC2022 GLOBAL OPTIMUMS
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
    for f in range(1, 31):
        p = mpy.CEC2022Functions(function_number=f, dimension=D)
        print(f"F{f:02d} @ 0:", p(np.zeros((1, D)))[0])
    print("---------------------\n")


def uniform_sampling(problem, budget):
    best = np.inf

    batch = 4096
    done = 0

    while done < budget:
        n = min(batch, budget - done)
        X = np.random.uniform(-100.0, 100.0, (n, D))
        vals = problem(X)
        b = np.min(vals)
        if b < best:
            best = b
        done += n

    return best


def main():
    np.random.seed(SEED)

    problem = mpy.CEC2022Functions(
        function_number=FUNCTION,
        dimension=D,
    )

    print(f"Problem: CEC2022_F{FUNCTION}")
    print(f"Dimension: {D}")
    print(f"Budget: {BUDGET}")

    fopt = CEC2022_BIASES[FUNCTION]

    print(f"fopt = {fopt:.12e}")

    # sanity_checks(problem)
    # sweep_functions()

    print("\nRunning uniform random sampling baseline...")
    #random_best = uniform_sampling(problem, BUDGET)
    random_best = uniform_sampling(problem, 10000)

    print(f"Uniform sampling best : {random_best:.12e}")
    print(f"Uniform sampling error: {abs(random_best-fopt):.12e}")

    print("\nStarting Hansen BIPOP-CMA-ES...\n")

    lower = -100.0
    upper = 100.0

    evals = 0
    next_progress = PROGRESS_STEP
    best_f = np.inf

    eval_history = []
    err_history = []

    def objective(x):
        nonlocal evals
        nonlocal best_f
        nonlocal next_progress

        f = problem(np.asarray(x).reshape(1, D))[0]

        evals += 1

        if f < best_f:
            best_f = f

        while evals >= next_progress:
            err = abs(best_f - fopt)

            print(
                f"evals={next_progress:10d}  "
                f"best_f={best_f:.12e}  "
                f"error={err:.12e}"
            )

            eval_history.append(next_progress)
            err_history.append(max(err, 1e-300))

            next_progress += PROGRESS_STEP

        return f

    sigma0 = 0.3

    xbest, es = cma.fmin2(
        objective,
        np.zeros(D),
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
    print("Best f  :", best_f)
    print("fopt    :", fopt)
    print("Error   :", abs(best_f - fopt))

    print("\nComparison")
    print("------------------------------")
    print(f"Uniform sampling        : {random_best:.12e}")
    print(f"BIPOP-CMA-ES (pycma)    : {best_f:.12e}")
    print(f"Improvement             : {random_best-best_f:.12e}")

    if eval_history:
        plt.figure(figsize=(8, 5))

        plt.semilogy(
            np.log10(np.asarray(eval_history) / D),
            err_history,
        )

        plt.xlabel("log10(evals / D)")
        plt.ylabel("|f - fopt|")
        plt.title(f"CEC2022 F{FUNCTION}, D={D}")

        plt.grid(True, which="both")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
