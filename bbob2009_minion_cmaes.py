#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import cocoex
import minionpy as mpy

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
    D = 40
    budget = 1000_000
    progress_step = 100_000

    problem = cocoex.BareProblem(
        suite_name="bbob",
        function=24,
        dimension=D,
        instance=1,
    )

    print("Problem:", problem.id)
    print("Dimension:", problem.dimension)
    print("Budget:", budget)
    
    print("\nRunning uniform random sampling baseline...")
    random_best = uniform_sampling(problem, 10000)

    print(f"Uniform sampling best : {random_best:.12e}")
    print(f"Uniform sampling error: {abs(random_best - fopt):.12e}")

    print("\nStarting CMA-ES...\n")

    fopt = float(problem.best_value())

    bounds = [(-5.0, 5.0)] * D

    # ------------------------------------------------------------
    # evaluation tracking (COCO-compliant)
    # ------------------------------------------------------------
    evals = 0
    next_progress = progress_step

    best_f = np.inf
    eval_history = []
    err_history = []

    def tracked_batch(X):
        nonlocal evals, best_f, next_progress

        fs = [float(problem(np.asarray(x))) for x in X]

        evals += len(X)

        gen_best = min(fs)
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

            next_progress += progress_step

        return fs

    # ------------------------------------------------------------
    # RCMAES options (key difference: built-in restarts)
    # ------------------------------------------------------------
    options = {
        "population_size": 0,   # auto (4 + 3 log D)
        "initial_step": 0.2,
        "bound_strategy": "reflect-random",
    }

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
        x0=None,
        bounds=bounds,
        algo="CMAES",
        maxevals=budget,
        seed=20250305,
        options=options,
    )

    result = optimizer.optimize()

    # ------------------------------------------------------------
    # final report
    # ------------------------------------------------------------
    final_err = abs(best_f - fopt)

    print("\nFinished")
    print("Evaluations :", evals)
    print("Best f      :", best_f)
    print("fopt        :", fopt)
    print("Final error :", final_err)

    try:
        print("Result f    :", result.fun)
    except Exception:
        pass

    # ------------------------------------------------------------
    # plot
    # ------------------------------------------------------------
    xvals = np.log10(np.asarray(eval_history) / D)

    plt.figure(figsize=(8, 5))
    plt.semilogy(xvals, err_history)

    plt.xlabel(r'$\log_{10}(\#\mathrm{evals}/D)$')
    plt.ylabel(r'$|f - f_{opt}|$')
    plt.title(f'BBOB f18, D={D}, RCMAES')
    plt.grid(True, which='both')

    plt.tight_layout()
    plt.show()
    
        # ------------------------------------------------------------
    # uniform random-search baseline
    # ------------------------------------------------------------
    rng = np.random.default_rng(20250306)

    Xrand = rng.uniform(
        low=-5.0,
        high=5.0,
        size=(budget, D),
    )

    random_best = min(
        float(problem(x))
        for x in Xrand
    )

    random_err = abs(random_best - fopt)

    print("\nUniform random baseline")
    print("Samples     :", budget)
    print("Best f      :", random_best)
    print("Error       :", random_err)
    
    plt.figure(figsize=(8, 5))
    plt.semilogy(xvals, err_history, label="CMA-ES")

    plt.axhline(
        random_err,
        linestyle="--",
        label=f"Uniform random ({budget:,} samples)"
    )

    plt.xlabel(r'$\log_{10}(\#\mathrm{evals}/D)$')
    plt.ylabel(r'$|f - f_{opt}|$')
    plt.title(f'BBOB f18, D={D}, CMA-ES')
    plt.grid(True, which='both')
    plt.legend()


if __name__ == "__main__":
    main()
