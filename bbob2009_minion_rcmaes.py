#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import cocoex
import minionpy as mpy


def main():
    D = 40
    budget = 100_000_000
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
        "initial_step": 0.3,
        "bound_strategy": "reflect-random",
    }

    # ------------------------------------------------------------
    # RCMAES optimizer
    # ------------------------------------------------------------
    optimizer = mpy.RCMAES(
        func=tracked_batch,
        bounds=bounds,
        x0=None,
        maxevals=budget,
        seed=20250306,
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


if __name__ == "__main__":
    main()
