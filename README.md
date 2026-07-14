## Derivative-free Optimization

This is the code for the issues [#7](https://github.com/khoirulmuzakka/Minion/issues/7) and others, including [#11](https://github.com/khoirulmuzakka/Minion/issues/11).

```bash

git clone https://github.com/aabbtree77/dfo.git
cd dfo

# Create virtual environment
uv venv

# Activate
source .venv/bin/activate

# Install packages
uv pip install \
    numpy \
    scipy \
    matplotlib \
    cma \
    coco-experiment \
    ipython \
    minionpy
```

BBOB-2009 is in `coco-experiment`, while CEC-20xx are automatically included in [minionpy](https://github.com/khoirulmuzakka/Minion), created by Dr. Khoirul Faiq Muzakka, who is a cool physicist. 

Minion includes and updates the state of the art algorithms from the most recent CEC competitions organized by [Dr. Ponnuthurai Nagaratnam Suganthan and others](https://github.com/P-N-Suganthan/2025-CEC/tree/main). 

I do not believe in numerous differential evolutions (DEs), including their combinations with the CMAES family (e.g. EA4Eig), but check out Minion's RCMAES which is worth trying before "fine tuning" with Nikolaus Hansen's BIPOP-aCMAES in [pycma](https://github.com/CMA-ES/pycma). The latter is highly optimized and nowhere close to what is in the tutorials, various older versions, and alternative implementations.

Minion is often updated, to upgrade:

```bash
uv pip install --upgrade minionpy
```

## Additional Notes

Hansen's pycma BIPOP-aCMAES is the best. Nonetheless, on very hard problems such as the composite functions in CEC2017 (F26 - F30), CEC2022 F12 (or even CEC2017 F20, which it solves), the algorithm struggles (but so do all the others):

```bash
evals=   3000000  best_f=2.021369686143e+03  error=2.136968614255e+01
NOTE (module=cma, iteration=1563):  
condition in coordinate system exceeded 1.0e+08, rescaled to 1.0e+00, 
condition changed from 7.6e+08 to 7.6e+04
/home/tokyo/f18/.venv/lib/python3.10/site-packages/cma/utilities/utils.py:369: UserWarning: 
        geno-pheno transformation introduced based on the
        current covariance matrix with condition 1.0e+12 -> 1.0e+00,
        injected solutions become "invalid" in this iteration (time=Jul  9 22:01:48 2026 class=CMAEvolutionStrategy method=alleviate_conditioning iteration=2508)
  warnings.warn(msg + ' (time={}'.format(time.asctime()[4:]) +
/home/tokyo/f18/.venv/lib/python3.10/site-packages/cma/utilities/utils.py:369: UserWarning: 
        geno-pheno transformation introduced based on the
        current covariance matrix with condition 1.0e+12 -> 1.0e+00,
        injected solutions become "invalid" in this iteration (time=Jul  9 22:02:51 2026 class=CMAEvolutionStrategy method=alleviate_conditioning iteration=1509)
  warnings.warn(msg + ' (time={}'.format(time.asctime()[4:]) +
evals=   4000000  best_f=2.021369686143e+03  error=2.136968614255e+01
```

I have not yet encountered a problem which is not solvable by BIPOP-aCMAES and solvable by any DE/PSO.

TBC...

