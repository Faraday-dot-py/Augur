"""Run scatter_conservation.py or scatter_regress.py of branch scatter-core-drift-armA (0cb0471) on fresh held-out scene seeds.
Usage: PYTHONPATH=. python scripts/scatter_core_drift_fresh_seeds.py conservation|regress <the target script's own arguments>"""
import sys

from scripts import scatter_conservation as sc
from scripts import scatter_regress as sr

FRESH_SEEDS = (12100, 12200, 12300)


def main():
    target = {"conservation": sc, "regress": sr}[sys.argv.pop(1)]
    sc.SEEDS = FRESH_SEEDS
    sr.SEEDS = FRESH_SEEDS
    target.main()


if __name__ == "__main__":
    main()
