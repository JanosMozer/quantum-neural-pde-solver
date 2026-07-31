"""Run both classical baseline generators' self-tests together.

Each baseline module is independently runnable (`python -m qt_pinn.baselines.low_rank`,
`python -m qt_pinn.baselines.mps`); this just gives one entry point that checks both at once.
"""

import subprocess
import sys


def run(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True, text=True,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr)
        raise AssertionError(f"{module} failed")


if __name__ == "__main__":
    run("qt_pinn.baselines.low_rank")
    run("qt_pinn.baselines.mps")
    print("PASS: both classical baseline generators pass self-tests at two dummy target sizes")
