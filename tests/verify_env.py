"""Gate 0: confirm the environment actually works on this machine before building anything on top of it."""

import os
import sys


def main() -> None:
    print(f"python: {sys.version}")

    import torch
    print(f"torch: {torch.__version__}  cuda available: {torch.cuda.is_available()}")

    import pennylane as qml
    print(f"pennylane: {qml.__version__}")

    import importlib.metadata
    print(f"pennylane-lightning: {importlib.metadata.version('pennylane-lightning')}")

    import quimb
    print(f"quimb: {quimb.__version__}")

    import scipy
    print(f"scipy: {scipy.__version__}")

    import numpy
    print(f"numpy: {numpy.__version__}")

    import matplotlib
    print(f"matplotlib: {matplotlib.__version__}")

    import yaml
    print(f"pyyaml: {yaml.__version__}")

    import qt_pinn
    print(f"qt_pinn package importable: ok (editable install)")

    print(f"cpu count: {os.cpu_count()}")

    dev = qml.device("default.qubit", wires=6)

    @qml.qnode(dev)
    def smoke():
        qml.Hadamard(wires=0)
        return qml.state()

    state = smoke()
    assert state.shape == (64,), f"expected 64-dim state, got {state.shape}"
    print("PASS: default.qubit smoke test ok")


if __name__ == "__main__":
    main()
