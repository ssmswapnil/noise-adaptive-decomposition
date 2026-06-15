# Noise-Adaptive Toffoli Decomposition for IBM Quantum Backends

A noise-aware gate decomposition strategy for Qiskit's transpiler. Instead of using a fixed Toffoli (CCX) decomposition for all backends, this project selects the optimal decomposition based on the device's real-time calibration data — choosing the variant that minimizes total gate error for the specific physical qubits being used.

## Motivation

IBM quantum devices are noisy, and crucially, **noise varies across the chip**. The CX gate between qubits (0,1) might have 0.5% error while CX between (3,4) has 2.1% error. The standard Toffoli gate admits multiple valid decompositions into CX + single-qubit gates, each using different CX qubit pairs. By choosing the decomposition whose CX pairs happen to fall on the lowest-error connections, we can improve circuit fidelity without any hardware changes.

## Architecture

```
src/
├── decompositions.py     # Catalog of 4 Toffoli decompositions (6-CX, 3-CX, swapped variants)
├── noise_extractor.py    # Extract gate errors, T1/T2 from IBM backends
├── cost_function.py      # Score decompositions using additive error model
└── transpiler_pass.py    # Custom Qiskit TransformationPass + pipeline integration

benchmarks/
└── run_benchmark.py      # End-to-end comparison: default vs noise-aware transpilation

tests/
└── test_decompositions.py  # Verify unitary correctness of all decompositions
```

## Decomposition Catalog

| Name | CX Count | Exact | Description |
|------|----------|-------|-------------|
| `6cx_standard` | 6 | Yes | Textbook decomposition (Barenco et al. 1995) |
| `3cx_relative_phase` | 3 | No* | Relative-phase Toffoli (Maslov 2016) |
| `6cx_controls_swapped` | 6 | Yes | Standard decomposition with control roles swapped |
| `3cx_rphase_swapped` | 3 | No* | Relative-phase with swapped controls |

*Relative-phase decompositions have correct computational basis action but introduce phases on non-|11⟩ states. Valid for measurement-based algorithms (Grover, QAOA).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Verify decompositions are correct
python tests/test_decompositions.py

# Run the full benchmark
python benchmarks/run_benchmark.py
```

## How It Works

1. **Extract noise**: Pull per-gate error rates from the backend's calibration data
2. **Score**: For each candidate decomposition, sum up the error rates of all its gates when mapped to specific physical qubits
3. **Select**: Pick the decomposition with the lowest total error
4. **Integrate**: A custom `TransformationPass` does this automatically during transpilation

```python
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from src.noise_extractor import extract_noise_profile
from src.transpiler_pass import transpile_with_noise_awareness

backend = FakeSherbrooke()
profile = extract_noise_profile(backend)

# Your circuit with Toffoli gates
optimized = transpile_with_noise_awareness(circuit, backend, profile)
```

## Key Concepts

- **Gate decomposition**: Breaking high-level gates (CCX) into native hardware gates (CX, SX, RZ)
- **Noise-aware compilation**: Using device calibration data to guide compilation decisions
- **Custom transpiler passes**: Extending Qiskit's modular transpilation pipeline
- **Benchmarking with Aer**: Noisy simulation to validate fidelity improvements

## References

- Barenco et al., "Elementary gates for quantum computation" (1995)
- Maslov, "Advantages of using relative-phase Toffoli gates" (2016)
- Qiskit Transpiler Documentation: https://docs.quantum.ibm.com/guides/transpile

## Author

Sai Swapnil Kumar Mishra — IIT (ISM) Dhanbad, Engineering Physics
