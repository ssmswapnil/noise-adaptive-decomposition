"""
run_benchmark.py — End-to-end benchmark: noise-aware vs default transpilation
===============================================================================

Fixed version:
  - Uses noiseless AerSimulator for ideal distributions (handles partial measurement)
  - Two-phase transpilation so CCX replacement actually happens
  - Verbose logging to show decomposition choices and gate count differences

Usage:
    python -m benchmarks.run_benchmark
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from src.decompositions import get_all_decompositions, verify_all_decompositions
from src.noise_extractor import extract_noise_profile, print_noise_summary
from src.cost_function import rank_all_decompositions, print_ranking
from src.transpiler_pass import transpile_with_noise_awareness


# ════════════════════════════════════════════════════════════
#  Test circuit generators
# ════════════════════════════════════════════════════════════

def make_toffoli_test_circuit() -> QuantumCircuit:
    """
    Simple 3-qubit test: |1,1,0> → CCX → |1,1,1>
    Ideal outcome: '111' with 100% probability.
    """
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.x(1)
    qc.ccx(0, 1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def make_multi_toffoli_circuit() -> QuantumCircuit:
    """
    3 consecutive Toffoli gates to amplify noise differences.
    |1,1,0> → CCX → CCX → CCX → |1,1,1>
    Ideal outcome: '111' with 100% probability.
    """
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.x(1)
    qc.ccx(0, 1, 2)  # |110> → |111>
    qc.ccx(0, 1, 2)  # |111> → |110>
    qc.ccx(0, 1, 2)  # |110> → |111>
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def make_grover_3qubit() -> QuantumCircuit:
    """
    3-qubit Grover's algorithm searching for |11>.
    Uses Toffoli in the oracle.
    Ideal outcome: '11' with high probability on measured qubits 0,1.
    """
    qc = QuantumCircuit(3, 2)

    # Initialize
    qc.h(0)
    qc.h(1)
    qc.x(2)
    qc.h(2)

    # Oracle: mark |11> using Toffoli
    qc.ccx(0, 1, 2)

    # Diffusion operator on qubits 0,1
    qc.h(0)
    qc.h(1)
    qc.x(0)
    qc.x(1)
    qc.h(1)
    qc.cx(0, 1)
    qc.h(1)
    qc.x(0)
    qc.x(1)
    qc.h(0)
    qc.h(1)

    qc.measure([0, 1], [0, 1])
    return qc


def make_double_toffoli_circuit() -> QuantumCircuit:
    """
    Two Toffoli gates on different target qubits.
    |1,1,0,0> → CCX(0,1,2) → CCX(0,1,3) → |1,1,1,1>
    Ideal outcome: '1111'.
    """
    qc = QuantumCircuit(4, 4)
    qc.x(0)
    qc.x(1)
    qc.ccx(0, 1, 2)
    qc.ccx(0, 1, 3)
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    return qc


# ════════════════════════════════════════════════════════════
#  Simulation helpers
# ════════════════════════════════════════════════════════════

def get_ideal_distribution(circuit: QuantumCircuit, shots: int = 100000) -> dict:
    """
    Get the ideal (noiseless) output distribution.

    Uses AerSimulator without noise — this correctly handles circuits
    that only measure a subset of qubits.
    """
    sim = AerSimulator()
    result = sim.run(circuit, shots=shots).result()
    counts = result.get_counts()
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def run_noisy_simulation(
    circuit: QuantumCircuit,
    noise_model: NoiseModel,
    shots: int = 16384,
) -> dict:
    """Run circuit on noisy simulator, return probability distribution."""
    sim = AerSimulator(noise_model=noise_model)
    result = sim.run(circuit, shots=shots).result()
    counts = result.get_counts()
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def compute_hellinger_fidelity(ideal: dict, noisy: dict) -> float:
    """
    Hellinger fidelity: F_H = (Σ √(p_i · q_i))²
    Returns value in [0, 1] where 1 = perfect match.
    """
    all_keys = set(ideal.keys()) | set(noisy.keys())
    fidelity_sum = 0.0
    for key in all_keys:
        p = ideal.get(key, 0.0)
        q = noisy.get(key, 0.0)
        fidelity_sum += np.sqrt(p * q)
    return fidelity_sum ** 2


def count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    """Count all 2-qubit gates (CX, ECR, CZ) in the circuit."""
    ops = circuit.count_ops()
    return sum(ops.get(g, 0) for g in ["cx", "ecr", "cz"])


# ════════════════════════════════════════════════════════════
#  Main benchmark
# ════════════════════════════════════════════════════════════

def run_benchmark():
    print("\n" + "=" * 70)
    print("  NOISE-ADAPTIVE TOFFOLI DECOMPOSITION — BENCHMARK")
    print("=" * 70)

    # ── Step 0: Verify decompositions ──
    print("\n[Step 0] Verifying decomposition catalog...")
    verify_all_decompositions()

    # ── Step 1: Load backend and noise ──
    print("\n[Step 1] Loading backend and extracting noise profile...")
    try:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        backend = FakeSherbrooke()
    except ImportError:
        print("  ERROR: qiskit-ibm-runtime not installed.")
        print("  Install with: pip install qiskit-ibm-runtime")
        return

    noise_profile = extract_noise_profile(backend)
    print_noise_summary(noise_profile)

    noise_model = NoiseModel.from_backend(backend)

    # ── Step 2: Show decomposition rankings ──
    print("\n[Step 2] Ranking decompositions for sample qubit triples...")
    decompositions = get_all_decompositions()

    from src.noise_extractor import get_qubit_triples
    all_triples = get_qubit_triples(noise_profile)
    sample_triples = all_triples[:3] if len(all_triples) >= 3 else all_triples

    for triple in sample_triples:
        rankings = rank_all_decompositions(decompositions, noise_profile, triple)
        print_ranking(rankings, triple)

    # ── Step 3: Define test circuits ──
    print("\n[Step 3] Building test circuits...")
    test_circuits = {
        "Single Toffoli": make_toffoli_test_circuit(),
        "Multi Toffoli (3x)": make_multi_toffoli_circuit(),
        "Double Toffoli (4q)": make_double_toffoli_circuit(),
        "Grover 3-qubit": make_grover_3qubit(),
    }

    for name, qc in test_circuits.items():
        ccx = qc.count_ops().get("ccx", 0)
        print(f"  {name}: {qc.num_qubits} qubits, {ccx} CCX gate(s)")

    # ── Step 4: Transpile and benchmark ──
    print("\n[Step 4] Running benchmarks...\n")

    shots = 16384
    results_summary = []

    header = f"  {'Circuit':<25} {'Method':<22} {'Depth':>6} {'2Q':>5} {'Fidelity':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for circ_name, circuit in test_circuits.items():

        # Ideal distribution (noiseless)
        ideal_dist = get_ideal_distribution(circuit)

        # ── A) Default transpilation ──
        default_pm = generate_preset_pass_manager(
            optimization_level=1, backend=backend
        )
        default_transpiled = default_pm.run(circuit)
        default_depth = default_transpiled.depth()
        default_2q = count_two_qubit_gates(default_transpiled)

        default_probs = run_noisy_simulation(default_transpiled, noise_model, shots)
        default_fid = compute_hellinger_fidelity(ideal_dist, default_probs)

        print(f"  {circ_name:<25} {'Default':<22} {default_depth:>6} {default_2q:>5} {default_fid:>10.4f}")

        # ── B) Noise-aware (allow relative-phase) ──
        print(f"  {'':<25} {'--- Noise-Aware ---':<22}")
        aware_transpiled = transpile_with_noise_awareness(
            circuit, backend, noise_profile,
            optimization_level=1, require_exact=False, verbose=True,
        )
        aware_depth = aware_transpiled.depth()
        aware_2q = count_two_qubit_gates(aware_transpiled)

        aware_probs = run_noisy_simulation(aware_transpiled, noise_model, shots)
        aware_fid = compute_hellinger_fidelity(ideal_dist, aware_probs)

        print(f"  {'':<25} {'Noise-Aware':<22} {aware_depth:>6} {aware_2q:>5} {aware_fid:>10.4f}")

        # ── C) Noise-aware (exact only) ──
        print(f"  {'':<25} {'--- Exact Only ---':<22}")
        exact_transpiled = transpile_with_noise_awareness(
            circuit, backend, noise_profile,
            optimization_level=1, require_exact=True, verbose=True,
        )
        exact_depth = exact_transpiled.depth()
        exact_2q = count_two_qubit_gates(exact_transpiled)

        exact_probs = run_noisy_simulation(exact_transpiled, noise_model, shots)
        exact_fid = compute_hellinger_fidelity(ideal_dist, exact_probs)

        print(f"  {'':<25} {'Noise-Aware (exact)':<22} {exact_depth:>6} {exact_2q:>5} {exact_fid:>10.4f}")

        # ── Improvements ──
        imp_aware = aware_fid - default_fid
        imp_exact = exact_fid - default_fid
        gap = max(1 - default_fid, 1e-10)
        cx_saved = default_2q - aware_2q

        print(f"  {'':<25} {'Δ Noise-Aware':<22} {'':>6} {f'{-cx_saved:+d}':>5} {imp_aware:>+10.4f} ({100*imp_aware/gap:+.1f}% of gap)")
        print(f"  {'':<25} {'Δ Exact':<22} {'':>6} {f'{-(default_2q - exact_2q):+d}':>5} {imp_exact:>+10.4f} ({100*imp_exact/gap:+.1f}% of gap)")
        print()

        results_summary.append({
            "circuit": circ_name,
            "default_fidelity": default_fid,
            "aware_fidelity": aware_fid,
            "exact_fidelity": exact_fid,
            "default_2q": default_2q,
            "aware_2q": aware_2q,
            "exact_2q": exact_2q,
            "default_depth": default_depth,
            "aware_depth": aware_depth,
            "exact_depth": exact_depth,
        })

    # ── Summary ──
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Circuit':<25} {'Δ Fidelity':>12} {'2Q Saved':>10} {'Default 2Q':>12} {'Aware 2Q':>10}")
    print("  " + "-" * 70)
    for r in results_summary:
        imp = r["aware_fidelity"] - r["default_fidelity"]
        saved = r["default_2q"] - r["aware_2q"]
        print(
            f"  {r['circuit']:<25} {imp:>+12.4f} {saved:>10} "
            f"{r['default_2q']:>12} {r['aware_2q']:>10}"
        )
    print("=" * 70)

    # ── Gate count comparison ──
    print("\n  KEY INSIGHT: 2-qubit gate counts")
    print("  " + "-" * 50)
    for r in results_summary:
        reduction = r["default_2q"] - r["aware_2q"]
        if r["default_2q"] > 0:
            pct = 100 * reduction / r["default_2q"]
        else:
            pct = 0
        print(
            f"  {r['circuit']:<25}: {r['default_2q']} → {r['aware_2q']} "
            f"({reduction} fewer, {pct:.0f}% reduction)"
        )
    print()


if __name__ == "__main__":
    run_benchmark()
