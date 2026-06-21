"""
decompositions.py — Catalog of Toffoli (CCX) decompositions
=============================================================

Provides multiple algebraically equivalent decompositions of the Toffoli
gate into CX + single-qubit gates (IBM native basis).

Each decomposition uses different CX qubit-pair patterns and/or gate counts,
making them respond differently to device noise. This is the foundation
for noise-adaptive selection.

Decompositions included:
    1. Standard 6-CX    — Exact unitary, textbook (Barenco et al. 1995)
    2. Relative-phase 3-CX — Approximate (correct classical action), Maslov 2016
    3. Controls-swapped 6-CX — Exact, different CX pair pattern
    4. Controls-swapped 3-CX — Approximate, different CX pair pattern
"""

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator
from typing import Dict, Tuple


# ────────────────────────────────────────────────────────────
#  Decomposition 1: Standard 6-CX Toffoli
# ────────────────────────────────────────────────────────────

def toffoli_6cx_standard() -> QuantumCircuit:

    qc = QuantumCircuit(3, name="toffoli_6cx_std")

    qc.h(2)
    qc.cx(1, 2)
    qc.tdg(2)
    qc.cx(0, 2)
    qc.t(2)
    qc.cx(1, 2)
    qc.tdg(2)
    qc.cx(0, 2)
    qc.t(1)
    qc.t(2)
    qc.h(2)
    qc.cx(0, 1)
    qc.t(0)
    qc.tdg(1)
    qc.cx(0, 1)

    return qc


# ────────────────────────────────────────────────────────────
#  Decomposition 2: Relative-Phase Toffoli (3 CX)
# ────────────────────────────────────────────────────────────

def toffoli_3cx_relative_phase() -> QuantumCircuit:
    """
    Relative-phase Toffoli using only 3 CX gates.

    This is NOT the exact Toffoli unitary — it introduces relative phases
    on some computational basis states. However, the classical action is
    correct: the target qubit flips if and only if both controls are |1>.

    This means it's a valid drop-in replacement for algorithms where only
    the measurement outcome matters (Grover's, QAOA, etc.), but not for
    algorithms that depend on interference between branches.

    CX pairs used: (1,2), (0,2), (1,2)
      → Qubit pair (1,2) appears 2 times (dominant error contributor)
      → Qubit pair (0,2) appears 1 time

    Reference: Maslov, "Advantages of using relative-phase Toffoli gates" (2016)
               Equivalent to Qiskit's RCCXGate
    """
    qc = QuantumCircuit(3, name="toffoli_3cx_rphase")

    qc.h(2)
    qc.t(2)
    qc.cx(1, 2)
    qc.tdg(2)
    qc.cx(0, 2)
    qc.t(2)
    qc.cx(1, 2)
    qc.tdg(2)
    qc.h(2)

    return qc


# ────────────────────────────────────────────────────────────
#  Decomposition 3: Controls-Swapped 6-CX
# ────────────────────────────────────────────────────────────

def toffoli_6cx_controls_swapped() -> QuantumCircuit:
    """
    Standard 6-CX decomposition with the two control roles exchanged.

    The Toffoli gate is symmetric in its two controls:
        CCX(a, b, t) = CCX(b, a, t)
    So swapping which qubit acts as 'first control' vs 'second control'
    gives an equally valid decomposition — but the CX gates now act on
    DIFFERENT physical qubit pairs. This is where noise-adaptive selection
    pays off.

    CX pairs used: (0,2), (1,2), (0,2), (1,2), (1,0), (1,0)
      → Qubit pair (0,2) appears 2 times (was 2 for pair (1,2) in standard)
      → Qubit pair (1,2) appears 2 times (was 2 for pair (0,2) in standard)
      → Qubit pair (1,0) appears 2 times (was (0,1) in standard)

    On a device where CX(0,2) is cleaner than CX(1,2), this variant
    benefits from the swapped emphasis.
    """
    qc = QuantumCircuit(3, name="toffoli_6cx_swapped")

    qc.h(2)
    qc.cx(0, 2)
    qc.tdg(2)
    qc.cx(1, 2)
    qc.t(2)
    qc.cx(0, 2)
    qc.tdg(2)
    qc.cx(1, 2)
    qc.t(0)
    qc.t(2)
    qc.h(2)
    qc.cx(1, 0)
    qc.t(1)
    qc.tdg(0)
    qc.cx(1, 0)

    return qc


# ────────────────────────────────────────────────────────────
#  Decomposition 4: Controls-Swapped Relative-Phase 3-CX
# ────────────────────────────────────────────────────────────

def toffoli_3cx_relative_phase_swapped() -> QuantumCircuit:
    """
    Relative-phase Toffoli (3 CX) with swapped control roles.

    Same idea as Decomposition 3 but applied to the 3-CX version.
    This shifts which CX pair is used twice.

    CX pairs used: (0,2), (1,2), (0,2)
      → Qubit pair (0,2) appears 2 times (dominant error contributor)
      → Qubit pair (1,2) appears 1 time

    Compare with Decomposition 2: (1,2)×2 + (0,2)×1
    If CX(0,2) is cleaner than CX(1,2), this variant is better.
    If CX(1,2) is cleaner, Decomposition 2 is better.
    """
    qc = QuantumCircuit(3, name="toffoli_3cx_rphase_swap")

    qc.h(2)
    qc.t(2)
    qc.cx(0, 2)
    qc.tdg(2)
    qc.cx(1, 2)
    qc.t(2)
    qc.cx(0, 2)
    qc.tdg(2)
    qc.h(2)

    return qc


# ════════════════════════════════════════════════════════════
#  CATALOG
# ════════════════════════════════════════════════════════════

DECOMPOSITION_CATALOG = {
    "6cx_standard": {
        "fn": toffoli_6cx_standard,
        "cx_count": 6,
        "exact": True,
        "description": "Standard 6-CX Toffoli (Barenco et al. 1995)",
        "cx_pairs": [(1, 2), (0, 2), (1, 2), (0, 2), (0, 1), (0, 1)],
    },
    "3cx_relative_phase": {
        "fn": toffoli_3cx_relative_phase,
        "cx_count": 3,
        "exact": False,
        "description": "Relative-phase 3-CX Toffoli (Maslov 2016)",
        "cx_pairs": [(1, 2), (0, 2), (1, 2)],
    },
    "6cx_controls_swapped": {
        "fn": toffoli_6cx_controls_swapped,
        "cx_count": 6,
        "exact": True,
        "description": "6-CX Toffoli with control roles swapped",
        "cx_pairs": [(0, 2), (1, 2), (0, 2), (1, 2), (1, 0), (1, 0)],
    },
    "3cx_rphase_swapped": {
        "fn": toffoli_3cx_relative_phase_swapped,
        "cx_count": 3,
        "exact": False,
        "description": "Relative-phase 3-CX with swapped controls",
        "cx_pairs": [(0, 2), (1, 2), (0, 2)],
    },
}


def get_all_decompositions() -> Dict[str, dict]:
    """
    Return the full catalog with circuit instances.

    Returns:
        Dict mapping decomposition name to a dict containing:
        - 'fn': the generator function
        - 'circuit': an instantiated QuantumCircuit
        - 'cx_count': number of CX gates
        - 'exact': True if unitary-exact, False if relative-phase
        - 'description': human-readable label
        - 'cx_pairs': list of (control, target) CX qubit pairs used
    """
    result = {}
    for name, info in DECOMPOSITION_CATALOG.items():
        result[name] = {
            **info,
            "circuit": info["fn"](),
        }
    return result


# ════════════════════════════════════════════════════════════
#  VERIFICATION
# ════════════════════════════════════════════════════════════

def verify_decomposition(qc: QuantumCircuit, label: str = "") -> Dict:
    """
    Verify a decomposition against the ideal Toffoli unitary.

    Checks two things:
      1. Exact matrix match (up to global phase)
      2. Computational basis action — does it flip the target
         iff both controls are |1>?

    Args:
        qc: The decomposition circuit (3 qubits)
        label: Optional label for printing

    Returns:
        dict with:
        - 'matrix_match': bool — True if unitary matches CCX up to global phase
        - 'max_error': float — largest element-wise deviation from CCX
        - 'action_correct': bool — True if classical truth table is correct
    """
    from qiskit.circuit.library import CCXGate

    ideal_op = Operator(CCXGate())
    test_op = Operator(qc)

    # Element-wise max deviation
    max_error = float(np.max(np.abs(ideal_op.data - test_op.data)))

    # Check equivalence up to global phase
    # If U_test = e^{iφ} U_ideal, then U_test / U_ideal has constant phase
    product = test_op.data @ np.conj(ideal_op.data.T)
    # For unitary equivalence up to global phase, product should be e^{iφ} I
    diag = np.diag(product)
    if np.all(np.abs(diag) > 1e-10):
        phases = np.angle(diag)
        phase_spread = float(np.max(phases) - np.min(phases))
        # Wrap phase differences
        phase_spread = min(phase_spread, 2 * np.pi - phase_spread)
        matrix_match = phase_spread < 1e-6
    else:
        matrix_match = max_error < 1e-6

    # Check computational basis action (truth table)
    # Toffoli: |a, b, c> → |a, b, c XOR (a AND b)>
    # Qiskit uses LSB ordering: qubit 0 is rightmost bit
    action_correct = True
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                # Qiskit LSB: state index = c * 4 + b * 2 + a
                # Wait — need to be careful about Qiskit's qubit ordering.
                # In Qiskit, qubit 0 is the LEAST significant bit.
                # Our circuit: qubit 0 = ctrl0, qubit 1 = ctrl1, qubit 2 = target
                # State |q2, q1, q0> has index = q0 + 2*q1 + 4*q2
                idx_in = a + 2 * b + 4 * c  # a=q0(ctrl0), b=q1(ctrl1), c=q2(target)
                c_out = c ^ (a & b)
                idx_out = a + 2 * b + 4 * c_out

                col = test_op.data[:, idx_in]
                max_row = int(np.argmax(np.abs(col)))
                if max_row != idx_out or np.abs(col[idx_out]) < 0.99:
                    action_correct = False

    return {
        "label": label,
        "matrix_match": matrix_match,
        "max_error": max_error,
        "action_correct": action_correct,
    }


def verify_all_decompositions() -> None:
    """
    Verify every decomposition in the catalog and print results.
    Run this to confirm all circuits are implemented correctly.
    """
    catalog = get_all_decompositions()

    print("=" * 70)
    print("  DECOMPOSITION VERIFICATION REPORT")
    print("=" * 70)

    for name, info in catalog.items():
        result = verify_decomposition(info["circuit"], label=name)
        status_matrix = "PASS" if (result["matrix_match"] or not info["exact"]) else "FAIL"
        status_action = "PASS" if result["action_correct"] else "FAIL"

        print(f"\n  {name}")
        print(f"    Description : {info['description']}")
        print(f"    CX count    : {info['cx_count']}")
        print(f"    Exact       : {info['exact']}")
        print(f"    Matrix match: {result['matrix_match']} ({status_matrix})")
        print(f"    Max error   : {result['max_error']:.2e}")
        print(f"    Action OK   : {result['action_correct']} ({status_action})")
        print(f"    CX pairs    : {info['cx_pairs']}")

    print("\n" + "=" * 70)


# ────────────────────────────────────────────────────────────
#  CLI entry point
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    verify_all_decompositions()
