"""
cost_function.py — Score and rank decompositions using device noise
===================================================================

Implements noise-aware cost functions that estimate total gate error
for a decomposition circuit when mapped to specific physical qubits.

The core idea: for a given qubit triple (e.g., physical qubits 4, 7, 10),
each decomposition in the catalog will accumulate different total error
because its CX gates hit different qubit pairs with different error rates.
This module computes that total error and picks the winner.

Cost models:
    1. Additive gate error  — sum of per-gate error rates (primary)
    2. Weighted model        — 2q gates weighted more heavily (optional)
"""

from qiskit.circuit import QuantumCircuit
from typing import Dict, List, Tuple, Optional

from src.noise_extractor import get_cx_error, get_single_qubit_error


def compute_additive_error_cost(
    circuit: QuantumCircuit,
    noise_profile: Dict,
    physical_qubits: Tuple[int, int, int],
) -> float:
    """
    Additive error cost model.

    Cost = Σ error(gate_i) for every gate in the decomposition circuit.

    The circuit's virtual qubits (0, 1, 2) are mapped to the given
    physical qubits, and each gate's error rate is looked up from the
    noise profile.

    Two-qubit (CX) gates dominate this sum because their error rates
    are typically 10-100x larger than single-qubit gates.

    Args:
        circuit: A 3-qubit decomposition circuit
        noise_profile: From extract_noise_profile()
        physical_qubits: (phys_ctrl0, phys_ctrl1, phys_target) — the
                         physical qubit assignment on the device

    Returns:
        Total estimated error probability (lower = better).
        This is an approximation — real error accumulation is
        multiplicative, but additive is a good first-order model
        when individual error rates are small.
    """
    # Map virtual → physical qubits
    qubit_map = {i: physical_qubits[i] for i in range(3)}
    total_cost = 0.0

    for instruction in circuit.data:
        gate = instruction.operation
        # Get virtual qubit indices for this instruction
        virtual_qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        physical = [qubit_map[q] for q in virtual_qubits]

        if gate.num_qubits == 2:
            pair = (physical[0], physical[1])
            total_cost += get_cx_error(noise_profile, pair)
        elif gate.num_qubits == 1:
            total_cost += get_single_qubit_error(
                noise_profile, physical[0], gate.name
            )

    return total_cost


def compute_weighted_error_cost(
    circuit: QuantumCircuit,
    noise_profile: Dict,
    physical_qubits: Tuple[int, int, int],
    two_qubit_weight: float = 10.0,
) -> float:
    """
    Weighted error cost model.

    Like additive error, but applies a multiplicative weight to
    two-qubit gate errors. This lets you emphasize the dominant
    error source even more aggressively.

    The default weight of 10.0 means: 'a CX error is treated as
    10x more important than a single-qubit error of the same rate.'

    Args:
        circuit: A 3-qubit decomposition circuit
        noise_profile: From extract_noise_profile()
        physical_qubits: Physical qubit assignment
        two_qubit_weight: Multiplier for 2-qubit gate errors

    Returns:
        Weighted total error cost (lower = better)
    """
    qubit_map = {i: physical_qubits[i] for i in range(3)}
    total_cost = 0.0

    for instruction in circuit.data:
        gate = instruction.operation
        virtual_qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        physical = [qubit_map[q] for q in virtual_qubits]

        if gate.num_qubits == 2:
            pair = (physical[0], physical[1])
            total_cost += two_qubit_weight * get_cx_error(noise_profile, pair)
        elif gate.num_qubits == 1:
            total_cost += get_single_qubit_error(
                noise_profile, physical[0], gate.name
            )

    return total_cost


# ════════════════════════════════════════════════════════════
#  Selection
# ════════════════════════════════════════════════════════════

def select_best_decomposition(
    decompositions: Dict,
    noise_profile: Dict,
    physical_qubits: Tuple[int, int, int],
    require_exact: bool = False,
    cost_fn: str = "additive",
) -> Tuple[str, QuantumCircuit, float]:
    """
    Score every decomposition in the catalog and return the best one.

    Args:
        decompositions: From get_all_decompositions()
        noise_profile: From extract_noise_profile()
        physical_qubits: Physical qubit triple
        require_exact: If True, skip relative-phase decompositions
        cost_fn: 'additive' or 'weighted'

    Returns:
        (name, circuit, cost) of the lowest-cost decomposition
    """
    scorer = (
        compute_additive_error_cost if cost_fn == "additive"
        else compute_weighted_error_cost
    )

    best_name = None
    best_circuit = None
    best_cost = float("inf")

    for name, info in decompositions.items():
        if require_exact and not info["exact"]:
            continue

        cost = scorer(info["circuit"], noise_profile, physical_qubits)

        if cost < best_cost:
            best_cost = cost
            best_name = name
            best_circuit = info["circuit"]

    return best_name, best_circuit, best_cost


def rank_all_decompositions(
    decompositions: Dict,
    noise_profile: Dict,
    physical_qubits: Tuple[int, int, int],
    cost_fn: str = "additive",
) -> List[Tuple[str, float, bool]]:
    """
    Rank all decompositions by cost for a given qubit assignment.

    Useful for analysis — see how much spread there is between the
    best and worst options.

    Args:
        decompositions: From get_all_decompositions()
        noise_profile: From extract_noise_profile()
        physical_qubits: Physical qubit triple
        cost_fn: 'additive' or 'weighted'

    Returns:
        List of (name, cost, is_exact) sorted by cost ascending.
    """
    scorer = (
        compute_additive_error_cost if cost_fn == "additive"
        else compute_weighted_error_cost
    )

    rankings = []
    for name, info in decompositions.items():
        cost = scorer(info["circuit"], noise_profile, physical_qubits)
        rankings.append((name, cost, info["exact"]))

    rankings.sort(key=lambda x: x[1])
    return rankings


def print_ranking(
    rankings: List[Tuple[str, float, bool]],
    physical_qubits: Tuple[int, int, int],
) -> None:
    """Pretty-print a decomposition ranking."""
    print(f"\n  Rankings for physical qubits {physical_qubits}:")
    print(f"  {'Rank':<5} {'Decomposition':<28} {'Cost':>10} {'Exact':>6}")
    print("  " + "-" * 52)
    for i, (name, cost, exact) in enumerate(rankings, 1):
        marker = "  ← best" if i == 1 else ""
        print(f"  {i:<5} {name:<28} {cost:>10.6f} {'yes' if exact else 'no':>6}{marker}")

    if len(rankings) >= 2:
        improvement = rankings[-1][1] - rankings[0][1]
        pct = 100 * improvement / rankings[-1][1] if rankings[-1][1] > 0 else 0
        print(f"\n  Potential error reduction: {improvement:.6f} ({pct:.1f}%)")
