"""
noise_extractor.py — Extract noise profiles from IBM quantum backends
======================================================================

Pulls gate error rates, gate durations, and T1/T2 decoherence times
from a backend's Target object (modern Qiskit 1.x API).

Works with both real IBM backends and fake (simulator) backends from
qiskit_ibm_runtime.fake_provider.

Usage:
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    from src.noise_extractor import extract_noise_profile, print_noise_summary

    backend = FakeSherbrooke()
    profile = extract_noise_profile(backend)
    print_noise_summary(profile)
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional


def extract_noise_profile(backend) -> Dict[str, Any]:
    """
    Extract a structured noise profile from an IBM backend.

    Reads the backend's Target object to pull per-gate, per-qubit
    error rates and durations.

    Args:
        backend: Any Qiskit BackendV2-compatible backend
                 (real IBM backend or FakeBackend)

    Returns:
        Dict with keys:
        - 'cx_errors':      {(ctrl, tgt): float}  — 2-qubit gate error rates
        - 'sx_errors':      {qubit: float}         — SX gate error rates
        - 'rz_errors':      {qubit: float}         — RZ gate error rates
        - 'x_errors':       {qubit: float}         — X gate error rates
        - 'cx_durations':   {(ctrl, tgt): float}   — 2-qubit gate durations (seconds)
        - 't1':             {qubit: float}          — T1 relaxation times (seconds)
        - 't2':             {qubit: float}          — T2 dephasing times (seconds)
        - 'num_qubits':     int
        - 'backend_name':   str
        - 'two_qubit_gate_name': str               — which 2q gate the backend uses
    """
    target = backend.target
    num_qubits = backend.num_qubits

    profile = {
        "cx_errors": {},
        "sx_errors": {},
        "rz_errors": {},
        "x_errors": {},
        "cx_durations": {},
        "t1": {},
        "t2": {},
        "num_qubits": num_qubits,
        "backend_name": backend.name,
    }

    # ── Identify the native 2-qubit gate ──
    # IBM backends use 'ecr' (newer), 'cx' (older), or 'cz'
    two_qubit_gate = None
    for gate_name in ["ecr", "cx", "cz"]:
        if gate_name in target.operation_names:
            two_qubit_gate = gate_name
            break

    profile["two_qubit_gate_name"] = two_qubit_gate or "unknown"

    # ── Extract 2-qubit gate errors and durations ──
    if two_qubit_gate:
        for qargs in target.qargs_for_operation_name(two_qubit_gate):
            props = target[two_qubit_gate][qargs]
            if props is not None:
                pair = tuple(qargs)
                if props.error is not None:
                    profile["cx_errors"][pair] = props.error
                if props.duration is not None:
                    profile["cx_durations"][pair] = props.duration

    # ── Extract single-qubit gate errors ──
    for gate_name in ["sx", "rz", "x"]:
        key = f"{gate_name}_errors"
        if gate_name in target.operation_names:
            for qargs in target.qargs_for_operation_name(gate_name):
                props = target[gate_name][qargs]
                if props is not None and props.error is not None:
                    profile[key][qargs[0]] = props.error

    # ── Extract T1 / T2 from qubit properties ──
    qubit_props = target.qubit_properties
    if qubit_props:
        for q in range(num_qubits):
            if q < len(qubit_props) and qubit_props[q] is not None:
                if qubit_props[q].t1 is not None:
                    profile["t1"][q] = qubit_props[q].t1
                if qubit_props[q].t2 is not None:
                    profile["t2"][q] = qubit_props[q].t2

    return profile


# ════════════════════════════════════════════════════════════
#  Convenience lookups
# ════════════════════════════════════════════════════════════

def get_cx_error(profile: Dict, qubit_pair: Tuple[int, int]) -> float:
    """
    Look up the CX error rate for a qubit pair.

    IBM backends support CX in only one direction for each pair.
    This function checks both (a, b) and (b, a) and returns the
    first match. If neither exists (qubits not connected), returns
    1.0 as a maximum penalty so the cost function naturally avoids
    that pair.

    Args:
        profile: From extract_noise_profile()
        qubit_pair: (control_qubit, target_qubit)

    Returns:
        Error rate (float between 0 and 1), or 1.0 if not connected.
    """
    pair = tuple(qubit_pair)
    if pair in profile["cx_errors"]:
        return profile["cx_errors"][pair]
    reverse = (pair[1], pair[0])
    if reverse in profile["cx_errors"]:
        return profile["cx_errors"][reverse]
    return 1.0


def get_single_qubit_error(profile: Dict, qubit: int, gate: str = "sx") -> float:
    """
    Look up single-qubit gate error rate.

    For gates like T, Tdg, S, Sdg, H — these decompose into SX + RZ
    on IBM hardware, so we approximate their error as SX error.
    RZ is a virtual (frame-change) gate with ~0 error on IBM devices.

    Args:
        profile: From extract_noise_profile()
        qubit: Physical qubit index
        gate: Gate name ('sx', 'rz', 'x', 'h', 't', 'tdg', 's', 'sdg')

    Returns:
        Error rate (float), or 0.0 if not found.
    """
    # Map composite gates to their dominant native-gate error
    gate_map = {
        "h": "sx",    # H ≈ RZ + SX + RZ
        "t": "rz",    # T is a phase gate ≈ virtual RZ
        "tdg": "rz",
        "s": "rz",
        "sdg": "rz",
        "u": "sx",    # U decomposes to SX + RZ
    }
    lookup_gate = gate_map.get(gate, gate)
    key = f"{lookup_gate}_errors"
    return profile.get(key, {}).get(qubit, 0.0)


def get_connected_pairs(profile: Dict) -> List[Tuple[int, int]]:
    """Return all qubit pairs that have a CX connection."""
    return list(profile["cx_errors"].keys())


def get_qubit_triples(profile: Dict) -> List[Tuple[int, int, int]]:
    """
    Find all 3-qubit combinations where all pairs are CX-connected.

    This is useful for finding valid physical qubit assignments for
    a Toffoli gate without needing SWAP insertions.

    Returns:
        List of (q0, q1, q2) tuples where CX exists between
        all three pairs (possibly in either direction).
    """
    connected = set()
    for pair in profile["cx_errors"].keys():
        connected.add(frozenset(pair))

    triples = []
    qubits = list(range(profile["num_qubits"]))

    for i in range(len(qubits)):
        for j in range(i + 1, len(qubits)):
            for k in range(j + 1, len(qubits)):
                a, b, c = qubits[i], qubits[j], qubits[k]
                if (
                    frozenset([a, b]) in connected
                    and frozenset([b, c]) in connected
                    and frozenset([a, c]) in connected
                ):
                    # All permutations of (a, b, c) as (ctrl0, ctrl1, target)
                    from itertools import permutations
                    for perm in permutations([a, b, c]):
                        triples.append(perm)

    return triples


# ════════════════════════════════════════════════════════════
#  Pretty printing
# ════════════════════════════════════════════════════════════

def print_noise_summary(profile: Dict) -> None:
    """Print a human-readable summary of the noise profile."""
    print("=" * 60)
    print(f"  NOISE PROFILE: {profile['backend_name']}")
    print("=" * 60)
    print(f"  Qubits        : {profile['num_qubits']}")
    print(f"  Native 2q gate: {profile.get('two_qubit_gate_name', 'unknown')}")

    cx_errors = profile["cx_errors"]
    if cx_errors:
        errors = list(cx_errors.values())
        print(f"\n  Two-qubit gate errors ({len(cx_errors)} connections):")
        print(f"    Min  : {min(errors):.6f}")
        print(f"    Max  : {max(errors):.6f}")
        print(f"    Mean : {np.mean(errors):.6f}")
        print(f"    Spread (max/min): {max(errors)/max(min(errors), 1e-15):.1f}x")

        sorted_pairs = sorted(cx_errors.items(), key=lambda x: x[1])
        print(f"\n    Best 5 pairs:")
        for pair, err in sorted_pairs[:5]:
            print(f"      {pair}: {err:.6f}")
        print(f"\n    Worst 5 pairs:")
        for pair, err in sorted_pairs[-5:]:
            print(f"      {pair}: {err:.6f}")

    sx_errors = profile.get("sx_errors", {})
    if sx_errors:
        errs = list(sx_errors.values())
        print(f"\n  SX gate errors ({len(sx_errors)} qubits):")
        print(f"    Min  : {min(errs):.2e}")
        print(f"    Max  : {max(errs):.2e}")
        print(f"    Mean : {np.mean(errs):.2e}")

    t1_vals = profile.get("t1", {})
    t2_vals = profile.get("t2", {})
    if t1_vals:
        t1s = list(t1_vals.values())
        print(f"\n  T1 relaxation times:")
        print(f"    Min  : {min(t1s)*1e6:.1f} μs")
        print(f"    Max  : {max(t1s)*1e6:.1f} μs")
    if t2_vals:
        t2s = list(t2_vals.values())
        print(f"\n  T2 dephasing times:")
        print(f"    Min  : {min(t2s)*1e6:.1f} μs")
        print(f"    Max  : {max(t2s)*1e6:.1f} μs")

    print("=" * 60)


# ────────────────────────────────────────────────────────────
#  CLI entry point
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

    backend = FakeSherbrooke()
    profile = extract_noise_profile(backend)
    print_noise_summary(profile)
