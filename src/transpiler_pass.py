"""
transpiler_pass.py — Noise-aware Toffoli decomposition for Qiskit
==================================================================

Two approaches are provided:

1. NoiseAdaptiveToffoliDecomposition (TransformationPass)
   - Works on the DAG level, best for custom pipeline construction
   - Must be inserted BEFORE Qiskit's Unroll3qOrMore pass

2. transpile_with_noise_awareness() (recommended)
   - Two-phase approach: scout layout first, then replace CCX, then transpile
   - Drop-in replacement for qiskit.transpile()
   - Handles the ordering problem automatically

The core issue: Qiskit's default pipeline decomposes CCX into native gates
in the `init` stage (via Unroll3qOrMore), BEFORE layout and routing run.
This means a pass inserted later in the pipeline will never see CCX gates.

Our two-phase solution:
  Phase 1 (Scout): Quick transpile at optimization_level=0 to discover
                    the physical qubit layout the transpiler would choose.
  Phase 2 (Replace): In the original circuit, replace each CCX with the
                      noise-optimal decomposition using the scouted layout.
  Phase 3 (Transpile): Transpile the modified circuit normally. The
                        transpiler now sees CX + single-qubit gates
                        (not CCX), and translates them to native gates.
"""

from qiskit.transpiler import PassManager, TransformationPass
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import CCXGate
from qiskit.converters import circuit_to_dag
from qiskit import transpile as qiskit_transpile
from typing import Dict, Optional, List, Tuple

from src.decompositions import get_all_decompositions
from src.cost_function import select_best_decomposition


class NoiseAdaptiveToffoliDecomposition(TransformationPass):
    """
    Qiskit TransformationPass that replaces CCX gates with
    noise-optimal decompositions.

    NOTE: This pass must see CCX gates in the DAG. If Qiskit's
    Unroll3qOrMore has already run, no CCX gates will remain.
    For general use, prefer transpile_with_noise_awareness() which
    handles this automatically.
    """

    def __init__(
        self,
        noise_profile: Dict,
        require_exact: bool = False,
        cost_fn: str = "additive",
    ):
        super().__init__()
        self.noise_profile = noise_profile
        self.require_exact = require_exact
        self.cost_fn = cost_fn
        self.decompositions = get_all_decompositions()
        self._selection_log = []

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        self._selection_log = []

        ccx_nodes = [
            node for node in dag.op_nodes()
            if isinstance(node.op, CCXGate)
        ]

        for node in ccx_nodes:
            physical_qubits = tuple(
                dag.find_bit(q).index for q in node.qargs
            )

            name, circuit, cost = select_best_decomposition(
                self.decompositions,
                self.noise_profile,
                physical_qubits,
                require_exact=self.require_exact,
                cost_fn=self.cost_fn,
            )

            self._selection_log.append({
                "physical_qubits": physical_qubits,
                "chosen_decomposition": name,
                "estimated_cost": cost,
            })

            decomp_dag = circuit_to_dag(circuit)
            dag.substitute_node_with_dag(node, decomp_dag)

        return dag

    @property
    def selection_log(self):
        return self._selection_log


# ════════════════════════════════════════════════════════════
#  Two-phase transpilation (recommended approach)
# ════════════════════════════════════════════════════════════

def _scout_layout(circuit: QuantumCircuit, backend) -> Dict[int, int]:
    """
    Phase 1: Quick transpile to discover which physical qubits
    the transpiler would assign to each virtual qubit.

    Returns:
        Dict mapping virtual qubit index → physical qubit index
    """
    scouted = qiskit_transpile(
        circuit, backend=backend, optimization_level=0
    )

    virt_to_phys = {}

    if scouted.layout and scouted.layout.initial_layout:
        initial_layout = scouted.layout.initial_layout

        # get_virtual_bits() returns {Qubit: physical_index}
        # where the Qubit objects are from the ORIGINAL circuit
        virt_bits = initial_layout.get_virtual_bits()

        for vbit, phys_idx in virt_bits.items():
            # Find the index of this virtual qubit in the original circuit.
            # The layout includes ancilla qubits added by the transpiler
            # (e.g., 124 ancillas on a 127-qubit backend for a 3-qubit circuit).
            # These don't exist in the original circuit, so we skip them.
            try:
                virt_idx = circuit.find_bit(vbit).index
                virt_to_phys[virt_idx] = phys_idx
            except Exception:
                # Ancilla qubit not in original circuit — skip
                pass

    # Fallback: identity mapping for any unmapped qubits
    for i in range(circuit.num_qubits):
        if i not in virt_to_phys:
            virt_to_phys[i] = i

    return virt_to_phys


def _replace_ccx_in_circuit(
    circuit: QuantumCircuit,
    noise_profile: Dict,
    virt_to_phys: Dict[int, int],
    require_exact: bool = False,
    cost_fn: str = "additive",
) -> Tuple[QuantumCircuit, List[Dict]]:
    """
    Phase 2: Walk the circuit and replace every CCX gate with
    the noise-optimal decomposition from the catalog.

    Returns:
        (modified_circuit, selection_log)
    """
    decompositions = get_all_decompositions()
    selection_log = []

    # Build new circuit with same structure
    modified = circuit.copy_empty_like()

    for instruction in circuit.data:
        gate = instruction.operation

        if gate.name == "ccx":
            # Get virtual qubit indices
            virt_indices = [
                circuit.find_bit(q).index for q in instruction.qubits
            ]
            # Map to physical qubits
            phys_indices = tuple(virt_to_phys[v] for v in virt_indices)

            # Select best decomposition
            name, decomp_circuit, cost = select_best_decomposition(
                decompositions,
                noise_profile,
                phys_indices,
                require_exact=require_exact,
                cost_fn=cost_fn,
            )

            selection_log.append({
                "virtual_qubits": tuple(virt_indices),
                "physical_qubits": phys_indices,
                "chosen_decomposition": name,
                "estimated_cost": cost,
            })

            # Compose the decomposition into the circuit
            modified.compose(
                decomp_circuit,
                qubits=virt_indices,
                inplace=True,
            )
        else:
            # Pass through all non-CCX gates unchanged
            modified.append(instruction)

    return modified, selection_log


def transpile_with_noise_awareness(
    circuit: QuantumCircuit,
    backend,
    noise_profile: Dict,
    optimization_level: int = 1,
    require_exact: bool = False,
    cost_fn: str = "additive",
    verbose: bool = False,
) -> QuantumCircuit:
    """
    Drop-in replacement for qiskit.transpile() with noise-adaptive
    Toffoli decomposition.

    Three-phase approach:
      1. Scout: quick transpile to discover physical qubit layout
      2. Replace: swap CCX for noise-optimal decompositions
      3. Transpile: full transpilation of the modified circuit

    Args:
        circuit: Input QuantumCircuit (may contain CCX gates)
        backend: IBM backend (real or fake)
        noise_profile: From extract_noise_profile()
        optimization_level: Qiskit optimization level (0-3)
        require_exact: Only use unitary-exact decompositions
        cost_fn: 'additive' or 'weighted'
        verbose: Print decomposition choices

    Returns:
        Fully transpiled QuantumCircuit
    """
    # Count CCX gates — if none, just do normal transpilation
    ccx_count = circuit.count_ops().get("ccx", 0)
    if ccx_count == 0:
        pm = generate_preset_pass_manager(
            optimization_level=optimization_level, backend=backend
        )
        return pm.run(circuit)

    # Phase 1: Scout the layout
    virt_to_phys = _scout_layout(circuit, backend)

    if verbose:
        print(f"  [Scout] Layout: {virt_to_phys}")

    # Phase 2: Replace CCX gates
    modified, selection_log = _replace_ccx_in_circuit(
        circuit, noise_profile, virt_to_phys,
        require_exact=require_exact, cost_fn=cost_fn,
    )

    if verbose:
        for entry in selection_log:
            print(
                f"  [Replace] CCX on virtual {entry['virtual_qubits']} "
                f"→ physical {entry['physical_qubits']} "
                f"→ {entry['chosen_decomposition']} (cost={entry['estimated_cost']:.6f})"
            )

    # Phase 3: Transpile the modified circuit
    # The transpiler now sees CX + single-qubit gates, not CCX
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level, backend=backend
    )
    result = pm.run(modified)

    # Attach selection log as metadata
    result._noise_aware_log = selection_log

    return result


def build_noise_aware_pass_manager(
    backend,
    noise_profile: Dict,
    optimization_level: int = 1,
    require_exact: bool = False,
    cost_fn: str = "additive",
) -> PassManager:
    """
    Build a pass manager with noise-adaptive decomposition inserted
    BEFORE Qiskit's standard unrolling.

    NOTE: This inserts the pass at the init stage. It works but the
    pass won't have physical qubit info (uses virtual indices as-is).
    For full noise-awareness with physical qubit mapping, use
    transpile_with_noise_awareness() instead.
    """
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
    )

    noise_pass = NoiseAdaptiveToffoliDecomposition(
        noise_profile=noise_profile,
        require_exact=require_exact,
        cost_fn=cost_fn,
    )

    # Prepend to init stage so it runs before Unroll3qOrMore
    noise_pm = PassManager([noise_pass])
    original_init = pm.init
    if original_init is not None:
        pm.init = noise_pm + original_init
    else:
        pm.init = noise_pm

    return pm
