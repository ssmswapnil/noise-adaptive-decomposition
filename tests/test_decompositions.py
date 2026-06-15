"""
test_decompositions.py — Verify all Toffoli decompositions are correct
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.decompositions import get_all_decompositions, verify_decomposition


def test_all_decompositions():
    """Test that every decomposition has correct computational basis action."""
    catalog = get_all_decompositions()
    all_passed = True

    for name, info in catalog.items():
        result = verify_decomposition(info["circuit"], label=name)

        # Every decomposition must have correct classical action
        assert result["action_correct"], (
            f"FAIL: {name} has incorrect computational basis action!"
        )

        # Exact decompositions must also match the unitary
        if info["exact"]:
            assert result["matrix_match"], (
                f"FAIL: {name} claims to be exact but unitary doesn't match CCX!"
            )
            print(f"  PASS (exact)  : {name}")
        else:
            print(f"  PASS (rphase) : {name}")

    print("\n  All decomposition tests passed!")


def test_cx_counts():
    """Verify the stated CX counts match actual circuit content."""
    catalog = get_all_decompositions()

    for name, info in catalog.items():
        circuit = info["circuit"]
        actual_cx = circuit.count_ops().get("cx", 0)
        expected_cx = info["cx_count"]
        assert actual_cx == expected_cx, (
            f"FAIL: {name} claims {expected_cx} CX but has {actual_cx}"
        )
        print(f"  PASS (cx={actual_cx}): {name}")

    print("\n  All CX count tests passed!")


if __name__ == "__main__":
    print("\n  Running decomposition verification tests...\n")
    test_all_decompositions()
    print()
    test_cx_counts()
