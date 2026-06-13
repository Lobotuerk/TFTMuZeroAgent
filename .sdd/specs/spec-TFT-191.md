# Technical Specification: TFT-191 Bug Fix

## 1. Overview

This document outlines the technical design for fixing a critical `IndexError` that occurs during combat calculations in the TFT simulator. The bug is caused by unsafe access to the `target.shields` list, which can be modified concurrently during shield damage calculation, leading to a race condition.

The fix will be applied in two locations:
1.  `TFTSet4Gym/TFTSet4Gym/tft_set4_gym/champion_functions.py`
2.  `TFTSet4Gym/TFTSet4Gym/tft_set4_gym/champion.py`

## 2. Root Cause Analysis

The traceback indicates the crash occurs at `champion_functions.py`, line 174, within the `attack` function.

```python
# champion_functions.py:168-178
while damage > 0 and len(target.shields) > 0:
    try:
        top_shield = target.shields[0]['amount']
        damage_to_shield = min(damage, top_shield)
        damage -= damage_to_shield
        target.shields[0]['amount'] -= damage_to_shield
        if target.shields[0]['amount'] < 0:
            target.shields.pop(0)
    except IndexError:
        # This handler is insufficient as it doesn't cover all access points
        break
```

The `while` loop condition `len(target.shields) > 0` is checked only at the start of each iteration. Inside the loop, `target.shields[0]` is accessed multiple times. If a shield is destroyed and popped from the list (`target.shields.pop(0)`), a subsequent access to `target.shields[0]` in the same iteration will raise an `IndexError`. The `try...except` block only covers the initial access, not the modification or the final check.

A nearly identical logic flaw exists in `champion.py` within the `spell` method, which also needs to be corrected.

## 3. Proposed Solution

The chosen solution (Approach A) is to safely reference the first shield in the list *once* per loop iteration. This avoids all race conditions related to the list being modified after the initial check.

### 3.1. `champion_functions.py` Modification

The `attack` function will be updated as follows:

**Old Logic (`champion_functions.py`):**
```python
while damage > 0 and len(target.shields) > 0:
    try:
        top_shield = target.shields[0]['amount']
        damage_to_shield = min(damage, top_shield)
        damage -= damage_to_shield
        target.shields[0]['amount'] -= damage_to_shield
        if target.shields[0]['amount'] < 0:
            target.shields.pop(0)
    except IndexError:
        break
```

**New Logic (`champion_functions.py`):**
```python
while damage > 0 and len(target.shields) > 0:
    # Safely reference the shield object once
    top_shield_ref = target.shields[0]
    
    damage_to_shield = min(damage, top_shield_ref['amount'])
    damage -= damage_to_shield
    top_shield_ref['amount'] -= damage_to_shield

    if top_shield_ref['amount'] <= 0:
        target.shields.pop(0)
```

### 3.2. `champion.py` Modification

The `spell` method in the `champion` class will be updated similarly.

**Old Logic (`champion.py`):**
```python
# Approximate logic from champion.py:279-286
while damage > 0 and len(target.shields) > 0:
    damage_to_shield = min(damage, target.shields[0]['amount'])
    damage -= damage_to_shield
    target.shields[0]['amount'] -= damage_to_shield
    if target.shields[0]['amount'] <= 0:
        target.shields.pop(0)
```

**New Logic (`champion.py`):**
```python
while damage > 0 and len(target.shields) > 0:
    # Safely reference the shield object once
    top_shield_ref = target.shields[0]

    damage_to_shield = min(damage, top_shield_ref['amount'])
    damage -= damage_to_shield
    top_shield_ref['amount'] -= damage_to_shield

    if top_shield_ref['amount'] <= 0:
        target.shields.pop(0)
```

## 4. Verification Plan

The fix will be verified by extending the existing test suite.

1.  **Locate Test File**: The existing shield tests are in `TFTSet4Gym/tests/test_shield_handling.py`.
2.  **Add New Test Case**: A new test will be created specifically to reproduce the `IndexError`. This test will involve a champion with a shield that is exactly destroyed by an incoming attack, forcing the `pop(0)` and subsequent unsafe access in the original code.
3.  **Run Tests**: All tests in the suite must pass, including the new regression test, to confirm the fix is effective and has not introduced regressions.

This approach ensures the bug is fixed and prevents it from recurring.
