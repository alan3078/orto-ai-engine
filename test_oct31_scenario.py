"""
Test the specific scenario: Oct 31=E should force Nov 1=E, and can extend to Nov 2=E for max block of 3.
"""

from src.services.solver import SolverService
from src.core.schemas import SolveRequest


def test_oct31_single_to_3_block():
    """
    Scenario: Oct 31 = E (single, incomplete)
    
    Constraints:
    - Min consecutive = 2
    - Max consecutive = 3
    - Post-block rest = 2 days
    
    Expected behavior:
    - Oct 31=E is incomplete (min 2 required)
    - Nov 1 MUST be E (forced to complete min block)
    - Nov 2 CAN be E (extends to max block of 3)
    - After block ends, 2 days rest required
    """
    print("\n" + "="*80)
    print("TEST: Oct 31 single E extends to max 3-block")
    print("="*80)
    
    staff = ['NURSE_A']
    time_slots = 10
    
    previous_month = [
        {'resource': 'NURSE_A', 'offset_days': 1, 'state': 2},  # Oct 31=E
        {'resource': 'NURSE_A', 'offset_days': 2, 'state': 0},  # Oct 30=O
        {'resource': 'NURSE_A', 'offset_days': 3, 'state': 0},
        {'resource': 'NURSE_A', 'offset_days': 4, 'state': 0},
        {'resource': 'NURSE_A', 'offset_days': 5, 'state': 0},
        {'resource': 'NURSE_A', 'offset_days': 6, 'state': 0},
        {'resource': 'NURSE_A', 'offset_days': 7, 'state': 0},
    ]
    
    constraints = [
        {'type': 'min_consecutive', 'resource': 'NURSE_A', 'time_slots': list(range(10)), 
         'target_state': 2, 'min_block': 2},
        {'type': 'horizontal_sum', 'resource': 'NURSE_A', 'time_slots': list(range(10)), 
         'target_state': 2, 'operator': '<=', 'value': 3},
        {'type': 'post_block_rest', 'resource': 'NURSE_A', 'target_state': 2, 'rest_days': 2},
    ]
    
    request = SolveRequest(
        config={'resources': staff, 'time_slots': time_slots, 'states': [0, 1, 2]},
        constraints=constraints,
        previous_month_assignments=previous_month,
    )
    
    solver = SolverService()
    result = solver.solve(request)
    
    assert result.status == 'OPTIMAL', f"Expected OPTIMAL, got {result.status}"
    
    schedule = result.schedule['NURSE_A']
    display = ' '.join(['O' if v==0 else '7' if v==1 else 'E' for v in schedule])
    
    print(f"\nPrevious Month:")
    print(f"  Oct 31 = E (single, incomplete)")
    print(f"  Oct 30 = O")
    print(f"\nCurrent Month (Nov 1-10):")
    print(f"  {display}")
    
    # Analyze the schedule
    print(f"\nAnalysis:")
    
    # Nov 1 must be E (forced)
    assert schedule[0] == 2, f"Nov 1 MUST be E (forced to complete min block), got {schedule[0]}"
    print(f"  ✓ Nov 1 = E (forced to complete min consecutive = 2)")
    
    # Check block structure
    # Find the first E block
    block_start = 0
    block_end = 0
    for i in range(len(schedule)):
        if schedule[i] == 2:
            if i == 0 or schedule[i-1] != 2:
                block_start = i
            if i == len(schedule) - 1 or schedule[i+1] != 2:
                block_end = i
                break
    
    block_size = block_end - block_start + 1
    print(f"  ✓ First block: Nov {block_start+1} to Nov {block_end+1} ({block_size} nights)")
    print(f"    Cross-month block: Oct 31 + Nov {block_start+1}-{block_end+1} = {block_size+1} nights total")
    
    # Block size should be 1-2 (making total with Oct 31 = 2-3)
    assert 1 <= block_size <= 2, f"First block in Nov should be 1-2 nights, got {block_size}"
    total_block_size = block_size + 1  # +1 for Oct 31
    print(f"    Total block size (including Oct 31): {total_block_size} nights")
    assert 2 <= total_block_size <= 3, f"Total block should be 2-3, got {total_block_size}"
    print(f"  ✓ Total block size = {total_block_size} (within 2-3 range)")
    
    # Rest days after block
    rest_start = block_end + 1
    if rest_start < len(schedule):
        rest_end = rest_start + 1  # 2 days rest
        if rest_end < len(schedule):
            assert schedule[rest_start] == 0 and schedule[rest_end] == 0, \
                f"Nov {rest_start+1}-{rest_end+1} should be rest (O), got {schedule[rest_start]}, {schedule[rest_end]}"
            print(f"  ✓ Nov {rest_start+1}-{rest_end+1} = O O (2 days rest after block ends)")
    
    # Verify max consecutive not exceeded IN CURRENT MONTH
    max_consecutive_current = 0
    current_consecutive = 0
    for i, state in enumerate(schedule):
        if state == 2:
            current_consecutive += 1
            max_consecutive_current = max(max_consecutive_current, current_consecutive)
        else:
            current_consecutive = 0
    
    print(f"  ✓ Max consecutive E in current month: {max_consecutive_current} (limit: 3)")
    assert max_consecutive_current <= 3, f"Max consecutive in current month should be <= 3, got {max_consecutive_current}"
    
    # Verify max consecutive INCLUDING previous month (cross-month check)
    # If Nov 1 is E, the cross-month block (Oct 31 + Nov 1) counts
    if schedule[0] == 2:
        # Count how many E's at the start of Nov
        cross_month_count = 1  # Oct 31
        for i in range(len(schedule)):
            if schedule[i] == 2:
                cross_month_count += 1
            else:
                break
        print(f"  ✓ Cross-month block (Oct 31 + Nov start): {cross_month_count} nights (limit: 3)")
        assert cross_month_count <= 3, f"Cross-month consecutive should be <= 3, got {cross_month_count}"
    
    print(f"\n{'='*80}")
    print(f"✓ TEST PASSED: Oct 31 single E correctly extends to {total_block_size}-night block")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    test_oct31_single_to_3_block()
