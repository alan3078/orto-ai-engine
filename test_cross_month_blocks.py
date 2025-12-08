"""
Cross-month block size and constraint tests.

Tests verify that blocks spanning month boundaries respect:
- Min consecutive (at least 2 nights)
- Max consecutive (at most 3 nights)
- Post-block rest (2 days after block ends)
- Night block gap (7 days between blocks)
"""

from src.services.solver import SolverService
from src.core.schemas import SolveRequest


def test_incomplete_single_extends_to_max():
    """Oct 31=E single should force Nov 1=E, and can extend to Nov 2=E for max block of 3"""
    print("\n" + "="*80)
    print("TEST 1: Incomplete single extends to max")
    print("="*80)
    
    staff = ['NUR001']
    time_slots = 7
    
    previous_month = [
        {'resource': 'NUR001', 'offset_days': 1, 'state': 2},  # Oct 31=E
        {'resource': 'NUR001', 'offset_days': 2, 'state': 0},  # Oct 30=O
        {'resource': 'NUR001', 'offset_days': 3, 'state': 0},
        {'resource': 'NUR001', 'offset_days': 4, 'state': 0},
        {'resource': 'NUR001', 'offset_days': 5, 'state': 0},
        {'resource': 'NUR001', 'offset_days': 6, 'state': 0},
        {'resource': 'NUR001', 'offset_days': 7, 'state': 0},
    ]
    
    constraints = [
        {'type': 'min_consecutive', 'resource': 'NUR001', 'time_slots': list(range(7)), 
         'target_state': 2, 'min_block': 2},
        {'type': 'horizontal_sum', 'resource': 'NUR001', 'time_slots': list(range(7)), 
         'target_state': 2, 'operator': '<=', 'value': 3},
        {'type': 'post_block_rest', 'resource': 'NUR001', 'target_state': 2, 'rest_days': 2},
    ]
    
    request = SolveRequest(
        config={'resources': staff, 'time_slots': time_slots, 'states': [0, 1, 2]},
        constraints=constraints,
        previous_month_assignments=previous_month,
    )
    
    solver = SolverService()
    result = solver.solve(request)
    
    print(f"\nStatus: {result.status}")
    assert result.status == 'OPTIMAL', f"Expected OPTIMAL, got {result.status}"
    
    schedule = result.schedule['NUR001']
    display = ' '.join(['O' if v==0 else '7' if v==1 else 'E' for v in schedule])
    print(f"Previous: Oct 31=E (Oct 30=O)")
    print(f"Current:  Nov 1-7: {display}")
    
    # Nov 1 must be E (forced continuation)
    assert schedule[0] == 2, f"Nov 1 should be E (2), got {schedule[0]}"
    print(f"✓ Nov 1 = E (forced to complete min block of 2)")
    
    # Nov 2 can be E (extends to max block of 3) OR be O (ends block at 2)
    # If Nov 2 is E, then Nov 3-4 must be O (rest)
    # If Nov 2 is O, then Nov 2-3 must be O (rest)
    
    if schedule[1] == 2:  # Block extends to 3 nights
        print(f"✓ Nov 2 = E (block extended to 3 nights: Oct 31-Nov 1-2)")
        assert schedule[2] == 0 and schedule[3] == 0, f"Nov 3-4 must be rest after 3-night block, got {schedule[2]}, {schedule[3]}"
        print(f"✓ Nov 3-4 = O O (rest after 3-night block)")
    else:  # Block ends at 2 nights
        print(f"✓ Nov 2 = O (block ends at 2 nights: Oct 31-Nov 1)")
        assert schedule[1] == 0 and schedule[2] == 0, f"Nov 2-3 must be rest after 2-night block, got {schedule[1]}, {schedule[2]}"
        print(f"✓ Nov 2-3 = O O (rest after 2-night block)")
    
    print(f"\n✓ TEST 1 PASSED")
    return True


def test_complete_block_prevents_extension():
    """Oct 30-31=EE complete block should not extend to Nov 1 (rest required)"""
    print("\n" + "="*80)
    print("TEST 2: Complete block prevents extension")
    print("="*80)
    
    staff = ['NUR001']
    time_slots = 7
    
    previous_month = [
        {'resource': 'NUR001', 'offset_days': 1, 'state': 2},  # Oct 31=E
        {'resource': 'NUR001', 'offset_days': 2, 'state': 2},  # Oct 30=E
        {'resource': 'NUR001', 'offset_days': 3, 'state': 0},  # Oct 29=O
        {'resource': 'NUR001', 'offset_days': 4, 'state': 0},
        {'resource': 'NUR001', 'offset_days': 5, 'state': 0},
        {'resource': 'NUR001', 'offset_days': 6, 'state': 0},
        {'resource': 'NUR001', 'offset_days': 7, 'state': 0},
    ]
    
    constraints = [
        {'type': 'min_consecutive', 'resource': 'NUR001', 'time_slots': list(range(7)), 
         'target_state': 2, 'min_block': 2},
        {'type': 'horizontal_sum', 'resource': 'NUR001', 'time_slots': list(range(7)), 
         'target_state': 2, 'operator': '<=', 'value': 3},
        {'type': 'post_block_rest', 'resource': 'NUR001', 'target_state': 2, 'rest_days': 2},
    ]
    
    request = SolveRequest(
        config={'resources': staff, 'time_slots': time_slots, 'states': [0, 1, 2]},
        constraints=constraints,
        previous_month_assignments=previous_month,
    )
    
    solver = SolverService()
    result = solver.solve(request)
    
    print(f"\nStatus: {result.status}")
    assert result.status == 'OPTIMAL', f"Expected OPTIMAL, got {result.status}"
    
    schedule = result.schedule['NUR001']
    display = ' '.join(['O' if v==0 else '7' if v==1 else 'E' for v in schedule])
    print(f"Previous: Oct 30-31=E E (Oct 29=O)")
    print(f"Current:  Nov 1-7: {display}")
    
    # Nov 1-2 must be O (rest after complete block)
    assert schedule[0] == 0 and schedule[1] == 0, f"Nov 1-2 should be rest (O), got {schedule[0]}, {schedule[1]}"
    print(f"✓ Nov 1-2 = O O (rest after complete 2-night block)")
    
    print(f"\n✓ TEST 2 PASSED")
    return True


def test_max_block_prevents_extension():
    """Oct 29-30-31=EEE (max block of 3) should not extend to Nov 1"""
    print("\n" + "="*80)
    print("TEST 3: Max block prevents extension")
    print("="*80)
    
    staff = ['NUR001']
    time_slots = 7
    
    previous_month = [
        {'resource': 'NUR001', 'offset_days': 1, 'state': 2},  # Oct 31=E
        {'resource': 'NUR001', 'offset_days': 2, 'state': 2},  # Oct 30=E
        {'resource': 'NUR001', 'offset_days': 3, 'state': 2},  # Oct 29=E
        {'resource': 'NUR001', 'offset_days': 4, 'state': 0},  # Oct 28=O
        {'resource': 'NUR001', 'offset_days': 5, 'state': 0},
        {'resource': 'NUR001', 'offset_days': 6, 'state': 0},
        {'resource': 'NUR001', 'offset_days': 7, 'state': 0},
    ]
    
    constraints = [
        {'type': 'min_consecutive', 'resource': 'NUR001', 'time_slots': list(range(7)), 
         'target_state': 2, 'min_block': 2},
        {'type': 'horizontal_sum', 'resource': 'NUR001', 'time_slots': list(range(7)), 
         'target_state': 2, 'operator': '<=', 'value': 3},
        {'type': 'post_block_rest', 'resource': 'NUR001', 'target_state': 2, 'rest_days': 2},
    ]
    
    request = SolveRequest(
        config={'resources': staff, 'time_slots': time_slots, 'states': [0, 1, 2]},
        constraints=constraints,
        previous_month_assignments=previous_month,
    )
    
    solver = SolverService()
    result = solver.solve(request)
    
    print(f"\nStatus: {result.status}")
    assert result.status == 'OPTIMAL', f"Expected OPTIMAL, got {result.status}"
    
    schedule = result.schedule['NUR001']
    display = ' '.join(['O' if v==0 else '7' if v==1 else 'E' for v in schedule])
    print(f"Previous: Oct 29-30-31=E E E (Oct 28=O)")
    print(f"Current:  Nov 1-7: {display}")
    
    # Nov 1 cannot be E (would violate max consecutive=3: Oct 29-30-31 + Nov 1 = 4)
    # Nov 1-2 must be O (rest after max block)
    assert schedule[0] == 0, f"Nov 1 must be O (rest), got {schedule[0]}"
    assert schedule[1] == 0, f"Nov 2 must be O (rest), got {schedule[1]}"
    print(f"✓ Nov 1 cannot be E (would exceed max consecutive 3)")
    print(f"✓ Nov 1-2 = O O (rest after 3-night block)")
    
    print(f"\n✓ TEST 3 PASSED")
    return True


def test_gap_enforcement_with_max_block():
    """7-day gap between blocks + max block of 3"""
    print("\n" + "="*80)
    print("TEST 4: Gap enforcement with max block")
    print("="*80)
    
    staff = ['NUR001']
    time_slots = 10  # Need more days to test gap
    
    previous_month = [
        {'resource': 'NUR001', 'offset_days': 1, 'state': 0},  # Oct 31=O
        {'resource': 'NUR001', 'offset_days': 2, 'state': 0},  # Oct 30=O
        {'resource': 'NUR001', 'offset_days': 3, 'state': 2},  # Oct 29=E (block end)
        {'resource': 'NUR001', 'offset_days': 4, 'state': 2},  # Oct 28=E
        {'resource': 'NUR001', 'offset_days': 5, 'state': 2},  # Oct 27=E (block start)
        {'resource': 'NUR001', 'offset_days': 6, 'state': 0},  # Oct 26=O
        {'resource': 'NUR001', 'offset_days': 7, 'state': 0},  # Oct 25=O
    ]
    
    constraints = [
        {'type': 'min_consecutive', 'resource': 'NUR001', 'time_slots': list(range(10)), 
         'target_state': 2, 'min_block': 2},
        {'type': 'horizontal_sum', 'resource': 'NUR001', 'time_slots': list(range(10)), 
         'target_state': 2, 'operator': '<=', 'value': 3},
        {'type': 'post_block_rest', 'resource': 'NUR001', 'target_state': 2, 'rest_days': 2},
        {'type': 'night_block_gap', 'resource': 'NUR001', 'time_slots': list(range(10)), 
         'target_state': 2, 'min_gap_days': 7},
    ]
    
    request = SolveRequest(
        config={'resources': staff, 'time_slots': time_slots, 'states': [0, 1, 2]},
        constraints=constraints,
        previous_month_assignments=previous_month,
    )
    
    solver = SolverService()
    result = solver.solve(request)
    
    print(f"\nStatus: {result.status}")
    assert result.status == 'OPTIMAL', f"Expected OPTIMAL, got {result.status}"
    
    schedule = result.schedule['NUR001']
    display = ' '.join(['O' if v==0 else '7' if v==1 else 'E' for v in schedule])
    print(f"Previous: Oct 27-28-29=E E E (3-night block ended Oct 29)")
    print(f"Current:  Nov 1-10: {display}")
    
    # Oct 29 block ended, so rest Oct 30-31 (already O)
    # 7-day gap: Oct 30, 31, Nov 1, 2, 3, 4, 5 = 7 days
    # Earliest new block can start: Nov 6
    
    # Nov 1-5 should have no E (within 7-day gap)
    for i in range(5):
        assert schedule[i] != 2, f"Nov {i+1} should not be E (within 7-day gap), got state {schedule[i]}"
    print(f"✓ Nov 1-5 have no E (respecting 7-day gap from Oct 29 block end)")
    
    # Nov 6+ can have E (gap satisfied)
    has_block_after_gap = any(schedule[i] == 2 for i in range(5, 10))
    if has_block_after_gap:
        print(f"✓ New block found after Nov 5 (gap satisfied)")
    
    print(f"\n✓ TEST 4 PASSED")
    return True


def test_multiple_staff_cross_month():
    """Multiple staff with different cross-month scenarios"""
    print("\n" + "="*80)
    print("TEST 5: Multiple staff with different cross-month scenarios")
    print("="*80)
    
    staff = ['NUR001', 'NUR002', 'NUR003']
    time_slots = 7
    
    previous_month = [
        # NUR001: Complete 2-block ending Oct 30-31
        {'resource': 'NUR001', 'offset_days': 1, 'state': 2},  # Oct 31=E
        {'resource': 'NUR001', 'offset_days': 2, 'state': 2},  # Oct 30=E
        {'resource': 'NUR001', 'offset_days': 3, 'state': 0},
        {'resource': 'NUR001', 'offset_days': 4, 'state': 0},
        {'resource': 'NUR001', 'offset_days': 5, 'state': 0},
        {'resource': 'NUR001', 'offset_days': 6, 'state': 0},
        {'resource': 'NUR001', 'offset_days': 7, 'state': 0},
        
        # NUR002: Incomplete 1-block ending Oct 31
        {'resource': 'NUR002', 'offset_days': 1, 'state': 2},  # Oct 31=E
        {'resource': 'NUR002', 'offset_days': 2, 'state': 0},  # Oct 30=O
        {'resource': 'NUR002', 'offset_days': 3, 'state': 0},
        {'resource': 'NUR002', 'offset_days': 4, 'state': 0},
        {'resource': 'NUR002', 'offset_days': 5, 'state': 0},
        {'resource': 'NUR002', 'offset_days': 6, 'state': 0},
        {'resource': 'NUR002', 'offset_days': 7, 'state': 0},
        
        # NUR003: Max 3-block ending Oct 29-30-31
        {'resource': 'NUR003', 'offset_days': 1, 'state': 2},  # Oct 31=E
        {'resource': 'NUR003', 'offset_days': 2, 'state': 2},  # Oct 30=E
        {'resource': 'NUR003', 'offset_days': 3, 'state': 2},  # Oct 29=E
        {'resource': 'NUR003', 'offset_days': 4, 'state': 0},
        {'resource': 'NUR003', 'offset_days': 5, 'state': 0},
        {'resource': 'NUR003', 'offset_days': 6, 'state': 0},
        {'resource': 'NUR003', 'offset_days': 7, 'state': 0},
    ]
    
    constraints = []
    for s in staff:
        constraints.extend([
            {'type': 'min_consecutive', 'resource': s, 'time_slots': list(range(7)), 
             'target_state': 2, 'min_block': 2},
            {'type': 'horizontal_sum', 'resource': s, 'time_slots': list(range(7)), 
             'target_state': 2, 'operator': '<=', 'value': 3},
            {'type': 'post_block_rest', 'resource': s, 'target_state': 2, 'rest_days': 2},
        ])
    
    request = SolveRequest(
        config={'resources': staff, 'time_slots': time_slots, 'states': [0, 1, 2]},
        constraints=constraints,
        previous_month_assignments=previous_month,
    )
    
    solver = SolverService()
    result = solver.solve(request)
    
    print(f"\nStatus: {result.status}")
    assert result.status == 'OPTIMAL', f"Expected OPTIMAL, got {result.status}"
    
    print("\nPrevious Month:")
    print("  NUR001: Oct 30-31 = E E (complete 2-block)")
    print("  NUR002: Oct 31 = E (incomplete 1-block)")
    print("  NUR003: Oct 29-30-31 = E E E (complete 3-block)")
    
    print("\nCurrent Month (Nov 1-7):")
    for s in staff:
        schedule = result.schedule[s]
        display = ' '.join(['O' if v==0 else '7' if v==1 else 'E' for v in schedule])
        print(f"  {s}: {display}")
    
    # Validate each staff
    nur001 = result.schedule['NUR001']
    assert nur001[0] == 0 and nur001[1] == 0, "NUR001: Nov 1-2 should be rest"
    print("\n✓ NUR001: Nov 1-2 = O O (rest after complete 2-block)")
    
    nur002 = result.schedule['NUR002']
    assert nur002[0] == 2, "NUR002: Nov 1 should be E (forced continuation)"
    print("✓ NUR002: Nov 1 = E (forced to complete block)")
    
    nur003 = result.schedule['NUR003']
    assert nur003[0] == 0 and nur003[1] == 0, "NUR003: Nov 1-2 should be rest"
    print("✓ NUR003: Nov 1-2 = O O (rest after max 3-block)")
    
    print(f"\n✓ TEST 5 PASSED")
    return True


if __name__ == '__main__':
    print("\n" + "="*80)
    print("CROSS-MONTH BLOCK SIZE AND CONSTRAINT TESTS")
    print("="*80)
    
    try:
        test_incomplete_single_extends_to_max()
        test_complete_block_prevents_extension()
        test_max_block_prevents_extension()
        test_gap_enforcement_with_max_block()
        test_multiple_staff_cross_month()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print()
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise
