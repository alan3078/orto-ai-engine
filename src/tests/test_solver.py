"""
Unit tests for the solver service.
Tests various constraint scenarios including infeasible cases.
"""

import pytest
from src.core.schemas import (
    ConfigModel,
    SolveRequest,
    PointConstraint,
    VerticalSumConstraint,
    PatternBlockConstraint,
    PreviousMonthAssignment,
    SlidingWindowConstraint,
)
from src.services.solver import solver_service


def test_simple_feasible_solution():
    """Test a simple feasible scheduling problem."""
    request = SolveRequest(
        config=ConfigModel(
            resources=["A", "B"],
            time_slots=2,
            states=[0, 1]
        ),
        constraints=[
            PointConstraint(
                type="point",
                resource="A",
                time_slot=0,
                state=0
            )
        ]
    )
    
    response = solver_service.solve(request)
    
    assert response.status in ["OPTIMAL", "FEASIBLE"]
    assert response.schedule is not None
    assert response.schedule["A"][0] == 0  # A must be off on day 0


def test_vertical_sum_constraint():
    """Test vertical sum constraint (coverage requirement)."""
    request = SolveRequest(
        config=ConfigModel(
            resources=["A", "B", "C"],
            time_slots=2,
            states=[0, 1]
        ),
        constraints=[
            VerticalSumConstraint(
                type="vertical_sum",
                time_slot="ALL",
                target_state=1,
                operator=">=",
                value=2
            )
        ]
    )
    
    response = solver_service.solve(request)
    
    assert response.status in ["OPTIMAL", "FEASIBLE"]
    assert response.schedule is not None
    
    # Verify at least 2 people working each day
    for t in range(2):
        working_count = sum(
            1 for resource in ["A", "B", "C"]
            if response.schedule[resource][t] == 1
        )
        assert working_count >= 2


def test_infeasible_scenario():
    """
    Test infeasible scenario: Impossible constraints.
    
    Scenario: 3 resources, all must be OFF, but also need 2 working.
    This is mathematically impossible.
    """
    request = SolveRequest(
        config=ConfigModel(
            resources=["A", "B", "C"],
            time_slots=1,
            states=[0, 1]
        ),
        constraints=[
            # Everyone must be OFF
            PointConstraint(type="point", resource="A", time_slot=0, state=0),
            PointConstraint(type="point", resource="B", time_slot=0, state=0),
            PointConstraint(type="point", resource="C", time_slot=0, state=0),
            # But we also need 2 people working - contradiction!
            VerticalSumConstraint(
                type="vertical_sum",
                time_slot=0,
                target_state=1,
                operator=">=",
                value=2
            )
        ]
    )
    
    response = solver_service.solve(request)
    
    # Should return INFEASIBLE, not crash
    assert response.status == "INFEASIBLE"
    assert response.schedule is None
    assert "No solution exists" in response.message


def test_conflicting_point_constraints():
    """
    Test conflicting point constraints.
    
    Scenario: Resource A at time 0 must be both 0 AND 1 (impossible).
    """
    request = SolveRequest(
        config=ConfigModel(
            resources=["A"],
            time_slots=1,
            states=[0, 1]
        ),
        constraints=[
            PointConstraint(
                type="point",
                resource="A",
                time_slot=0,
                state=0
            ),
            PointConstraint(
                type="point",
                resource="A",
                time_slot=0,
                state=1
            )
        ]
    )
    
    response = solver_service.solve(request)
    
    assert response.status == "INFEASIBLE"
    assert response.schedule is None


def test_complex_scenario():
    """
    Test a more complex realistic scenario.
    
    Requirements:
    - 5 nurses over 7 days
    - Nurse A must be off on day 0
    - At least 3 nurses working every day
    - No more than 4 nurses working any day
    """
    request = SolveRequest(
        config=ConfigModel(
            resources=["A", "B", "C", "D", "E"],
            time_slots=7,
            states=[0, 1]
        ),
        constraints=[
            PointConstraint(
                type="point",
                resource="A",
                time_slot=0,
                state=0
            ),
            VerticalSumConstraint(
                type="vertical_sum",
                time_slot="ALL",
                target_state=1,
                operator=">=",
                value=3
            ),
            VerticalSumConstraint(
                type="vertical_sum",
                time_slot="ALL",
                target_state=1,
                operator="<=",
                value=4
            )
        ]
    )
    
    response = solver_service.solve(request)
    
    assert response.status in ["OPTIMAL", "FEASIBLE"]
    assert response.schedule is not None
    
    # Verify constraints
    assert response.schedule["A"][0] == 0  # A off on day 0
    
    for t in range(7):
        working_count = sum(
            1 for resource in ["A", "B", "C", "D", "E"]
            if response.schedule[resource][t] == 1
        )
        assert 3 <= working_count <= 4


def test_empty_constraints():
    """Test solving with no constraints (should always succeed)."""
    request = SolveRequest(
        config=ConfigModel(
            resources=["A", "B"],
            time_slots=3,
            states=[0, 1]
        ),
        constraints=[]
    )
    
    response = solver_service.solve(request)
    
    assert response.status in ["OPTIMAL", "FEASIBLE"]
    assert response.schedule is not None
    assert len(response.schedule["A"]) == 3
    assert len(response.schedule["B"]) == 3


def test_validation_resource_not_in_config():
    """Test that validation catches invalid resource in constraint."""
    with pytest.raises(ValueError, match="not in resources list"):
        SolveRequest(
            config=ConfigModel(
                resources=["A", "B"],
                time_slots=2,
                states=[0, 1]
            ),
            constraints=[
                PointConstraint(
                    type="point",
                    resource="Z",  # Not in resources
                    time_slot=0,
                    state=0
                )
            ]
        )


def test_validation_time_slot_out_of_range():
    """Test that validation catches time_slot >= time_slots."""
    with pytest.raises(ValueError, match="time_slot"):
        SolveRequest(
            config=ConfigModel(
                resources=["A"],
                time_slots=2,
                states=[0, 1]
            ),
            constraints=[
                PointConstraint(
                    type="point",
                    resource="A",
                    time_slot=5,  # Out of range
                    state=0
                )
            ]
        )


def test_validation_invalid_state():
    """Test that validation catches invalid state value."""
    with pytest.raises(ValueError, match="not in states"):
        SolveRequest(
            config=ConfigModel(
                resources=["A"],
                time_slots=2,
                states=[0, 1]
            ),
            constraints=[
                PointConstraint(
                    type="point",
                    resource="A",
                    time_slot=0,
                    state=5  # Invalid state
                )
            ]
        )


def test_pattern_block_night_to_day():
    """
    Test pattern_block constraint: Block Night→Day transition.
    
    Scenario: 3-state system (OFF=0, DAY=1, NIGHT=2).
    Constraint: No resource can have NIGHT followed immediately by DAY.
    
    Expected: Solver either avoids NIGHT→DAY or returns INFEASIBLE if impossible.
    """
    request = SolveRequest(
        config=ConfigModel(
            resources=["A", "B"],
            time_slots=5,
            states=[0, 1, 2]  # OFF, DAY, NIGHT
        ),
        constraints=[
            # Force A to work nights on days 0-2
            PointConstraint(type="point", resource="A", time_slot=0, state=2),
            PointConstraint(type="point", resource="A", time_slot=1, state=2),
            PointConstraint(type="point", resource="A", time_slot=2, state=2),
            # Block NIGHT→DAY pattern
            PatternBlockConstraint(
                type="pattern_block",
                pattern=["NIGHT", "DAY"],
                resources="ALL",
                state_mapping={"NIGHT": 2, "DAY": 1, "OFF": 0}
            )
        ]
    )
    
    response = solver_service.solve(request)
    
    # Should find a solution (or be infeasible if we add more constraints)
    assert response.status in ["OPTIMAL", "FEASIBLE", "INFEASIBLE"]
    
    if response.status in ["OPTIMAL", "FEASIBLE"]:
        # Verify no NIGHT→DAY transitions exist
        for resource in ["A", "B"]:
            schedule = response.schedule[resource]
            for t in range(len(schedule) - 1):
                current_state = schedule[t]
                next_state = schedule[t + 1]
                # Assert: NOT (current_state == 2 AND next_state == 1)
                if current_state == 2:  # NIGHT
                    assert next_state != 1, f"Found forbidden NIGHT→DAY transition at {resource} time {t}→{t+1}"


def test_pattern_block_forces_off_after_night():
    """
    Test pattern_block with forced day shift: Should result in OFF or INFEASIBLE.
    
    Scenario: Force NIGHT at t=0, force DAY at t=1, apply NIGHT→DAY block.
    Expected: INFEASIBLE (since NIGHT→DAY is blocked but required).
    """
    request = SolveRequest(
        config=ConfigModel(
            resources=["A"],
            time_slots=2,
            states=[0, 1, 2]  # OFF, DAY, NIGHT
        ),
        constraints=[
            # Force NIGHT at time 0
            PointConstraint(type="point", resource="A", time_slot=0, state=2),
            # Force DAY at time 1
            PointConstraint(type="point", resource="A", time_slot=1, state=1),
            # Block NIGHT→DAY pattern
            PatternBlockConstraint(
                type="pattern_block",
                pattern=["NIGHT", "DAY"],
                resources="ALL",
                state_mapping={"NIGHT": 2, "DAY": 1, "OFF": 0}
            )
        ]
    )
    
    response = solver_service.solve(request)
    
    # Should be INFEASIBLE since we require NIGHT→DAY but block it
    assert response.status == "INFEASIBLE"
    assert response.schedule is None


def test_pattern_block_allows_night_to_off():
    """
    Test that pattern_block only blocks specific transition, not all from NIGHT.
    
    Scenario: NIGHT at t=0, OFF at t=1 should be allowed (only NIGHT→DAY blocked).
    """
    request = SolveRequest(
        config=ConfigModel(
            resources=["A"],
            time_slots=2,
            states=[0, 1, 2]  # OFF, DAY, NIGHT
        ),
        constraints=[
            # Force NIGHT at time 0
            PointConstraint(type="point", resource="A", time_slot=0, state=2),
            # Force OFF at time 1 (should be allowed)
            PointConstraint(type="point", resource="A", time_slot=1, state=0),
            # Block NIGHT→DAY pattern
            PatternBlockConstraint(
                type="pattern_block",
                pattern=["NIGHT", "DAY"],
                resources="ALL",
                state_mapping={"NIGHT": 2, "DAY": 1, "OFF": 0}
            )
        ]
    )
    
    response = solver_service.solve(request)
    
    # Should be feasible (NIGHT→OFF is allowed)
    assert response.status in ["OPTIMAL", "FEASIBLE"]
    assert response.schedule["A"] == [2, 0]


# ============================================================================
# Previous Month Integration Tests (FN/ADM/RST/002)
# ============================================================================


def test_previous_month_no_assignments():
    """Test that solver works normally when no previous month assignments provided."""
    request = SolveRequest(
        config=ConfigModel(
            resources=["A", "B"],
            time_slots=3,
            states=[0, 1]
        ),
        constraints=[
            VerticalSumConstraint(
                type="vertical_sum",
                time_slot="ALL",
                target_state=1,
                operator=">=",
                value=1
            )
        ],
        previous_month_assignments=None
    )
    
    response = solver_service.solve(request)
    
    assert response.status in ["OPTIMAL", "FEASIBLE"]
    assert response.schedule is not None
    assert len(response.schedule["A"]) == 3
    assert len(response.schedule["B"]) == 3


def test_previous_month_basic_integration():
    """Test that previous month assignments are respected as fixed constraints."""
    request = SolveRequest(
        config=ConfigModel(
            resources=["A", "B"],
            time_slots=3,
            states=[0, 1, 2]  # OFF, DAY, NIGHT
        ),
        constraints=[],
        previous_month_assignments=[
            # A worked NIGHT on last day of previous month
            PreviousMonthAssignment(resource="A", offset_days=1, state=2),
            # B was OFF on last day of previous month  
            PreviousMonthAssignment(resource="B", offset_days=1, state=0),
        ]
    )
    
    response = solver_service.solve(request)
    
    assert response.status in ["OPTIMAL", "FEASIBLE"]
    assert response.schedule is not None
    # Output should only contain current month (3 days)
    assert len(response.schedule["A"]) == 3
    assert len(response.schedule["B"]) == 3


def test_previous_month_pattern_block_at_boundary():
    """
    Test pattern_block constraint blocks Night→Day at month boundary.
    
    Scenario: 
    - Resource A had NIGHT on last day of previous month (offset_days=1)
    - PatternBlock constraint blocks NIGHT→DAY
    - First day of current month should NOT be DAY for A
    """
    request = SolveRequest(
        config=ConfigModel(
            resources=["A"],
            time_slots=2,
            states=[0, 1, 2]  # OFF, DAY, NIGHT
        ),
        constraints=[
            # Block NIGHT→DAY pattern
            PatternBlockConstraint(
                type="pattern_block",
                pattern=["NIGHT", "DAY"],
                resources="ALL",
                state_mapping={"NIGHT": 2, "DAY": 1, "OFF": 0}
            )
        ],
        previous_month_assignments=[
            # A worked NIGHT on last day of previous month
            PreviousMonthAssignment(resource="A", offset_days=1, state=2),
        ]
    )
    
    response = solver_service.solve(request)
    
    assert response.status in ["OPTIMAL", "FEASIBLE"]
    assert response.schedule is not None
    # First day of current month should NOT be DAY (state=1) due to pattern block
    assert response.schedule["A"][0] != 1, "First day should not be DAY after NIGHT"


def test_previous_month_pattern_block_infeasible():
    """
    Test that infeasible scenario with previous month is correctly detected.
    
    Scenario:
    - Resource A had NIGHT on last day of previous month
    - We force A to have DAY on first day of current month
    - PatternBlock forbids NIGHT→DAY
    - Should be INFEASIBLE
    """
    request = SolveRequest(
        config=ConfigModel(
            resources=["A"],
            time_slots=2,
            states=[0, 1, 2]  # OFF, DAY, NIGHT
        ),
        constraints=[
            # Force DAY on first day of current month
            PointConstraint(type="point", resource="A", time_slot=0, state=1),
            # Block NIGHT→DAY pattern
            PatternBlockConstraint(
                type="pattern_block",
                pattern=["NIGHT", "DAY"],
                resources="ALL",
                state_mapping={"NIGHT": 2, "DAY": 1, "OFF": 0}
            )
        ],
        previous_month_assignments=[
            # A worked NIGHT on last day of previous month
            PreviousMonthAssignment(resource="A", offset_days=1, state=2),
        ]
    )
    
    response = solver_service.solve(request)
    
    # Should be INFEASIBLE since NIGHT→DAY is blocked but required
    assert response.status == "INFEASIBLE"


def test_previous_month_multiple_days():
    """Test with multiple previous month days (e.g., last 3 days)."""
    request = SolveRequest(
        config=ConfigModel(
            resources=["A"],
            time_slots=3,
            states=[0, 1, 2]  # OFF, DAY, NIGHT
        ),
        constraints=[
            PatternBlockConstraint(
                type="pattern_block",
                pattern=["NIGHT", "DAY"],
                resources="ALL",
                state_mapping={"NIGHT": 2, "DAY": 1, "OFF": 0}
            )
        ],
        previous_month_assignments=[
            # A's assignments from previous month
            # offset_days=3: third-to-last day - DAY
            PreviousMonthAssignment(resource="A", offset_days=3, state=1),
            # offset_days=2: second-to-last day - NIGHT
            PreviousMonthAssignment(resource="A", offset_days=2, state=2),
            # offset_days=1: last day - NIGHT
            PreviousMonthAssignment(resource="A", offset_days=1, state=2),
        ]
    )
    
    response = solver_service.solve(request)
    
    assert response.status in ["OPTIMAL", "FEASIBLE"]
    # First day should not be DAY (due to previous NIGHT)
    assert response.schedule["A"][0] != 1


def test_previous_month_allows_valid_transition():
    """Test that valid transitions from previous month are allowed."""
    request = SolveRequest(
        config=ConfigModel(
            resources=["A"],
            time_slots=2,
            states=[0, 1, 2]  # OFF, DAY, NIGHT
        ),
        constraints=[
            # Force DAY on first day
            PointConstraint(type="point", resource="A", time_slot=0, state=1),
            # Block NIGHT→DAY pattern
            PatternBlockConstraint(
                type="pattern_block",
                pattern=["NIGHT", "DAY"],
                resources="ALL",
                state_mapping={"NIGHT": 2, "DAY": 1, "OFF": 0}
            )
        ],
        previous_month_assignments=[
            # A was OFF on last day of previous month (not NIGHT)
            PreviousMonthAssignment(resource="A", offset_days=1, state=0),
        ]
    )
    
    response = solver_service.solve(request)
    
    # Should be FEASIBLE since OFF→DAY is allowed
    assert response.status in ["OPTIMAL", "FEASIBLE"]
    assert response.schedule["A"][0] == 1  # DAY is allowed


def test_previous_month_schedule_output_excludes_overlap():
    """Verify that output schedule only contains current month days."""
    request = SolveRequest(
        config=ConfigModel(
            resources=["A", "B"],
            time_slots=5,
            states=[0, 1, 2]
        ),
        constraints=[],
        previous_month_assignments=[
            PreviousMonthAssignment(resource="A", offset_days=1, state=2),
            PreviousMonthAssignment(resource="A", offset_days=2, state=2),
            PreviousMonthAssignment(resource="B", offset_days=1, state=1),
        ]
    )
    
    response = solver_service.solve(request)
    
    assert response.status in ["OPTIMAL", "FEASIBLE"]
    # Output should only have 5 slots (current month), not 7 (5 + 2 overlap)
    assert len(response.schedule["A"]) == 5
    assert len(response.schedule["B"]) == 5


def test_previous_month_validation_invalid_resource():
    """Test validation catches invalid resource in previous month assignments."""
    with pytest.raises(ValueError, match="not in resources list"):
        SolveRequest(
            config=ConfigModel(
                resources=["A", "B"],
                time_slots=3,
                states=[0, 1, 2]
            ),
            constraints=[],
            previous_month_assignments=[
                PreviousMonthAssignment(resource="Z", offset_days=1, state=1),
            ]
        )


def test_previous_month_validation_invalid_state():
    """Test validation catches invalid state in previous month assignments."""
    with pytest.raises(ValueError, match="not in states"):
        SolveRequest(
            config=ConfigModel(
                resources=["A"],
                time_slots=3,
                states=[0, 1, 2]
            ),
            constraints=[],
            previous_month_assignments=[
                PreviousMonthAssignment(resource="A", offset_days=1, state=5),
            ]
        )


def test_previous_month_sliding_window_enforces_rest():
    """Test that sliding_window constraint enforces rest days after night shift from previous month.
    
    Scenario: NUR006 worked Night (E=2) on Oct 30 (offset_days=2).
    With post_night_rest rule (work_days=1, rest_days=2), they need 2 days off.
    Oct 31 (offset_days=1) was Off (0), so Nov 1 (slot 0) must also be Off.
    """
    request = SolveRequest(
        config=ConfigModel(
            resources=["NUR006"],
            time_slots=5,  # Nov 1-5
            states=[0, 1, 2]  # 0=Off, 1=Day, 2=Night
        ),
        constraints=[
            SlidingWindowConstraint(
                resource="NUR006",
                work_days=1,
                rest_days=2,
                target_state=2  # Night
            )
        ],
        previous_month_assignments=[
            # Oct 30 = Night (E), offset_days=2
            PreviousMonthAssignment(resource="NUR006", offset_days=2, state=2),
            # Oct 31 = Off, offset_days=1
            PreviousMonthAssignment(resource="NUR006", offset_days=1, state=0),
        ]
    )
    
    response = solver_service.solve(request)
    
    assert response.status in ["OPTIMAL", "FEASIBLE"]
    # Nov 1 (slot 0) MUST be Off (state 0) because:
    # Oct 30 was Night → needs 2 rest days → Oct 31 (Off) + Nov 1 (must be Off)
    assert response.schedule["NUR006"][0] == 0, \
        f"Expected Nov 1 to be Off (0), got {response.schedule['NUR006'][0]}"


def test_previous_month_sliding_window_allows_work_after_rest():
    """Test that work is allowed after proper rest period from previous month night.
    
    Scenario: Night on Oct 29 (offset_days=3), Off on Oct 30 and Oct 31.
    Nov 1 can be any state because 2 rest days already taken.
    """
    request = SolveRequest(
        config=ConfigModel(
            resources=["A"],
            time_slots=3,
            states=[0, 1, 2]
        ),
        constraints=[
            SlidingWindowConstraint(
                resource="A",
                work_days=1,
                rest_days=2,
                target_state=2  # Night
            ),
            # Force Nov 1 to be Day (1) - should be allowed
            PointConstraint(
                resource="A",
                time_slot=0,
                state=1
            )
        ],
        previous_month_assignments=[
            # Oct 29 = Night, offset_days=3
            PreviousMonthAssignment(resource="A", offset_days=3, state=2),
            # Oct 30 = Off, offset_days=2
            PreviousMonthAssignment(resource="A", offset_days=2, state=0),
            # Oct 31 = Off, offset_days=1
            PreviousMonthAssignment(resource="A", offset_days=1, state=0),
        ]
    )
    
    response = solver_service.solve(request)
    
    assert response.status in ["OPTIMAL", "FEASIBLE"]
    # Nov 1 should be Day (1) as requested - rest was already satisfied
    assert response.schedule["A"][0] == 1
