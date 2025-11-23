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
