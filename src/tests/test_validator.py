"""
Unit tests for the constraint validator service.

Part of FN/BE/ENG/002 - Roster Validator (Audit Mode)
"""

import pytest
from src.core.schemas import (
    ConfigModel,
    PointConstraint,
    VerticalSumConstraint,
    HorizontalSumConstraint,
    SlidingWindowConstraint,
    PatternBlockConstraint,
    AttributeVerticalSumConstraint,
    ResourceStateCountConstraint,
)
from src.services.validator import ConstraintValidator


# ============================================================================
# Point Constraint Tests
# ============================================================================

def test_validate_point_pass():
    """Test point constraint validation - PASS case"""
    config = ConfigModel(resources=["A", "B"], time_slots=3, states=[0, 1])
    schedule = {"A": [0, 1, 0], "B": [1, 0, 1]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = PointConstraint(resource="A", time_slot=1, state=1)
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "PASS"
    assert len(result.violations) == 0
    assert "A at time 1 is state 1" in result.details


def test_validate_point_fail():
    """Test point constraint validation - FAIL case"""
    config = ConfigModel(resources=["A", "B"], time_slots=3, states=[0, 1])
    schedule = {"A": [0, 1, 0], "B": [1, 0, 1]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = PointConstraint(resource="A", time_slot=1, state=0)
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "FAIL"
    assert len(result.violations) == 1
    assert result.violations[0]["actual"] == 1
    assert result.violations[0]["expected"] == 0


# ============================================================================
# Vertical Sum Constraint Tests
# ============================================================================

def test_validate_vertical_sum_all_pass():
    """Test vertical sum constraint with ALL time slots - PASS"""
    config = ConfigModel(resources=["A", "B", "C"], time_slots=3, states=[0, 1])
    schedule = {"A": [1, 1, 0], "B": [1, 0, 1], "C": [0, 1, 1]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = VerticalSumConstraint(
        time_slot="ALL",
        target_state=1,
        operator=">=",
        value=2
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "PASS"
    assert len(result.violations) == 0


def test_validate_vertical_sum_all_fail():
    """Test vertical sum constraint with ALL time slots - FAIL"""
    config = ConfigModel(resources=["A", "B"], time_slots=3, states=[0, 1])
    schedule = {"A": [0, 0, 0], "B": [0, 1, 0]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = VerticalSumConstraint(
        time_slot="ALL",
        target_state=1,
        operator=">=",
        value=2
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "FAIL"
    assert len(result.violations) == 3  # Fails at all slots (0, 1, 2) - only 0 or 1 workers
    assert result.violations[0]["time_slot"] == 0
    assert result.violations[1]["time_slot"] == 1
    assert result.violations[2]["time_slot"] == 2


def test_validate_vertical_sum_specific_slot_pass():
    """Test vertical sum constraint for specific time slot - PASS"""
    config = ConfigModel(resources=["A", "B", "C"], time_slots=3, states=[0, 1])
    schedule = {"A": [1, 0, 0], "B": [1, 0, 0], "C": [1, 0, 0]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = VerticalSumConstraint(
        time_slot=0,
        target_state=1,
        operator="==",
        value=3
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "PASS"


def test_validate_vertical_sum_less_than_or_equal():
    """Test vertical sum with <= operator"""
    config = ConfigModel(resources=["A", "B"], time_slots=2, states=[0, 1])
    schedule = {"A": [1, 1], "B": [0, 1]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = VerticalSumConstraint(
        time_slot="ALL",
        target_state=1,
        operator="<=",
        value=2
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "PASS"


# ============================================================================
# Horizontal Sum Constraint Tests
# ============================================================================

def test_validate_horizontal_sum_pass():
    """Test horizontal sum constraint - PASS"""
    config = ConfigModel(resources=["A", "B"], time_slots=7, states=[0, 1])
    schedule = {"A": [1, 1, 1, 0, 0, 1, 0], "B": [0, 0, 0, 1, 1, 1, 1]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = HorizontalSumConstraint(
        resource="A",
        time_slots=[0, 1, 2, 3, 4, 5, 6],
        target_state=1,
        operator="<=",
        value=5
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "PASS"
    assert "4 occurrences" in result.details


def test_validate_horizontal_sum_fail():
    """Test horizontal sum constraint - FAIL"""
    config = ConfigModel(resources=["A"], time_slots=7, states=[0, 1])
    schedule = {"A": [1, 1, 1, 1, 1, 1, 0]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = HorizontalSumConstraint(
        resource="A",
        time_slots=[0, 1, 2, 3, 4, 5, 6],
        target_state=1,
        operator="<=",
        value=3
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "FAIL"
    assert len(result.violations) == 1
    assert result.violations[0]["actual"] == 6


# ============================================================================
# Sliding Window Constraint Tests
# ============================================================================

def test_validate_sliding_window_pass():
    """Test sliding window constraint - PASS (proper rest after work)"""
    config = ConfigModel(resources=["A"], time_slots=7, states=[0, 1])
    schedule = {"A": [1, 1, 1, 0, 0, 1, 0]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = SlidingWindowConstraint(
        resource="A",
        work_days=3,
        rest_days=2,
        target_state=1
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "PASS"


def test_validate_sliding_window_fail():
    """Test sliding window constraint - FAIL (insufficient rest)"""
    config = ConfigModel(resources=["A"], time_slots=7, states=[0, 1])
    schedule = {"A": [1, 1, 1, 1, 0, 1, 1]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = SlidingWindowConstraint(
        resource="A",
        work_days=3,
        rest_days=2,
        target_state=1
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "FAIL"
    assert len(result.violations) > 0
    assert "rest days" in result.violations[0]["issue"]


# ============================================================================
# Pattern Block Constraint Tests
# ============================================================================

def test_validate_pattern_block_pass():
    """Test pattern block constraint - PASS (no forbidden transitions)"""
    config = ConfigModel(resources=["A", "B"], time_slots=4, states=[0, 1, 2])
    schedule = {"A": [0, 1, 1, 0], "B": [1, 2, 0, 1]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = PatternBlockConstraint(
        pattern=["NIGHT", "DAY"],
        resources="ALL",
        state_mapping={"NIGHT": 2, "DAY": 1}
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "PASS"


def test_validate_pattern_block_fail():
    """Test pattern block constraint - FAIL (forbidden transition found)"""
    config = ConfigModel(resources=["A", "B"], time_slots=4, states=[0, 1, 2])
    schedule = {"A": [0, 2, 1, 0], "B": [1, 0, 2, 1]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = PatternBlockConstraint(
        pattern=["NIGHT", "DAY"],
        resources="ALL",
        state_mapping={"NIGHT": 2, "DAY": 1}
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "FAIL"
    assert len(result.violations) >= 2  # Both A and B have violations
    assert "transition forbidden" in result.violations[0]["issue"]


def test_validate_pattern_block_invalid_mapping():
    """Test pattern block with invalid state mapping"""
    config = ConfigModel(resources=["A"], time_slots=3, states=[0, 1])
    schedule = {"A": [0, 1, 0]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = PatternBlockConstraint(
        pattern=["NIGHT", "DAY"],
        resources="ALL",
        state_mapping={"NIGHT": 99}  # Missing DAY mapping
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "FAIL"
    assert "Invalid state_mapping" in result.details


# ============================================================================
# Attribute Vertical Sum Constraint Tests
# ============================================================================

def test_validate_attribute_vertical_sum_pass():
    """Test attribute vertical sum constraint - PASS"""
    config = ConfigModel(
        resources=["A", "B", "C"],
        time_slots=3,
        states=[0, 1],
        resource_attributes={
            "A": {"gender": "F", "roles": ["IC"]},
            "B": {"gender": "M", "roles": ["RN"]},
            "C": {"gender": "F", "roles": ["RN"]}
        }
    )
    schedule = {"A": [1, 1, 0], "B": [0, 1, 1], "C": [1, 0, 1]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = AttributeVerticalSumConstraint(
        time_slot="ALL",
        target_state=1,
        operator=">=",
        value=1,
        attribute="gender",
        attribute_values=["F"]
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "PASS"


def test_validate_attribute_vertical_sum_fail():
    """Test attribute vertical sum constraint - FAIL"""
    config = ConfigModel(
        resources=["A", "B"],
        time_slots=3,
        states=[0, 1],
        resource_attributes={
            "A": {"gender": "F"},
            "B": {"gender": "M"}
        }
    )
    schedule = {"A": [0, 0, 0], "B": [1, 1, 1]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = AttributeVerticalSumConstraint(
        time_slot="ALL",
        target_state=1,
        operator=">=",
        value=1,
        attribute="gender",
        attribute_values=["F"]
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "FAIL"
    assert len(result.violations) == 3  # All time slots fail


def test_validate_attribute_vertical_sum_with_roles():
    """Test attribute vertical sum with roles filter"""
    config = ConfigModel(
        resources=["A", "B", "C"],
        time_slots=2,
        states=[0, 1],
        resource_attributes={
            "A": {"roles": ["IC"]},
            "B": {"roles": ["RN"]},
            "C": {"roles": ["RN"]}
        }
    )
    schedule = {"A": [1, 0], "B": [1, 1], "C": [1, 1]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = AttributeVerticalSumConstraint(
        time_slot="ALL",
        target_state=1,
        operator=">=",
        value=2,
        attribute="roles",
        attribute_values=["RN"]
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "PASS"


# ============================================================================
# Resource State Count Constraint Tests
# ============================================================================

def test_validate_resource_state_count_pass():
    """Test resource state count constraint - PASS"""
    config = ConfigModel(resources=["A"], time_slots=7, states=[0, 1, 2])
    schedule = {"A": [0, 1, 2, 2, 1, 0, 2]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = ResourceStateCountConstraint(
        resource="A",
        time_slots=[0, 1, 2, 3, 4, 5, 6],
        target_state=2,
        operator=">=",
        value=3
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "PASS"
    assert "3 occurrences" in result.details


def test_validate_resource_state_count_fail():
    """Test resource state count constraint - FAIL"""
    config = ConfigModel(resources=["A"], time_slots=7, states=[0, 1, 2])
    schedule = {"A": [0, 1, 2, 2, 1, 0, 0]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = ResourceStateCountConstraint(
        resource="A",
        time_slots=[0, 1, 2, 3, 4, 5, 6],
        target_state=2,
        operator=">=",
        value=5
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "FAIL"
    assert len(result.violations) == 1
    assert result.violations[0]["actual"] == 2


def test_validate_resource_state_count_exact_match():
    """Test resource state count with == operator"""
    config = ConfigModel(resources=["A"], time_slots=5, states=[0, 1])
    schedule = {"A": [1, 1, 1, 0, 0]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = ResourceStateCountConstraint(
        resource="A",
        time_slots=[0, 1, 2, 3, 4],
        target_state=1,
        operator="==",
        value=3
    )
    result = validator.validate_constraint(constraint, 0)
    
    assert result.status == "PASS"


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================

def test_validate_empty_constraints():
    """Test validation with no constraints (should pass)"""
    config = ConfigModel(resources=["A"], time_slots=3, states=[0, 1])
    schedule = {"A": [0, 1, 0]}
    validator = ConstraintValidator(config, schedule)
    
    results = []
    for idx, constraint in enumerate([]):
        result = validator.validate_constraint(constraint, idx)
        results.append(result)
    
    assert len(results) == 0


def test_validate_multiple_constraints_mixed():
    """Test validation with multiple constraints - mixed PASS/FAIL"""
    config = ConfigModel(resources=["A", "B"], time_slots=3, states=[0, 1])
    schedule = {"A": [0, 1, 0], "B": [1, 0, 1]}
    validator = ConstraintValidator(config, schedule)
    
    constraints = [
        PointConstraint(resource="A", time_slot=0, state=0),  # PASS
        PointConstraint(resource="B", time_slot=0, state=0),  # FAIL
        VerticalSumConstraint(time_slot="ALL", target_state=1, operator=">=", value=1)  # PASS
    ]
    
    results = []
    for idx, constraint in enumerate(constraints):
        result = validator.validate_constraint(constraint, idx)
        results.append(result)
    
    assert len(results) == 3
    assert results[0].status == "PASS"
    assert results[1].status == "FAIL"
    assert results[2].status == "PASS"


def test_validate_constraint_with_name():
    """Test that constraint names are preserved in results"""
    config = ConfigModel(resources=["A"], time_slots=2, states=[0, 1])
    schedule = {"A": [0, 1]}
    validator = ConstraintValidator(config, schedule)
    
    constraint = PointConstraint(resource="A", time_slot=0, state=0)
    result = validator.validate_constraint(constraint, 0, name="Amy's Day Off")
    
    assert result.constraint_name == "Amy's Day Off"
    assert result.status == "PASS"


def test_check_operator_helper():
    """Test the _check_operator static method"""
    assert ConstraintValidator._check_operator(5, ">=", 3) == True
    assert ConstraintValidator._check_operator(5, ">=", 5) == True
    assert ConstraintValidator._check_operator(5, ">=", 6) == False
    
    assert ConstraintValidator._check_operator(3, "<=", 5) == True
    assert ConstraintValidator._check_operator(5, "<=", 5) == True
    assert ConstraintValidator._check_operator(6, "<=", 5) == False
    
    assert ConstraintValidator._check_operator(5, "==", 5) == True
    assert ConstraintValidator._check_operator(5, "==", 4) == False
    
    assert ConstraintValidator._check_operator(5, "invalid", 5) == False
