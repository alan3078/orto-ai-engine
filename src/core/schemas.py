"""
Pydantic schemas for the Universal Scheduler Solver API.
Defines request and response models with validation.
"""

from typing import List, Literal, Union, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class ConfigModel(BaseModel):
    """Configuration for the scheduling problem."""
    
    resources: List[str] = Field(
        ...,
        min_length=1,
        description="List of resource IDs (e.g., staff/nurse IDs)"
    )
    time_slots: int = Field(
        ...,
        gt=0,
        description="Number of time slots (e.g., days, shifts)"
    )
    states: List[int] = Field(
        ...,
        min_length=2,
        description="Possible states for each slot (e.g., [0, 1] for Off/Work)"
    )
    resource_attributes: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Optional mapping: resource_id -> attribute dict (e.g., gender: 'F', roles: ['IC','RN'])"
    )
    
    @field_validator('states')
    @classmethod
    def validate_states(cls, v):
        """Ensure states start from 0 and are sequential."""
        if v != list(range(len(v))):
            raise ValueError(f"States must be sequential starting from 0, got {v}")
        return v


class PointConstraint(BaseModel):
    """Constraint: Resource X at Time Y must be State Z."""
    
    type: Literal["point"] = "point"
    resource: str = Field(..., description="Resource ID")
    time_slot: int = Field(..., ge=0, description="Time slot index")
    state: int = Field(..., ge=0, description="Required state value")


class VerticalSumConstraint(BaseModel):
    """Constraint: Sum of specific state across resources at time slot(s)."""
    
    type: Literal["vertical_sum"] = "vertical_sum"
    time_slot: Union[int, Literal["ALL"]] = Field(
        ...,
        description="Time slot index or 'ALL' for all time slots"
    )
    target_state: int = Field(..., ge=0, description="State to count")
    operator: Literal[">=", "<=", "=="] = Field(..., description="Comparison operator")
    value: int = Field(..., ge=0, description="Target value for comparison")


class HorizontalSumConstraint(BaseModel):
    """Constraint: Count consecutive occurrences of state for a single resource."""
    
    type: Literal["horizontal_sum"] = "horizontal_sum"
    resource: str = Field(..., description="Resource ID")
    time_slots: List[int] = Field(
        ...,
        min_length=1,
        description="Consecutive time slots to evaluate"
    )
    target_state: int = Field(..., ge=0, description="State to count")
    operator: Literal[">=", "<=", "=="] = Field(..., description="Comparison operator")
    value: int = Field(..., ge=0, description="Maximum/minimum consecutive occurrences")


class SlidingWindowConstraint(BaseModel):
    """Constraint: Enforce work-rest patterns (e.g., 5 days work, 2 days rest)."""
    
    type: Literal["sliding_window"] = "sliding_window"
    resource: str = Field(..., description="Resource ID")
    work_days: int = Field(..., gt=0, description="Number of consecutive work days")
    rest_days: int = Field(..., gt=0, description="Number of required rest days after work")
    target_state: int = Field(..., ge=1, description="State representing 'work' (typically 1)")


class PatternBlockConstraint(BaseModel):
    """Constraint: Block specific state transition patterns (e.g., Night→Day)."""
    
    type: Literal["pattern_block"] = "pattern_block"
    pattern: List[str] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Two-element pattern to block (e.g., ['NIGHT', 'DAY'])"
    )
    resources: Literal["ALL"] = Field(
        "ALL",
        description="Apply to all resources (currently only 'ALL' supported)"
    )
    state_mapping: Optional[dict[str, int]] = Field(
        None,
        description="Optional mapping from pattern names to state integers (e.g., {'NIGHT': 2, 'DAY': 1})"
    )


class AttributeVerticalSumConstraint(BaseModel):
    """Constraint: Sum of specific state across filtered resources by attribute.
    Example: At least 1 female (gender=F) working (state=1) each time slot.
    """

    type: Literal["attribute_vertical_sum"] = "attribute_vertical_sum"
    time_slot: Union[int, Literal["ALL"]] = Field(
        ...,
        description="Time slot index or 'ALL' for all time slots"
    )
    target_state: int = Field(..., ge=0, description="State to count")
    operator: Literal[">=", "<=", "=="] = Field(..., description="Comparison operator")
    value: int = Field(..., ge=0, description="Target value for comparison")
    attribute: str = Field(..., description="Attribute key to filter (e.g., gender, role)")
    attribute_values: List[str] = Field(
        ..., min_length=1, description="Allowed attribute values to include in filtered set"
    )

class ResourceStateCountConstraint(BaseModel):
    """Constraint: Total occurrences of a state for a resource across specified time slots.
    Used for enforcing monthly/night shift counts (e.g., 4–6 night shifts).
    """

    type: Literal["resource_state_count"] = "resource_state_count"
    resource: str = Field(..., description="Resource ID")
    time_slots: List[int] = Field(..., min_length=1, description="List of time slots to evaluate")
    target_state: int = Field(..., ge=0, description="State to count")
    operator: Literal[">=", "<=", "=="] = Field(..., description="Comparison operator")
    value: int = Field(..., ge=0, description="Required total occurrences (comparison target)")


ConstraintModel = Union[
    PointConstraint,
    VerticalSumConstraint,
    HorizontalSumConstraint,
    SlidingWindowConstraint,
    PatternBlockConstraint,
    AttributeVerticalSumConstraint,
    ResourceStateCountConstraint,
]


class SolveRequest(BaseModel):
    """Request model for the solve endpoint."""
    
    config: ConfigModel
    constraints: List[ConstraintModel] = Field(
        default_factory=list,
        description="List of constraints to apply"
    )
    
    @field_validator('constraints')
    @classmethod
    def validate_constraints(cls, v, info):
        """Validate constraints against config."""
        if 'config' not in info.data:
            return v
            
        config = info.data['config']
        
        for constraint in v:
            if isinstance(constraint, PointConstraint):
                # Validate resource exists
                if constraint.resource not in config.resources:
                    raise ValueError(
                        f"Resource '{constraint.resource}' not in resources list"
                    )
                # Validate time_slot in range
                if constraint.time_slot >= config.time_slots:
                    raise ValueError(
                        f"time_slot {constraint.time_slot} >= time_slots {config.time_slots}"
                    )
                # Validate state in range
                if constraint.state not in config.states:
                    raise ValueError(
                        f"state {constraint.state} not in states {config.states}"
                    )
                    
            elif isinstance(constraint, VerticalSumConstraint):
                # Validate time_slot if not "ALL"
                if constraint.time_slot != "ALL":
                    if constraint.time_slot >= config.time_slots:
                        raise ValueError(
                            f"time_slot {constraint.time_slot} >= time_slots {config.time_slots}"
                        )
                # Validate target_state
                if constraint.target_state not in config.states:
                    raise ValueError(
                        f"target_state {constraint.target_state} not in states {config.states}"
                    )
                # Validate value not exceeding resource count
                if constraint.value > len(config.resources):
                    raise ValueError(
                        f"value {constraint.value} > number of resources {len(config.resources)}"
                    )
                    
            elif isinstance(constraint, HorizontalSumConstraint):
                # Validate resource exists
                if constraint.resource not in config.resources:
                    raise ValueError(
                        f"Resource '{constraint.resource}' not in resources list"
                    )
                # Validate all time_slots in range
                for ts in constraint.time_slots:
                    if ts >= config.time_slots:
                        raise ValueError(
                            f"time_slot {ts} >= time_slots {config.time_slots}"
                        )
                # Validate target_state
                if constraint.target_state not in config.states:
                    raise ValueError(
                        f"target_state {constraint.target_state} not in states {config.states}"
                    )
                # Validate value not exceeding time_slots length
                if constraint.value > len(constraint.time_slots):
                    raise ValueError(
                        f"value {constraint.value} > time_slots length {len(constraint.time_slots)}"
                    )
                    
            elif isinstance(constraint, SlidingWindowConstraint):
                # Validate resource exists
                if constraint.resource not in config.resources:
                    raise ValueError(
                        f"Resource '{constraint.resource}' not in resources list"
                    )
                # Validate target_state
                if constraint.target_state not in config.states:
                    raise ValueError(
                        f"target_state {constraint.target_state} not in states {config.states}"
                    )
                # Validate pattern fits in roster
                pattern_length = constraint.work_days + constraint.rest_days
                if pattern_length > config.time_slots:
                    raise ValueError(
                        f"Pattern length ({pattern_length}) > time_slots {config.time_slots}"
                    )
                    
            elif isinstance(constraint, PatternBlockConstraint):
                # Validate pattern length
                if len(constraint.pattern) != 2:
                    raise ValueError(
                        f"Pattern must have exactly 2 elements, got {len(constraint.pattern)}"
                    )
                # Validate state_mapping if provided
                if constraint.state_mapping:
                    for state_val in constraint.state_mapping.values():
                        if state_val not in config.states:
                            raise ValueError(
                                f"state_mapping value {state_val} not in states {config.states}"
                            )
            elif isinstance(constraint, AttributeVerticalSumConstraint):
                # Validate time slot
                if constraint.time_slot != "ALL" and constraint.time_slot >= config.time_slots:
                    raise ValueError(
                        f"time_slot {constraint.time_slot} >= time_slots {config.time_slots}"
                    )
                if constraint.target_state not in config.states:
                    raise ValueError(
                        f"target_state {constraint.target_state} not in states {config.states}"
                    )
                if constraint.value > len(config.resources):
                    raise ValueError(
                        f"value {constraint.value} > number of resources {len(config.resources)}"
                    )
                # Attribute presence check (best-effort)
                if config.resource_attributes is None:
                    raise ValueError("attribute_vertical_sum requires resource_attributes in config")
            elif isinstance(constraint, ResourceStateCountConstraint):
                if constraint.resource not in config.resources:
                    raise ValueError(
                        f"Resource '{constraint.resource}' not in resources list"
                    )
                for ts in constraint.time_slots:
                    if ts >= config.time_slots:
                        raise ValueError(
                            f"time_slot {ts} >= time_slots {config.time_slots}"
                        )
                if constraint.target_state not in config.states:
                    raise ValueError(
                        f"target_state {constraint.target_state} not in states {config.states}"
                    )
                if constraint.value > len(constraint.time_slots):
                    raise ValueError(
                        f"value {constraint.value} > time_slots length {len(constraint.time_slots)}"
                    )
        
        return v


class SolveResponse(BaseModel):
    """Response model for the solve endpoint."""
    
    status: Literal["OPTIMAL", "FEASIBLE", "INFEASIBLE", "ERROR"] = Field(
        ...,
        description="Solver status"
    )
    schedule: Optional[dict[str, List[int]]] = Field(
        None,
        description="Schedule matrix: resource_id -> [state_per_timeslot]"
    )
    message: Optional[str] = Field(
        None,
        description="Additional information or error message"
    )
    solve_time_ms: Optional[float] = Field(
        None,
        description="Time taken to solve in milliseconds"
    )


# ============================================================================
# Validation Schemas (FN/BE/ENG/002 - Roster Validator)
# ============================================================================


class ValidateRequest(BaseModel):
    """Request model for the validate endpoint."""
    
    config: ConfigModel
    schedule: Dict[str, List[int]] = Field(
        ...,
        description="Existing schedule to validate: resource_id -> [state_per_timeslot]"
    )
    constraints: List[ConstraintModel] = Field(
        default_factory=list,
        description="List of constraints to validate against"
    )
    constraint_names: Optional[List[str]] = Field(
        None,
        description="Optional list of human-readable constraint names (same order as constraints)"
    )
    
    @field_validator('schedule')
    @classmethod
    def validate_schedule(cls, v, info):
        """Validate schedule matrix matches config."""
        if 'config' not in info.data:
            return v
            
        config = info.data['config']
        
        # Check all resources are present
        if set(v.keys()) != set(config.resources):
            raise ValueError(
                f"Schedule resources {set(v.keys())} don't match config resources {set(config.resources)}"
            )
        
        # Check all state arrays have correct length and valid states
        for resource_id, states in v.items():
            if len(states) != config.time_slots:
                raise ValueError(
                    f"Resource {resource_id} has {len(states)} time slots, expected {config.time_slots}"
                )
            for state in states:
                if state not in config.states:
                    raise ValueError(
                        f"Resource {resource_id} has invalid state {state}, allowed states: {config.states}"
                    )
        
        return v


class ConstraintValidationResult(BaseModel):
    """Result for a single constraint validation."""
    
    constraint_index: int = Field(..., description="Index of constraint in input array")
    constraint_type: str = Field(..., description="Type of constraint (point, vertical_sum, etc.)")
    constraint_name: Optional[str] = Field(None, description="Human-readable name if provided")
    status: Literal["PASS", "FAIL"] = Field(..., description="Validation status")
    details: Optional[str] = Field(None, description="Human-readable explanation of result")
    violations: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="List of specific violations (resource, time_slot, expected, actual)"
    )


class ValidateResponse(BaseModel):
    """Response model for the validate endpoint."""
    
    overall_status: Literal["PASS", "FAIL"] = Field(
        ...,
        description="PASS if all constraints pass, FAIL if any fail"
    )
    total_constraints: int = Field(..., description="Total number of constraints checked")
    passed_constraints: int = Field(..., description="Number of constraints that passed")
    failed_constraints: int = Field(..., description="Number of constraints that failed")
    results: List[ConstraintValidationResult] = Field(
        ...,
        description="Detailed results for each constraint"
    )
    validation_time_ms: Optional[float] = Field(
        None,
        description="Time taken to validate in milliseconds"
    )
