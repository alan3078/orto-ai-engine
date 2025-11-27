"""
Validator service for checking constraints against existing schedules.
Uses imperative validation (no OR-Tools) for performance.

Part of FN/BE/ENG/002 - Roster Validator (Audit Mode)
"""

from typing import List, Dict, Any, Optional
from src.core.schemas import (
    ConfigModel,
    ConstraintModel,
    PointConstraint,
    VerticalSumConstraint,
    HorizontalSumConstraint,
    SlidingWindowConstraint,
    PatternBlockConstraint,
    AttributeVerticalSumConstraint,
    ResourceStateCountConstraint,
    CompoundAttributeVerticalSumConstraint,
    ConstraintValidationResult,
)


class ConstraintValidator:
    """Validates constraints against an existing schedule matrix."""
    
    def __init__(self, config: ConfigModel, schedule: Dict[str, List[int]]):
        """
        Initialize validator with configuration and schedule.
        
        Args:
            config: Configuration including resources, time_slots, states
            schedule: Schedule matrix {resource_id: [state_per_timeslot]}
        """
        self.config = config
        self.schedule = schedule
        self.resource_attrs = config.resource_attributes or {}
    
    def validate_constraint(
        self, 
        constraint: ConstraintModel, 
        index: int,
        name: Optional[str] = None
    ) -> ConstraintValidationResult:
        """
        Validate a single constraint against the schedule.
        
        Args:
            constraint: Constraint to validate
            index: Index of constraint in input array
            name: Optional human-readable name for the constraint
            
        Returns:
            ConstraintValidationResult with status and details
        """
        
        if isinstance(constraint, PointConstraint):
            return self._validate_point(constraint, index, name)
        elif isinstance(constraint, VerticalSumConstraint):
            return self._validate_vertical_sum(constraint, index, name)
        elif isinstance(constraint, HorizontalSumConstraint):
            return self._validate_horizontal_sum(constraint, index, name)
        elif isinstance(constraint, SlidingWindowConstraint):
            return self._validate_sliding_window(constraint, index, name)
        elif isinstance(constraint, PatternBlockConstraint):
            return self._validate_pattern_block(constraint, index, name)
        elif isinstance(constraint, AttributeVerticalSumConstraint):
            return self._validate_attribute_vertical_sum(constraint, index, name)
        elif isinstance(constraint, ResourceStateCountConstraint):
            return self._validate_resource_state_count(constraint, index, name)
        elif isinstance(constraint, CompoundAttributeVerticalSumConstraint):
            return self._validate_compound_attribute_vertical_sum(constraint, index, name)
        else:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="unknown",
                constraint_name=name,
                status="FAIL",
                details=f"Unsupported constraint type: {type(constraint).__name__}",
                violations=[]
            )
    
    def _validate_point(
        self, 
        constraint: PointConstraint, 
        index: int,
        name: Optional[str]
    ) -> ConstraintValidationResult:
        """Validate point constraint: Resource X at Time Y must be State Z."""
        actual_state = self.schedule[constraint.resource][constraint.time_slot]
        
        if actual_state == constraint.state:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="point",
                constraint_name=name,
                status="PASS",
                details=f"{constraint.resource} at time {constraint.time_slot} is state {constraint.state} ✓",
                violations=[]
            )
        else:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="point",
                constraint_name=name,
                status="FAIL",
                details=f"{constraint.resource} at time {constraint.time_slot} is state {actual_state}, expected {constraint.state}",
                violations=[{
                    "resource": constraint.resource,
                    "time_slot": constraint.time_slot,
                    "expected": constraint.state,
                    "actual": actual_state
                }]
            )
    
    def _validate_vertical_sum(
        self, 
        constraint: VerticalSumConstraint, 
        index: int,
        name: Optional[str]
    ) -> ConstraintValidationResult:
        """Validate vertical sum: Count of target_state across resources at time slot(s)."""
        violations = []
        
        time_slots = (
            range(self.config.time_slots) 
            if constraint.time_slot == "ALL" 
            else [constraint.time_slot]
        )
        
        for ts in time_slots:
            count = sum(
                1 for resource_states in self.schedule.values() 
                if resource_states[ts] == constraint.target_state
            )
            
            # Check operator
            passes = self._check_operator(count, constraint.operator, constraint.value)
            
            if not passes:
                violations.append({
                    "time_slot": ts,
                    "expected": f"{constraint.operator} {constraint.value}",
                    "actual": count,
                    "target_state": constraint.target_state
                })
        
        if violations:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="vertical_sum",
                constraint_name=name,
                status="FAIL",
                details=f"Failed at {len(violations)} time slot(s): {', '.join(str(v['time_slot']) for v in violations)}",
                violations=violations
            )
        else:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="vertical_sum",
                constraint_name=name,
                status="PASS",
                details=f"All time slots meet {constraint.operator} {constraint.value} workers in state {constraint.target_state} ✓",
                violations=[]
            )
    
    def _validate_horizontal_sum(
        self, 
        constraint: HorizontalSumConstraint, 
        index: int,
        name: Optional[str]
    ) -> ConstraintValidationResult:
        """Validate horizontal sum: Count of target_state for resource across time slots."""
        resource_states = self.schedule[constraint.resource]
        count = sum(
            1 for ts in constraint.time_slots 
            if resource_states[ts] == constraint.target_state
        )
        
        passes = self._check_operator(count, constraint.operator, constraint.value)
        
        if passes:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="horizontal_sum",
                constraint_name=name,
                status="PASS",
                details=f"{constraint.resource} has {count} occurrences of state {constraint.target_state} (meets {constraint.operator} {constraint.value}) ✓",
                violations=[]
            )
        else:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="horizontal_sum",
                constraint_name=name,
                status="FAIL",
                details=f"{constraint.resource} has {count} occurrences of state {constraint.target_state}, expected {constraint.operator} {constraint.value}",
                violations=[{
                    "resource": constraint.resource,
                    "time_slots": constraint.time_slots,
                    "expected": f"{constraint.operator} {constraint.value}",
                    "actual": count,
                    "target_state": constraint.target_state
                }]
            )
    
    def _validate_sliding_window(
        self, 
        constraint: SlidingWindowConstraint, 
        index: int,
        name: Optional[str]
    ) -> ConstraintValidationResult:
        """Validate sliding window: work-rest pattern enforcement."""
        resource_states = self.schedule[constraint.resource]
        violations = []
        
        window_size = constraint.work_days + constraint.rest_days
        
        for start in range(len(resource_states) - window_size + 1):
            window = resource_states[start:start + window_size]
            work_days_count = sum(1 for s in window[:constraint.work_days] if s == constraint.target_state)
            
            # If work_days period is fully worked, check rest_days
            if work_days_count == constraint.work_days:
                rest_days_count = sum(1 for s in window[constraint.work_days:] if s != constraint.target_state)
                if rest_days_count < constraint.rest_days:
                    violations.append({
                        "resource": constraint.resource,
                        "start_time_slot": start,
                        "end_time_slot": start + window_size - 1,
                        "issue": f"After {constraint.work_days} work days, only {rest_days_count}/{constraint.rest_days} rest days"
                    })
        
        if violations:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="sliding_window",
                constraint_name=name,
                status="FAIL",
                details=f"{constraint.resource} violates {constraint.work_days} work / {constraint.rest_days} rest pattern at {len(violations)} position(s)",
                violations=violations
            )
        else:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="sliding_window",
                constraint_name=name,
                status="PASS",
                details=f"{constraint.resource} meets {constraint.work_days} work / {constraint.rest_days} rest pattern ✓",
                violations=[]
            )
    
    def _validate_pattern_block(
        self, 
        constraint: PatternBlockConstraint, 
        index: int,
        name: Optional[str]
    ) -> ConstraintValidationResult:
        """Validate pattern block: Ensure forbidden transitions don't occur."""
        violations = []
        
        # Get state mapping
        state_map = constraint.state_mapping or {}
        pattern_states = [state_map.get(p, None) for p in constraint.pattern]
        
        if None in pattern_states:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="pattern_block",
                constraint_name=name,
                status="FAIL",
                details=f"Invalid state_mapping for pattern {constraint.pattern}",
                violations=[]
            )
        
        # Check all resources
        for resource_id, resource_states in self.schedule.items():
            for t in range(len(resource_states) - 1):
                if (resource_states[t] == pattern_states[0] and 
                    resource_states[t + 1] == pattern_states[1]):
                    violations.append({
                        "resource": resource_id,
                        "time_slot": t,
                        "pattern": constraint.pattern,
                        "issue": f"{constraint.pattern[0]} → {constraint.pattern[1]} transition forbidden"
                    })
        
        if violations:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="pattern_block",
                constraint_name=name,
                status="FAIL",
                details=f"Found {len(violations)} forbidden {constraint.pattern[0]}→{constraint.pattern[1]} transitions",
                violations=violations
            )
        else:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="pattern_block",
                constraint_name=name,
                status="PASS",
                details=f"No forbidden {constraint.pattern[0]}→{constraint.pattern[1]} transitions ✓",
                violations=[]
            )
    
    def _validate_attribute_vertical_sum(
        self, 
        constraint: AttributeVerticalSumConstraint, 
        index: int,
        name: Optional[str]
    ) -> ConstraintValidationResult:
        """Validate attribute vertical sum: Filtered resource count by attribute."""
        # Filter resources by attribute
        filtered_resources = []
        for res_id in self.config.resources:
            if res_id not in self.resource_attrs:
                continue
            if constraint.attribute not in self.resource_attrs[res_id]:
                continue
            
            res_attr_value = self.resource_attrs[res_id][constraint.attribute]
            # Handle both single values and lists
            if isinstance(res_attr_value, list):
                # Check if any value in the list matches any allowed value
                if any(val in constraint.attribute_values for val in res_attr_value):
                    filtered_resources.append(res_id)
            else:
                # Direct comparison for scalar values
                if res_attr_value in constraint.attribute_values:
                    filtered_resources.append(res_id)
        
        violations = []
        time_slots = (
            range(self.config.time_slots) 
            if constraint.time_slot == "ALL" 
            else [constraint.time_slot]
        )
        
        for ts in time_slots:
            count = sum(
                1 for res_id in filtered_resources
                if self.schedule[res_id][ts] == constraint.target_state
            )
            
            passes = self._check_operator(count, constraint.operator, constraint.value)
            
            if not passes:
                violations.append({
                    "time_slot": ts,
                    "expected": f"{constraint.operator} {constraint.value}",
                    "actual": count,
                    "attribute_filter": f"{constraint.attribute} in {constraint.attribute_values}"
                })
        
        if violations:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="attribute_vertical_sum",
                constraint_name=name,
                status="FAIL",
                details=f"Failed at {len(violations)} time slot(s) for {constraint.attribute} filter",
                violations=violations
            )
        else:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="attribute_vertical_sum",
                constraint_name=name,
                status="PASS",
                details=f"All time slots meet {constraint.operator} {constraint.value} filtered workers ✓",
                violations=[]
            )
    
    def _validate_resource_state_count(
        self, 
        constraint: ResourceStateCountConstraint, 
        index: int,
        name: Optional[str]
    ) -> ConstraintValidationResult:
        """Validate resource state count: Total occurrences of state for resource."""
        resource_states = self.schedule[constraint.resource]
        count = sum(
            1 for ts in constraint.time_slots
            if resource_states[ts] == constraint.target_state
        )
        
        passes = self._check_operator(count, constraint.operator, constraint.value)
        
        if passes:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="resource_state_count",
                constraint_name=name,
                status="PASS",
                details=f"{constraint.resource} has {count} occurrences of state {constraint.target_state} (meets {constraint.operator} {constraint.value}) ✓",
                violations=[]
            )
        else:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="resource_state_count",
                constraint_name=name,
                status="FAIL",
                details=f"{constraint.resource} has {count} occurrences of state {constraint.target_state}, expected {constraint.operator} {constraint.value}",
                violations=[{
                    "resource": constraint.resource,
                    "expected": f"{constraint.operator} {constraint.value}",
                    "actual": count,
                    "target_state": constraint.target_state
                }]
            )

    def _validate_compound_attribute_vertical_sum(
        self,
        constraint: CompoundAttributeVerticalSumConstraint,
        index: int,
        name: Optional[str]
    ) -> ConstraintValidationResult:
        """Validate compound attribute vertical sum: Filtered resource count by multiple attributes (AND logic)."""
        # Filter resources by ALL attributes (AND logic)
        filtered_resources = []
        for res_id in self.config.resources:
            if res_id not in self.resource_attrs:
                continue
            
            # Check all attribute filters (AND logic)
            all_match = True
            for attr_key, attr_values in constraint.attribute_filters.items():
                if attr_key not in self.resource_attrs[res_id]:
                    all_match = False
                    break
                
                res_attr_value = self.resource_attrs[res_id][attr_key]
                attr_values_set = set(attr_values)
                
                # Handle both single values and lists
                if isinstance(res_attr_value, list):
                    match = any(str(val) in attr_values_set for val in res_attr_value)
                else:
                    match = str(res_attr_value) in attr_values_set
                
                if not match:
                    all_match = False
                    break
            
            if all_match:
                filtered_resources.append(res_id)
        
        violations = []
        time_slots = (
            range(self.config.time_slots) 
            if constraint.time_slot == "ALL" 
            else [constraint.time_slot]
        )
        
        for ts in time_slots:
            count = sum(
                1 for res_id in filtered_resources
                if self.schedule[res_id][ts] == constraint.target_state
            )
            
            passes = self._check_operator(count, constraint.operator, constraint.value)
            
            if not passes:
                violations.append({
                    "time_slot": ts,
                    "expected": f"{constraint.operator} {constraint.value}",
                    "actual": count,
                    "attribute_filters": constraint.attribute_filters,
                    "filtered_resources": filtered_resources
                })
        
        filter_desc = " AND ".join(f"{k}={v}" for k, v in constraint.attribute_filters.items())
        
        if violations:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="compound_attribute_vertical_sum",
                constraint_name=name,
                status="FAIL",
                details=f"Failed at {len(violations)} time slot(s) for compound filter ({filter_desc})",
                violations=violations
            )
        else:
            return ConstraintValidationResult(
                constraint_index=index,
                constraint_type="compound_attribute_vertical_sum",
                constraint_name=name,
                status="PASS",
                details=f"All time slots meet {constraint.operator} {constraint.value} workers matching ({filter_desc}) ✓",
                violations=[]
            )
    
    @staticmethod
    def _check_operator(actual: int, operator: str, expected: int) -> bool:
        """
        Check if actual value meets operator comparison with expected.
        
        Args:
            actual: Actual count value
            operator: Comparison operator (>=, <=, ==)
            expected: Expected value
            
        Returns:
            True if comparison passes, False otherwise
        """
        if operator == ">=":
            return actual >= expected
        elif operator == "<=":
            return actual <= expected
        elif operator == "==":
            return actual == expected
        else:
            return False
