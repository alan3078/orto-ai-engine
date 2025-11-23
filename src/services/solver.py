"""
Solver service using Google OR-Tools CP-SAT.
Handles the core mathematical optimization logic.
"""

import time
from typing import Dict, List, Tuple
from ortools.sat.python import cp_model

from src.core.schemas import (
    SolveRequest,
    SolveResponse,
    PointConstraint,
    VerticalSumConstraint,
    HorizontalSumConstraint,
    SlidingWindowConstraint,
    PatternBlockConstraint,
    AttributeVerticalSumConstraint,
    ResourceStateCountConstraint,
)


class SolverService:
    """Service for solving scheduling problems using OR-Tools."""
    
    def __init__(self):
        self.model = None
        self.shifts = {}
        self.resources = []
        self.time_slots = 0
        self.states = []
        self.resource_attributes = {}
    
    def solve(self, request: SolveRequest) -> SolveResponse:
        """
        Solve the scheduling problem.
        
        Args:
            request: SolveRequest with config and constraints
            
        Returns:
            SolveResponse with status and schedule
        """
        start_time = time.time()
        
        try:
            # Initialize model and variables
            self._initialize_model(request.config)
            
            # Apply all constraints
            for constraint in request.constraints:
                self._apply_constraint(constraint)
            
            # Solve the model
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 300.0  # 5 minutes max
            
            status = solver.Solve(self.model)
            solve_time_ms = (time.time() - start_time) * 1000
            
            # Process results
            if status == cp_model.OPTIMAL:
                schedule = self._extract_schedule(solver)
                return SolveResponse(
                    status="OPTIMAL",
                    schedule=schedule,
                    message="Found optimal solution",
                    solve_time_ms=solve_time_ms
                )
            elif status == cp_model.FEASIBLE:
                schedule = self._extract_schedule(solver)
                return SolveResponse(
                    status="FEASIBLE",
                    schedule=schedule,
                    message="Found feasible solution (may not be optimal)",
                    solve_time_ms=solve_time_ms
                )
            elif status == cp_model.INFEASIBLE:
                return SolveResponse(
                    status="INFEASIBLE",
                    message="No solution exists that satisfies all constraints",
                    solve_time_ms=solve_time_ms
                )
            else:
                return SolveResponse(
                    status="ERROR",
                    message=f"Solver returned status: {solver.StatusName(status)}",
                    solve_time_ms=solve_time_ms
                )
                
        except Exception as e:
            solve_time_ms = (time.time() - start_time) * 1000
            return SolveResponse(
                status="ERROR",
                message=f"Error during solving: {str(e)}",
                solve_time_ms=solve_time_ms
            )
    
    def _initialize_model(self, config):
        """
        Initialize the CP-SAT model and create decision variables.
        
        Creates a matrix of IntVar variables where each represents:
        shifts[(resource, time_slot)] = state
        """
        self.model = cp_model.CpModel()
        self.resources = config.resources
        self.time_slots = config.time_slots
        self.states = config.states
        self.resource_attributes = getattr(config, 'resource_attributes', {}) or {}
        self.shifts = {}
        
        # Create decision variables
        # Each variable represents the state of a resource at a specific time slot
        for resource in self.resources:
            for t in range(self.time_slots):
                var_name = f"shift_{resource}_{t}"
                # Variable can take any value from states (e.g., 0 to len(states)-1)
                self.shifts[(resource, t)] = self.model.NewIntVar(
                    min(self.states),
                    max(self.states),
                    var_name
                )
    
    def _apply_constraint(self, constraint):
        """Apply a single constraint to the model."""
        if isinstance(constraint, PointConstraint):
            self._apply_point_constraint(constraint)
        elif isinstance(constraint, VerticalSumConstraint):
            self._apply_vertical_sum_constraint(constraint)
        elif isinstance(constraint, HorizontalSumConstraint):
            self._apply_horizontal_sum_constraint(constraint)
        elif isinstance(constraint, SlidingWindowConstraint):
            self._apply_sliding_window_constraint(constraint)
        elif isinstance(constraint, PatternBlockConstraint):
            self._apply_pattern_block_constraint(constraint)
        elif isinstance(constraint, AttributeVerticalSumConstraint):
            self._apply_attribute_vertical_sum_constraint(constraint)
        elif isinstance(constraint, ResourceStateCountConstraint):
            self._apply_resource_state_count_constraint(constraint)
    
    def _apply_point_constraint(self, constraint: PointConstraint):
        """
        Apply point constraint: Resource X at Time Y must be State Z.
        
        Example: Nurse "A" on Day 0 must be OFF (state=0)
        """
        resource = constraint.resource
        time_slot = constraint.time_slot
        state = constraint.state
        
        # Add constraint: shifts[(resource, time_slot)] == state
        self.model.Add(self.shifts[(resource, time_slot)] == state)
    
    def _apply_vertical_sum_constraint(self, constraint: VerticalSumConstraint):
        """
        Apply vertical sum constraint: Count of specific state at time slot(s).
        
        Example: At least 2 people must be working (state=1) on every day
        
        Since shifts are IntVars, we need to create BoolVars to count them:
        is_target_state = 1 if shift == target_state else 0
        """
        target_state = constraint.target_state
        operator = constraint.operator
        value = constraint.value
        
        # Determine which time slots to apply to
        if constraint.time_slot == "ALL":
            time_slots_to_check = range(self.time_slots)
        else:
            time_slots_to_check = [constraint.time_slot]
        
        # Apply constraint to each time slot
        for t in time_slots_to_check:
            # Create boolean variables for "is this resource at target state?"
            bool_vars = []
            for resource in self.resources:
                # Create a BoolVar that is 1 if shift == target_state
                bool_var = self.model.NewBoolVar(f"is_{resource}_{t}_{target_state}")
                
                # Link BoolVar to IntVar:
                # bool_var is True <=> shifts[(resource, t)] == target_state
                self.model.Add(
                    self.shifts[(resource, t)] == target_state
                ).OnlyEnforceIf(bool_var)
                self.model.Add(
                    self.shifts[(resource, t)] != target_state
                ).OnlyEnforceIf(bool_var.Not())
                
                bool_vars.append(bool_var)
            
            # Apply the sum constraint
            sum_expr = sum(bool_vars)
            
            if operator == ">=":
                self.model.Add(sum_expr >= value)
            elif operator == "<=":
                self.model.Add(sum_expr <= value)
            elif operator == "==":
                self.model.Add(sum_expr == value)
    
    def _apply_horizontal_sum_constraint(self, constraint: HorizontalSumConstraint):
        """
        Apply horizontal sum constraint: Limit consecutive occurrences of a state.
        
        Example: Bob can work maximum 3 consecutive days
        
        Strategy: Use sliding window to check all consecutive sequences.
        For each window of size (value + 1), ensure not all are target_state.
        """
        resource = constraint.resource
        time_slots = constraint.time_slots
        target_state = constraint.target_state
        operator = constraint.operator
        value = constraint.value
        
        if operator == "<=":
            # Max consecutive: For each window of (value + 1) slots,
            # at least one must NOT be target_state
            window_size = value + 1
            
            for start_idx in range(len(time_slots) - window_size + 1):
                window = time_slots[start_idx:start_idx + window_size]
                
                # Create bool vars for each slot in window
                bool_vars = []
                for t in window:
                    bool_var = self.model.NewBoolVar(
                        f"h_{resource}_{t}_{target_state}_{start_idx}"
                    )
                    
                    # bool_var is True if shift == target_state
                    self.model.Add(
                        self.shifts[(resource, t)] == target_state
                    ).OnlyEnforceIf(bool_var)
                    self.model.Add(
                        self.shifts[(resource, t)] != target_state
                    ).OnlyEnforceIf(bool_var.Not())
                    
                    bool_vars.append(bool_var)
                
                # Sum must be <= value (equivalent to: not all window_size are target_state)
                self.model.Add(sum(bool_vars) <= value)
        
        elif operator == ">=":
            # Min consecutive: Ensure at least one sequence of (value) consecutive target_states
            # This is complex - require at least one window of size (value) to all be target_state
            
            # Create a bool var for each possible window indicating "all target_state"
            window_bools = []
            
            for start_idx in range(len(time_slots) - value + 1):
                window = time_slots[start_idx:start_idx + value]
                
                # Bool var: True if all slots in window are target_state
                all_match = self.model.NewBoolVar(
                    f"window_match_{resource}_{start_idx}_{value}"
                )
                
                # Create bool vars for each slot
                slot_bools = []
                for t in window:
                    slot_bool = self.model.NewBoolVar(
                        f"h_ge_{resource}_{t}_{target_state}_{start_idx}"
                    )
                    
                    self.model.Add(
                        self.shifts[(resource, t)] == target_state
                    ).OnlyEnforceIf(slot_bool)
                    self.model.Add(
                        self.shifts[(resource, t)] != target_state
                    ).OnlyEnforceIf(slot_bool.Not())
                    
                    slot_bools.append(slot_bool)
                
                # all_match is True if all slot_bools are True
                self.model.AddBoolAnd(slot_bools).OnlyEnforceIf(all_match)
                self.model.AddBoolOr([slot_bool.Not() for slot_bool in slot_bools]).OnlyEnforceIf(all_match.Not())
                
                window_bools.append(all_match)
            
            # At least one window must match
            self.model.AddBoolOr(window_bools)
        
        elif operator == "==":
            # Exactly N consecutive: Complex, implement as >= AND <=
            # For now, enforce <= and >= separately
            
            # Max consecutive (<=)
            window_size = value + 1
            for start_idx in range(len(time_slots) - window_size + 1):
                window = time_slots[start_idx:start_idx + window_size]
                
                bool_vars = []
                for t in window:
                    bool_var = self.model.NewBoolVar(
                        f"h_eq_{resource}_{t}_{target_state}_{start_idx}"
                    )
                    
                    self.model.Add(
                        self.shifts[(resource, t)] == target_state
                    ).OnlyEnforceIf(bool_var)
                    self.model.Add(
                        self.shifts[(resource, t)] != target_state
                    ).OnlyEnforceIf(bool_var.Not())
                    
                    bool_vars.append(bool_var)
                
                self.model.Add(sum(bool_vars) <= value)
            
            # Min consecutive (>=)
            window_bools = []
            for start_idx in range(len(time_slots) - value + 1):
                window = time_slots[start_idx:start_idx + value]
                
                all_match = self.model.NewBoolVar(
                    f"window_match_eq_{resource}_{start_idx}_{value}"
                )
                
                slot_bools = []
                for t in window:
                    slot_bool = self.model.NewBoolVar(
                        f"h_eq_ge_{resource}_{t}_{target_state}_{start_idx}"
                    )
                    
                    self.model.Add(
                        self.shifts[(resource, t)] == target_state
                    ).OnlyEnforceIf(slot_bool)
                    self.model.Add(
                        self.shifts[(resource, t)] != target_state
                    ).OnlyEnforceIf(slot_bool.Not())
                    
                    slot_bools.append(slot_bool)
                
                self.model.AddBoolAnd(slot_bools).OnlyEnforceIf(all_match)
                self.model.AddBoolOr([sb.Not() for sb in slot_bools]).OnlyEnforceIf(all_match.Not())
                
                window_bools.append(all_match)
            
            self.model.AddBoolOr(window_bools)
    
    def _apply_sliding_window_constraint(self, constraint: SlidingWindowConstraint):
        """
        Apply sliding window constraint: Work-rest pattern enforcement.
        
        Example: After working 5 consecutive days, must have 2 days rest
        
        Strategy: For each window of (work_days + rest_days) slots,
        if first work_days are all target_state, then next rest_days must be OFF (state 0).
        """
        resource = constraint.resource
        work_days = constraint.work_days
        rest_days = constraint.rest_days
        target_state = constraint.target_state
        pattern_length = work_days + rest_days
        
        # Iterate through all possible pattern positions
        for start_idx in range(self.time_slots - pattern_length + 1):
            work_window = list(range(start_idx, start_idx + work_days))
            rest_window = list(range(start_idx + work_days, start_idx + pattern_length))
            
            # Create bool vars for work window (all must be target_state)
            work_bools = []
            for t in work_window:
                work_bool = self.model.NewBoolVar(
                    f"sw_work_{resource}_{t}_{start_idx}"
                )
                
                self.model.Add(
                    self.shifts[(resource, t)] == target_state
                ).OnlyEnforceIf(work_bool)
                self.model.Add(
                    self.shifts[(resource, t)] != target_state
                ).OnlyEnforceIf(work_bool.Not())
                
                work_bools.append(work_bool)
            
            # Create bool var indicating "all work days worked"
            all_worked = self.model.NewBoolVar(
                f"sw_all_worked_{resource}_{start_idx}"
            )
            
            # all_worked is True if all work_bools are True
            self.model.AddBoolAnd(work_bools).OnlyEnforceIf(all_worked)
            self.model.AddBoolOr([wb.Not() for wb in work_bools]).OnlyEnforceIf(all_worked.Not())
            
            # Create bool vars for rest window (all must be OFF = state 0)
            rest_bools = []
            for t in rest_window:
                rest_bool = self.model.NewBoolVar(
                    f"sw_rest_{resource}_{t}_{start_idx}"
                )
                
                # rest_bool is True if shift == 0 (OFF)
                self.model.Add(
                    self.shifts[(resource, t)] == 0
                ).OnlyEnforceIf(rest_bool)
                self.model.Add(
                    self.shifts[(resource, t)] != 0
                ).OnlyEnforceIf(rest_bool.Not())
                
                rest_bools.append(rest_bool)
            
            # Implication: if all_worked, then all rest_bools must be True
            # Equivalent to: all_worked => AND(rest_bools)
            for rest_bool in rest_bools:
                self.model.AddImplication(all_worked, rest_bool)
    
    def _apply_pattern_block_constraint(self, constraint: PatternBlockConstraint):
        """
        Apply pattern block constraint: Prevent specific state transitions.
        
        Example: Block Night→Day transition (state 2 followed by state 1)
        
        Strategy: For each consecutive pair of time slots (t, t+1),
        ensure NOT (shift[t] == from_state AND shift[t+1] == to_state)
        """
        # Default state mapping if not provided
        state_mapping = constraint.state_mapping or {
            'NIGHT': 2,
            'DAY': 1,
            'OFF': 0,
        }
        
        # Extract from/to states from pattern
        from_pattern = constraint.pattern[0]
        to_pattern = constraint.pattern[1]
        
        from_state = state_mapping.get(from_pattern)
        to_state = state_mapping.get(to_pattern)
        
        if from_state is None or to_state is None:
            # Skip if pattern names not in mapping
            print(f"[Solver] Warning: Pattern {constraint.pattern} not in state_mapping, skipping")
            return
        
        # Apply to all resources (constraint.resources == "ALL")
        for resource in self.resources:
            # For each consecutive pair of time slots
            for t in range(self.time_slots - 1):
                # Create bool vars for the forbidden transition
                is_from = self.model.NewBoolVar(
                    f"pb_{resource}_{t}_is_{from_state}"
                )
                is_to = self.model.NewBoolVar(
                    f"pb_{resource}_{t+1}_is_{to_state}"
                )
                
                # is_from is True if shift[t] == from_state
                self.model.Add(
                    self.shifts[(resource, t)] == from_state
                ).OnlyEnforceIf(is_from)
                self.model.Add(
                    self.shifts[(resource, t)] != from_state
                ).OnlyEnforceIf(is_from.Not())
                
                # is_to is True if shift[t+1] == to_state
                self.model.Add(
                    self.shifts[(resource, t + 1)] == to_state
                ).OnlyEnforceIf(is_to)
                self.model.Add(
                    self.shifts[(resource, t + 1)] != to_state
                ).OnlyEnforceIf(is_to.Not())
                
                # Forbid the combination: NOT (is_from AND is_to)
                # Equivalent to: is_from => NOT is_to
                self.model.AddImplication(is_from, is_to.Not())

    def _apply_attribute_vertical_sum_constraint(self, constraint: AttributeVerticalSumConstraint):
        """Apply attribute filtered vertical sum constraint.
        Count resources with attribute in attribute_values AND state == target_state.
        """
        target_state = constraint.target_state
        operator = constraint.operator
        value = constraint.value
        attribute = constraint.attribute
        attribute_values = set(constraint.attribute_values)

        if constraint.time_slot == "ALL":
            slots = range(self.time_slots)
        else:
            slots = [constraint.time_slot]

        # Pre-filter resource list
        filtered_resources = []
        for r in self.resources:
            if r not in self.resource_attributes:
                continue
            if attribute not in self.resource_attributes[r]:
                continue
            raw_val = self.resource_attributes[r][attribute]
            include = False
            if isinstance(raw_val, list):
                include = any(str(v) in attribute_values for v in raw_val)
            else:
                include = str(raw_val) in attribute_values
            if include:
                filtered_resources.append(r)

        for t in slots:
            bool_vars = []
            for resource in filtered_resources:
                b = self.model.NewBoolVar(f"attr_{attribute}_{resource}_{t}_{target_state}")
                self.model.Add(self.shifts[(resource, t)] == target_state).OnlyEnforceIf(b)
                self.model.Add(self.shifts[(resource, t)] != target_state).OnlyEnforceIf(b.Not())
                bool_vars.append(b)
            sum_expr = sum(bool_vars)
            if operator == ">=":
                self.model.Add(sum_expr >= value)
            elif operator == "<=":
                self.model.Add(sum_expr <= value)
            elif operator == "==":
                self.model.Add(sum_expr == value)

    def _apply_resource_state_count_constraint(self, constraint: ResourceStateCountConstraint):
        """Apply total count of target_state occurrences for a resource across given time slots."""
        resource = constraint.resource
        target_state = constraint.target_state
        operator = constraint.operator
        value = constraint.value
        time_slots = constraint.time_slots

        bool_vars = []
        for t in time_slots:
            b = self.model.NewBoolVar(f"rsc_{resource}_{t}_{target_state}")
            self.model.Add(self.shifts[(resource, t)] == target_state).OnlyEnforceIf(b)
            self.model.Add(self.shifts[(resource, t)] != target_state).OnlyEnforceIf(b.Not())
            bool_vars.append(b)
        sum_expr = sum(bool_vars)
        if operator == ">=":
            self.model.Add(sum_expr >= value)
        elif operator == "<=":
            self.model.Add(sum_expr <= value)
        elif operator == "==":
            self.model.Add(sum_expr == value)
    
    def _extract_schedule(self, solver: cp_model.CpSolver) -> Dict[str, List[int]]:
        """
        Extract the schedule from the solved model.
        
        Returns:
            Dictionary mapping resource_id to list of states per time slot
        """
        schedule = {}
        
        for resource in self.resources:
            schedule[resource] = []
            for t in range(self.time_slots):
                state_value = solver.Value(self.shifts[(resource, t)])
                schedule[resource].append(state_value)
        
        return schedule


# Singleton instance
solver_service = SolverService()
