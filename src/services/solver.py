"""Solver service using Google OR-Tools CP-SAT.
Handles the core mathematical optimization logic.

Startup resilience: OR-Tools may not have wheels yet for very new Python
versions (e.g. 3.13). We attempt import and degrade gracefully so the API
still starts and health checks succeed instead of crashing the process.
"""

import time
from typing import Dict, List

try:  # Graceful import for environments where OR-Tools wheel is unavailable
    from ortools.sat.python import cp_model  # type: ignore
except Exception as e:  # pragma: no cover
    cp_model = None  # type: ignore
    print(f"[startup] WARNING: Failed to import OR-Tools cp_model: {e}")

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
    CompoundAttributeVerticalSumConstraint,
    PreviousMonthAssignment,
    MinConsecutiveConstraint,
    MaxConsecutiveConstraint,
    NightBlockGapConstraint,
    PostBlockRestConstraint,
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
        self.penalties = []  # Track penalty terms for soft constraints
        self.soft_violations = {}  # Track slack variables for reporting
        # Previous month integration (FN/ADM/RST/002)
        self.previous_month_offset = 0  # Number of previous month days included
        self.current_month_start = 0  # Index where current month starts
    
    def solve(self, request: SolveRequest) -> SolveResponse:
        """
        Solve the scheduling problem.
        
        Args:
            request: SolveRequest with config and constraints
            
        Returns:
            SolveResponse with status and schedule
        """
        start_time = time.time()

        if cp_model is None:
            # Degrade gracefully; caller can surface message to user.
            return SolveResponse(
                status="ERROR",
                message="OR-Tools not available in current runtime (wheel missing for Python version)",
                solve_time_ms=(time.time() - start_time) * 1000,
            )
        
        try:
            # Initialize model and variables (includes previous month offset)
            self._initialize_model(request.config, request.previous_month_assignments)
            
            # Apply previous month assignments as fixed constraints (if any)
            if request.previous_month_assignments:
                print(f"[Solver] Applying {len(request.previous_month_assignments)} previous month assignments")
                for pa in request.previous_month_assignments[:5]:  # Log first 5
                    print(f"  - {pa.resource}: offset_days={pa.offset_days}, state={pa.state}")
                self._apply_previous_month_assignments(request.previous_month_assignments)
            else:
                print("[Solver] No previous month assignments provided")
            
            # Apply all constraints
            for constraint in request.constraints:
                self._apply_constraint(constraint)
            
            # Add fairness optimization: minimize variance across all shift types
            self._add_fairness_objective()
            
            # Add optimization objective (fairness penalties + soft constraint penalties)
            if self.penalties:
                self.model.Minimize(sum(self.penalties))
            
            # Solve the model
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 300.0  # 5 minutes max
            
            status = solver.Solve(self.model)
            solve_time_ms = (time.time() - start_time) * 1000
            
            # Process results
            if status == cp_model.OPTIMAL:
                schedule = self._extract_schedule(solver)
                total_penalty = solver.ObjectiveValue() if self.penalties else 0
                return SolveResponse(
                    status="OPTIMAL",
                    schedule=schedule,
                    message=f"Found optimal solution (penalty: {total_penalty})" if self.penalties else "Found optimal solution",
                    solve_time_ms=solve_time_ms
                )
            elif status == cp_model.FEASIBLE:
                schedule = self._extract_schedule(solver)
                total_penalty = solver.ObjectiveValue() if self.penalties else 0
                return SolveResponse(
                    status="FEASIBLE",
                    schedule=schedule,
                    message=f"Found feasible solution (penalty: {total_penalty})" if self.penalties else "Found feasible solution (may not be optimal)",
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
    
    def _initialize_model(self, config, previous_month_assignments=None):
        """
        Initialize the CP-SAT model and create decision variables.
        
        Creates a matrix of IntVar variables where each represents:
        shifts[(resource, time_slot)] = state
        
        When previous_month_assignments are provided, the time range is extended
        to include those days (with negative indices) so that constraints like
        pattern_block can check transitions across the month boundary.
        
        The internal index mapping is:
        - Previous month days: 0 to (previous_month_offset - 1)
        - Current month days: previous_month_offset to (previous_month_offset + time_slots - 1)
        
        External constraints reference current month slots (0 to time_slots-1).
        Internal methods translate to actual internal indices.
        """
        self.model = cp_model.CpModel()
        self.resources = config.resources
        self.time_slots = config.time_slots  # Current month slots (user perspective)
        self.states = config.states
        self.resource_attributes = getattr(config, 'resource_attributes', {}) or {}
        self.shifts = {}
        self.penalties = []  # Reset penalties for new solve
        self.soft_violations = {}  # Reset violations tracking
        
        # Calculate previous month offset
        if previous_month_assignments:
            # Find max offset_days to determine how many days we need
            self.previous_month_offset = max(a.offset_days for a in previous_month_assignments)
        else:
            self.previous_month_offset = 0
        
        self.current_month_start = self.previous_month_offset
        total_time_slots = self.previous_month_offset + self.time_slots
        
        # Create decision variables for all time slots (including previous month)
        # Each variable represents the state of a resource at a specific time slot
        for resource in self.resources:
            for t in range(total_time_slots):
                var_name = f"shift_{resource}_{t}"
                # Variable can take any value from states (e.g., 0 to len(states)-1)
                self.shifts[(resource, t)] = self.model.NewIntVar(
                    min(self.states),
                    max(self.states),
                    var_name
                )
    
    def _to_internal_index(self, external_time_slot):
        """Convert external time slot (0-based for current month) to internal index."""
        return external_time_slot + self.previous_month_offset
    
    def _apply_previous_month_assignments(self, assignments: List[PreviousMonthAssignment]):
        """Apply previous month assignments as fixed (hard) point constraints.
        
        These assignments are locked and ensure continuity constraints work
        correctly across the month boundary.
        
        offset_days=1 → internal index (previous_month_offset - 1) = last previous day
        offset_days=2 → internal index (previous_month_offset - 2) = second-to-last
        
        Also tracks values in _previous_month_values for cross-month constraint logic.
        """
        # Initialize tracking dict for min_consecutive cross-month logic
        self._previous_month_values: dict[str, dict[int, int]] = {}
        
        for assignment in assignments:
            # Convert offset_days to internal index
            # offset_days=1 means last day of previous month
            internal_index = self.previous_month_offset - assignment.offset_days
            
            # Track the value for cross-month constraint logic
            if assignment.resource not in self._previous_month_values:
                self._previous_month_values[assignment.resource] = {}
            self._previous_month_values[assignment.resource][internal_index] = assignment.state
            
            # Apply as fixed constraint (hard)
            self.model.Add(
                self.shifts[(assignment.resource, internal_index)] == assignment.state
            )
    
    def _add_fairness_objective(self):
        """Add fairness optimization: minimize variance for each shift type.
        
        For each state (except OFF=0), minimize (max_count - min_count) across all resources.
        This ensures fair distribution of ALL shift types automatically.
        
        Penalty weight is lower than soft constraints so hard constraints take priority.
        
        Note: Only counts current month slots, not previous month overlap days.
        """
        FAIRNESS_WEIGHT = 10  # Lower weight than soft constraint penalties (100)
        
        # For each state > 0 (skip OFF state)
        for state in self.states:
            if state == 0:  # Skip OFF state
                continue
            
            # Create IntVar for count of this state per resource
            state_counts = {}
            for resource in self.resources:
                # Count how many times this resource is in this state (current month only)
                bool_vars = []
                for ext_t in range(self.time_slots):  # External time slots (current month)
                    int_t = self._to_internal_index(ext_t)  # Convert to internal
                    bv = self.model.NewBoolVar(f"fair_{resource}_{ext_t}_{state}")
                    self.model.Add(self.shifts[(resource, int_t)] == state).OnlyEnforceIf(bv)
                    self.model.Add(self.shifts[(resource, int_t)] != state).OnlyEnforceIf(bv.Not())
                    bool_vars.append(bv)
                
                # Total count for this resource and state
                count_var = self.model.NewIntVar(0, self.time_slots, f"count_{resource}_{state}")
                self.model.Add(count_var == sum(bool_vars))
                state_counts[resource] = count_var
            
            # Find max and min across all resources for this state
            all_counts = list(state_counts.values())
            max_count = self.model.NewIntVar(0, self.time_slots, f"max_state_{state}")
            min_count = self.model.NewIntVar(0, self.time_slots, f"min_state_{state}")
            
            self.model.AddMaxEquality(max_count, all_counts)
            self.model.AddMinEquality(min_count, all_counts)
            
            # Variance = max - min (we want to minimize this)
            variance = self.model.NewIntVar(0, self.time_slots, f"variance_state_{state}")
            self.model.Add(variance == max_count - min_count)
            
            # Add penalty for variance
            self.penalties.append(variance * FAIRNESS_WEIGHT)
    
    def _apply_constraint(self, constraint):
        """Apply a single constraint to the model.
        
        Hard constraints (is_required=True) are enforced strictly.
        Soft constraints (is_required=False) use slack variables with penalties.
        """
        is_required = getattr(constraint, 'is_required', True)
        
        if isinstance(constraint, PointConstraint):
            self._apply_point_constraint(constraint)
        elif isinstance(constraint, VerticalSumConstraint):
            if is_required:
                self._apply_vertical_sum_constraint(constraint)
            else:
                self._apply_vertical_sum_constraint_soft(constraint)
        elif isinstance(constraint, HorizontalSumConstraint):
            self._apply_horizontal_sum_constraint(constraint)
        elif isinstance(constraint, SlidingWindowConstraint):
            self._apply_sliding_window_constraint(constraint)
        elif isinstance(constraint, PatternBlockConstraint):
            self._apply_pattern_block_constraint(constraint)
        elif isinstance(constraint, AttributeVerticalSumConstraint):
            self._apply_attribute_vertical_sum_constraint(constraint)
        elif isinstance(constraint, ResourceStateCountConstraint):
            is_required = getattr(constraint, 'is_required', True)
            if is_required:
                self._apply_resource_state_count_constraint(constraint)
            else:
                self._apply_resource_state_count_constraint_soft(constraint)
        elif isinstance(constraint, CompoundAttributeVerticalSumConstraint):
            self._apply_compound_attribute_vertical_sum_constraint(constraint)
        elif isinstance(constraint, MinConsecutiveConstraint):
            self._apply_min_consecutive_constraint(constraint)
        elif isinstance(constraint, MaxConsecutiveConstraint):
            self._apply_max_consecutive_constraint(constraint)
        elif isinstance(constraint, NightBlockGapConstraint):
            self._apply_night_block_gap_constraint(constraint)
        elif isinstance(constraint, PostBlockRestConstraint):
            self._apply_post_block_rest_constraint(constraint)
    
    def _apply_point_constraint(self, constraint: PointConstraint):
        """
        Apply point constraint: Resource X at Time Y must be State Z.
        
        Example: Nurse "A" on Day 0 must be OFF (state=0)
        
        Note: time_slot is external (current month), converted to internal index.
        """
        resource = constraint.resource
        ext_time_slot = constraint.time_slot
        state = constraint.state
        
        # Convert to internal index
        int_time_slot = self._to_internal_index(ext_time_slot)
        
        # Add constraint: shifts[(resource, time_slot)] == state
        self.model.Add(self.shifts[(resource, int_time_slot)] == state)
    
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
        
        # Determine which external time slots to apply to
        if constraint.time_slot == "ALL":
            ext_time_slots_to_check = range(self.time_slots)
        else:
            ext_time_slots_to_check = [constraint.time_slot]
        
        # Apply constraint to each time slot
        for ext_t in ext_time_slots_to_check:
            int_t = self._to_internal_index(ext_t)
            # Create boolean variables for "is this resource at target state?"
            bool_vars = []
            for resource in self.resources:
                # Create a BoolVar that is 1 if shift == target_state
                bool_var = self.model.NewBoolVar(f"is_{resource}_{ext_t}_{target_state}")
                
                # Link BoolVar to IntVar:
                # bool_var is True <=> shifts[(resource, t)] == target_state
                self.model.Add(
                    self.shifts[(resource, int_t)] == target_state
                ).OnlyEnforceIf(bool_var)
                self.model.Add(
                    self.shifts[(resource, int_t)] != target_state
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
        
        Cross-month handling: Include previous month slots in window checks so that
        windows can span across month boundaries (e.g., Oct 31 + Nov 1-3 for max=3).
        
        Note: constraint.time_slots are external indices, converted to internal.
        """
        resource = constraint.resource
        ext_time_slots = constraint.time_slots
        target_state = constraint.target_state
        operator = constraint.operator
        value = constraint.value
        
        if operator == "<=":
            # Max consecutive: For each window of (value + 1) slots,
            # at least one must NOT be target_state
            window_size = value + 1
            
            # CROSS-MONTH: Build combined slot list including previous month
            # Previous month internal indices: 0 to (previous_month_offset - 1)
            # Current month external 0 -> internal previous_month_offset
            all_int_slots = []
            
            # Add previous month slots (these are already in internal indices)
            for prev_int_t in range(self.previous_month_offset):
                all_int_slots.append(prev_int_t)
            
            # Add current month slots (convert external to internal)
            sorted_ext_slots = sorted(ext_time_slots)
            for ext_t in sorted_ext_slots:
                int_t = self._to_internal_index(ext_t)
                all_int_slots.append(int_t)
            
            # Create windows that may span previous + current month
            # We need to constrain windows that include at least one current month slot
            for start_idx in range(len(all_int_slots) - window_size + 1):
                window = all_int_slots[start_idx:start_idx + window_size]
                
                # Only enforce constraint if at least one slot in window is from current month
                # (i.e., internal index >= previous_month_offset)
                has_current_month = any(int_t >= self.previous_month_offset for int_t in window)
                if not has_current_month:
                    continue  # Skip windows entirely in previous month (already fixed)
                
                # Create bool vars for each slot in window
                bool_vars = []
                for int_t in window:
                    bool_var = self.model.NewBoolVar(
                        f"h_{resource}_{int_t}_{target_state}_{start_idx}"
                    )
                    
                    # bool_var is True if shift == target_state
                    self.model.Add(
                        self.shifts[(resource, int_t)] == target_state
                    ).OnlyEnforceIf(bool_var)
                    self.model.Add(
                        self.shifts[(resource, int_t)] != target_state
                    ).OnlyEnforceIf(bool_var.Not())
                    
                    bool_vars.append(bool_var)
                
                # Sum must be <= value (equivalent to: not all window_size are target_state)
                self.model.Add(sum(bool_vars) <= value)
        
        elif operator == ">=":
            # Min consecutive: Ensure at least one sequence of (value) consecutive target_states
            # This is complex - require at least one window of size (value) to all be target_state
            
            # Convert external time slots to internal indices
            sorted_ext_slots = sorted(ext_time_slots)
            int_time_slots = [self._to_internal_index(ext_t) for ext_t in sorted_ext_slots]
            
            # Create a bool var for each possible window indicating "all target_state"
            window_bools = []
            
            for start_idx in range(len(int_time_slots) - value + 1):
                window = int_time_slots[start_idx:start_idx + value]
                
                # Bool var: True if all slots in window are target_state
                all_match = self.model.NewBoolVar(
                    f"window_match_{resource}_{start_idx}_{value}"
                )
                
                # Create bool vars for each slot
                slot_bools = []
                for int_t in window:
                    slot_bool = self.model.NewBoolVar(
                        f"h_ge_{resource}_{int_t}_{target_state}_{start_idx}"
                    )
                    
                    self.model.Add(
                        self.shifts[(resource, int_t)] == target_state
                    ).OnlyEnforceIf(slot_bool)
                    self.model.Add(
                        self.shifts[(resource, int_t)] != target_state
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
            
            # Convert external time slots to internal indices
            sorted_ext_slots = sorted(ext_time_slots)
            int_time_slots = [self._to_internal_index(ext_t) for ext_t in sorted_ext_slots]
            
            # Max consecutive (<=)
            window_size = value + 1
            for start_idx in range(len(int_time_slots) - window_size + 1):
                window = int_time_slots[start_idx:start_idx + window_size]
                
                bool_vars = []
                for int_t in window:
                    bool_var = self.model.NewBoolVar(
                        f"h_eq_{resource}_{int_t}_{target_state}_{start_idx}"
                    )
                    
                    self.model.Add(
                        self.shifts[(resource, int_t)] == target_state
                    ).OnlyEnforceIf(bool_var)
                    self.model.Add(
                        self.shifts[(resource, int_t)] != target_state
                    ).OnlyEnforceIf(bool_var.Not())
                    
                    bool_vars.append(bool_var)
                
                self.model.Add(sum(bool_vars) <= value)
            
            # Min consecutive (>=)
            window_bools = []
            for start_idx in range(len(int_time_slots) - value + 1):
                window = int_time_slots[start_idx:start_idx + value]
                
                all_match = self.model.NewBoolVar(
                    f"window_match_eq_{resource}_{start_idx}_{value}"
                )
                
                slot_bools = []
                for int_t in window:
                    slot_bool = self.model.NewBoolVar(
                        f"h_eq_ge_{resource}_{int_t}_{target_state}_{start_idx}"
                    )
                    
                    self.model.Add(
                        self.shifts[(resource, int_t)] == target_state
                    ).OnlyEnforceIf(slot_bool)
                    self.model.Add(
                        self.shifts[(resource, int_t)] != target_state
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
        
        IMPORTANT: When previous_month_assignments are provided, we check patterns
        that START in the previous month and END in the current month, BUT only
        if the rest window is ENTIRELY in the current month. We cannot modify
        previous month data (it's fixed), so we only enforce rest for days we control.
        """
        resource = constraint.resource
        work_days = constraint.work_days
        rest_days = constraint.rest_days
        target_state = constraint.target_state
        pattern_length = work_days + rest_days
        
        # Calculate starting range including previous month overlap
        # Start from negative indices if we have previous month data
        start_range = -self.previous_month_offset if self.previous_month_offset > 0 else 0
        end_range = self.time_slots - pattern_length + 1
        
        # Iterate through all possible pattern positions (external indices)
        # This now includes patterns starting in previous month (-previous_month_offset to -1)
        for ext_start_idx in range(start_range, end_range):
            work_window = [self._to_internal_index(ext_start_idx + i) for i in range(work_days)]
            rest_window = [self._to_internal_index(ext_start_idx + work_days + i) for i in range(rest_days)]
            
            # Skip if rest window extends beyond current month
            # (we can't enforce rest in next month)
            if any(int_t >= self.previous_month_offset + self.time_slots for int_t in rest_window):
                continue
            
            # Skip if rest window includes previous month days (we can't change them)
            # Only enforce rest for days in the current month (int_t >= previous_month_offset)
            if any(int_t < self.previous_month_offset for int_t in rest_window):
                continue
            
            # Create bool vars for work window (all must be target_state)
            work_bools = []
            for int_t in work_window:
                work_bool = self.model.NewBoolVar(
                    f"sw_work_{resource}_{int_t}_{ext_start_idx}"
                )
                
                self.model.Add(
                    self.shifts[(resource, int_t)] == target_state
                ).OnlyEnforceIf(work_bool)
                self.model.Add(
                    self.shifts[(resource, int_t)] != target_state
                ).OnlyEnforceIf(work_bool.Not())
                
                work_bools.append(work_bool)
            
            # Create bool var indicating "all work days worked"
            all_worked = self.model.NewBoolVar(
                f"sw_all_worked_{resource}_{ext_start_idx}"
            )
            
            # all_worked is True if all work_bools are True
            self.model.AddBoolAnd(work_bools).OnlyEnforceIf(all_worked)
            self.model.AddBoolOr([wb.Not() for wb in work_bools]).OnlyEnforceIf(all_worked.Not())
            
            # Create bool vars for rest window (all must be OFF = state 0)
            rest_bools = []
            for int_t in rest_window:
                rest_bool = self.model.NewBoolVar(
                    f"sw_rest_{resource}_{int_t}_{ext_start_idx}"
                )
                
                # rest_bool is True if shift == 0 (OFF)
                self.model.Add(
                    self.shifts[(resource, int_t)] == 0
                ).OnlyEnforceIf(rest_bool)
                self.model.Add(
                    self.shifts[(resource, int_t)] != 0
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
        
        IMPORTANT: When previous_month_assignments are provided, this constraint
        ALSO checks transitions from previous month days to current month days.
        This ensures patterns like Night→Day are blocked across month boundaries.
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
        
        # Calculate total internal time slots (includes previous month)
        total_internal_slots = self.previous_month_offset + self.time_slots
        
        # Apply to all resources (constraint.resources == "ALL")
        for resource in self.resources:
            # For each consecutive pair of internal time slots
            # This includes previous_month_day → first_current_day transitions!
            for int_t in range(total_internal_slots - 1):
                is_from = self.model.NewBoolVar(
                    f"pb_{resource}_{int_t}_is_{from_state}"
                )
                is_to = self.model.NewBoolVar(
                    f"pb_{resource}_{int_t+1}_is_{to_state}"
                )
                
                # is_from is True if shift[int_t] == from_state
                self.model.Add(
                    self.shifts[(resource, int_t)] == from_state
                ).OnlyEnforceIf(is_from)
                self.model.Add(
                    self.shifts[(resource, int_t)] != from_state
                ).OnlyEnforceIf(is_from.Not())
                
                # is_to is True if shift[int_t+1] == to_state
                self.model.Add(
                    self.shifts[(resource, int_t + 1)] == to_state
                ).OnlyEnforceIf(is_to)
                self.model.Add(
                    self.shifts[(resource, int_t + 1)] != to_state
                ).OnlyEnforceIf(is_to.Not())
                
                # Forbid the combination: NOT (is_from AND is_to)
                # Equivalent to: is_from => NOT is_to
                self.model.AddImplication(is_from, is_to.Not())

    def _apply_attribute_vertical_sum_constraint(self, constraint: AttributeVerticalSumConstraint):
        """Apply attribute filtered vertical sum constraint.
        Count resources with attribute in attribute_values AND state == target_state.
        
        Note: External time slots converted to internal indices.
        """
        target_state = constraint.target_state
        operator = constraint.operator
        value = constraint.value
        attribute = constraint.attribute
        attribute_values = set(constraint.attribute_values)

        if constraint.time_slot == "ALL":
            ext_slots = range(self.time_slots)
        else:
            ext_slots = [constraint.time_slot]

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

        for ext_t in ext_slots:
            int_t = self._to_internal_index(ext_t)
            bool_vars = []
            for resource in filtered_resources:
                b = self.model.NewBoolVar(f"attr_{attribute}_{resource}_{ext_t}_{target_state}")
                self.model.Add(self.shifts[(resource, int_t)] == target_state).OnlyEnforceIf(b)
                self.model.Add(self.shifts[(resource, int_t)] != target_state).OnlyEnforceIf(b.Not())
                bool_vars.append(b)
            sum_expr = sum(bool_vars)
            if operator == ">=":
                self.model.Add(sum_expr >= value)
            elif operator == "<=":
                self.model.Add(sum_expr <= value)
            elif operator == "==":
                self.model.Add(sum_expr == value)

    def _apply_resource_state_count_constraint(self, constraint: ResourceStateCountConstraint):
        """Apply total count of target_state occurrences for a resource across given time slots.
        
        Note: External time slots converted to internal indices.
        """
        resource = constraint.resource
        target_state = constraint.target_state
        operator = constraint.operator
        value = constraint.value
        ext_time_slots = constraint.time_slots

        bool_vars = []
        for ext_t in ext_time_slots:
            int_t = self._to_internal_index(ext_t)
            b = self.model.NewBoolVar(f"rsc_{resource}_{ext_t}_{target_state}")
            self.model.Add(self.shifts[(resource, int_t)] == target_state).OnlyEnforceIf(b)
            self.model.Add(self.shifts[(resource, int_t)] != target_state).OnlyEnforceIf(b.Not())
            bool_vars.append(b)
        sum_expr = sum(bool_vars)
        if operator == ">=":
            self.model.Add(sum_expr >= value)
        elif operator == "<=":
            self.model.Add(sum_expr <= value)
        elif operator == "==":
            self.model.Add(sum_expr == value)

    def _apply_resource_state_count_constraint_soft(self, constraint: ResourceStateCountConstraint):
        """Apply soft resource state count constraint with slack variables for violations.
        
        Instead of strict enforcement, allows violations with penalties.
        Used for fairness constraints like night distribution.
        
        Note: External time slots converted to internal indices.
        """
        resource = constraint.resource
        target_state = constraint.target_state
        operator = constraint.operator
        value = constraint.value
        ext_time_slots = constraint.time_slots
        
        PENALTY_WEIGHT = 100  # Penalty per unit of violation

        # Count occurrences of target_state
        bool_vars = []
        for ext_t in ext_time_slots:
            int_t = self._to_internal_index(ext_t)
            b = self.model.NewBoolVar(f"rsc_soft_{resource}_{ext_t}_{target_state}")
            self.model.Add(self.shifts[(resource, int_t)] == target_state).OnlyEnforceIf(b)
            self.model.Add(self.shifts[(resource, int_t)] != target_state).OnlyEnforceIf(b.Not())
            bool_vars.append(b)
        
        count = sum(bool_vars)
        
        # Create slack variable for violation
        max_violation = len(ext_time_slots)
        slack = self.model.NewIntVar(0, max_violation, f"slack_rsc_{resource}_{target_state}")
        
        if operator == ">=":
            # count + slack >= value (slack measures shortfall)
            self.model.Add(count + slack >= value)
        elif operator == "<=":
            # count <= value + slack (slack measures excess)
            self.model.Add(count <= value + slack)
        elif operator == "==":
            # Use two slack vars for under/over
            slack_under = self.model.NewIntVar(0, max_violation, f"slack_under_{resource}_{target_state}")
            slack_over = self.model.NewIntVar(0, max_violation, f"slack_over_{resource}_{target_state}")
            self.model.Add(count + slack_under - slack_over == value)
            # Total slack is under + over
            self.model.Add(slack == slack_under + slack_over)
        
        # Add penalty (weight per unit of violation)
        self.penalties.append(slack * PENALTY_WEIGHT)
        
        # Track for reporting
        self.soft_violations[f"rsc_{resource}_{target_state}"] = slack

    def _apply_vertical_sum_constraint_soft(self, constraint: VerticalSumConstraint):
        """Apply soft vertical sum constraint with slack variables for violations.
        
        Instead of strict enforcement, allows violations with penalties.
        
        Note: External time slots converted to internal indices.
        """
        target_state = constraint.target_state
        operator = constraint.operator
        value = constraint.value
        
        PENALTY_WEIGHT = 100  # Penalty per unit of violation
        
        # Determine which external time slots to apply to
        if constraint.time_slot == "ALL":
            ext_time_slots_to_check = range(self.time_slots)
        else:
            ext_time_slots_to_check = [constraint.time_slot]
        
        # Apply soft constraint to each time slot
        for ext_t in ext_time_slots_to_check:
            int_t = self._to_internal_index(ext_t)
            # Create boolean variables for "is this resource at target state?"
            bool_vars = []
            for resource in self.resources:
                bool_var = self.model.NewBoolVar(f"soft_vs_{resource}_{ext_t}_{target_state}")
                
                self.model.Add(
                    self.shifts[(resource, int_t)] == target_state
                ).OnlyEnforceIf(bool_var)
                self.model.Add(
                    self.shifts[(resource, int_t)] != target_state
                ).OnlyEnforceIf(bool_var.Not())
                
                bool_vars.append(bool_var)
            
            count = sum(bool_vars)
            
            # Create slack variable for violation at this time slot
            max_violation = len(self.resources)
            slack = self.model.NewIntVar(0, max_violation, f"slack_vs_{ext_t}_{target_state}")
            
            if operator == ">=":
                self.model.Add(count + slack >= value)
            elif operator == "<=":
                self.model.Add(count <= value + slack)
            elif operator == "==":
                slack_under = self.model.NewIntVar(0, max_violation, f"slack_vs_under_{t}_{target_state}")
                slack_over = self.model.NewIntVar(0, max_violation, f"slack_vs_over_{t}_{target_state}")
                self.model.Add(count + slack_under - slack_over == value)
                self.model.Add(slack == slack_under + slack_over)
            
            # Add penalty
            self.penalties.append(slack * PENALTY_WEIGHT)

    def _apply_compound_attribute_vertical_sum_constraint(
        self, 
        constraint: CompoundAttributeVerticalSumConstraint
    ):
        """Apply compound attribute filtered vertical sum constraint.
        
        Filters resources where ALL attribute conditions match (AND logic),
        then counts resources in target_state at each time slot.
        
        Example: At least 1 female IC (gender=F AND roles contains IC) on each day shift.
        
        Note: External time slots converted to internal indices.
        """
        target_state = constraint.target_state
        operator = constraint.operator
        value = constraint.value
        attribute_filters = constraint.attribute_filters

        if constraint.time_slot == "ALL":
            ext_slots = range(self.time_slots)
        else:
            ext_slots = [constraint.time_slot]

        # Pre-filter resources matching ALL attribute conditions (AND logic)
        filtered_resources = []
        for r in self.resources:
            if r not in self.resource_attributes:
                continue
            
            # Check all attribute filters (AND logic between attributes)
            all_match = True
            for attr_key, attr_values in attribute_filters.items():
                if attr_key not in self.resource_attributes[r]:
                    all_match = False
                    break
                
                raw_val = self.resource_attributes[r][attr_key]
                attr_values_set = set(attr_values)
                
                # Handle list attributes (e.g., roles: ['IC', 'RN'])
                # ANY-match within the attribute's value list
                if isinstance(raw_val, list):
                    match = any(str(v) in attr_values_set for v in raw_val)
                else:
                    match = str(raw_val) in attr_values_set
                
                if not match:
                    all_match = False
                    break
            
            if all_match:
                filtered_resources.append(r)

        # Apply vertical sum constraint to filtered resources
        for ext_t in ext_slots:
            int_t = self._to_internal_index(ext_t)
            bool_vars = []
            for resource in filtered_resources:
                b = self.model.NewBoolVar(
                    f"compound_{resource}_{ext_t}_{target_state}"
                )
                self.model.Add(
                    self.shifts[(resource, int_t)] == target_state
                ).OnlyEnforceIf(b)
                self.model.Add(
                    self.shifts[(resource, int_t)] != target_state
                ).OnlyEnforceIf(b.Not())
                bool_vars.append(b)
            
            sum_expr = sum(bool_vars)
            if operator == ">=":
                self.model.Add(sum_expr >= value)
            elif operator == "<=":
                self.model.Add(sum_expr <= value)
            elif operator == "==":
                self.model.Add(sum_expr == value)

    def _apply_min_consecutive_constraint(self, constraint: MinConsecutiveConstraint):
        """
        Apply minimum consecutive constraint: No isolated single occurrences.
        
        Example: Night shifts must occur in blocks of at least 2 (min_block=2).
        Pattern O E O is forbidden (single E), must be O E E O or similar.
        
        Implementation: For each occurrence of target_state, at least one neighbor
        must also be target_state (except at boundaries where we're more lenient).
        
        Cross-month handling: 
        1. Include previous month in neighbor checks for first days of current month.
        2. If previous month ends with an INCOMPLETE block (size < min_block),
           FORCE first days of current month to be target_state to complete the block.
        """
        resource = constraint.resource
        ext_time_slots = constraint.time_slots
        target_state = constraint.target_state
        min_block = constraint.min_block
        
        if min_block < 2:
            return  # No constraint needed
        
        # Sort time slots
        sorted_slots = sorted(ext_time_slots)
        n = len(sorted_slots)
        
        if n < min_block:
            return  # Not enough slots to form a block
        
        # Include previous month days in the slot range for cross-month constraints
        # Previous month internal indices: 0 to (previous_month_offset - 1)
        # Current month external 0 -> internal previous_month_offset
        all_internal_slots = []
        
        # Add previous month slots (if any)
        for prev_int_t in range(self.previous_month_offset):
            all_internal_slots.append((-self.previous_month_offset + prev_int_t, prev_int_t))  # (virtual_ext, int_t)
        
        # Add current month slots
        for ext_t in sorted_slots:
            int_t = self._to_internal_index(ext_t)
            all_internal_slots.append((ext_t, int_t))
        
        # Create boolean vars for ALL slots (including previous month)
        is_target = {}
        for (ext_t, int_t) in all_internal_slots:
            b = self.model.NewBoolVar(f"min_consec_{resource}_{ext_t}_is_{target_state}")
            self.model.Add(self.shifts[(resource, int_t)] == target_state).OnlyEnforceIf(b)
            self.model.Add(self.shifts[(resource, int_t)] != target_state).OnlyEnforceIf(b.Not())
            is_target[ext_t] = b
        
        # CROSS-MONTH: Check if previous month ends with an incomplete block
        # If so, FORCE continuation into current month to complete the block
        if self.previous_month_offset > 0:
            self._force_incomplete_block_continuation(resource, target_state, min_block, sorted_slots)
        
        # For min_block=2: if slot[i] is target, then either slot[i-1] or slot[i+1] must be target
        # This prevents isolated singles
        # Only enforce on CURRENT month slots (not previous month), but neighbors can be from previous month
        for i, ext_t in enumerate(sorted_slots):
            neighbors = []
            
            # Find index in all_internal_slots for this current month slot
            all_ext_slots = [s[0] for s in all_internal_slots]
            full_idx = all_ext_slots.index(ext_t)
            
            # Previous neighbor (could be from previous month)
            if full_idx > 0:
                prev_ext = all_ext_slots[full_idx - 1]
                neighbors.append(is_target[prev_ext])
            
            # Next neighbor (current month only)
            if full_idx < len(all_ext_slots) - 1:
                next_ext = all_ext_slots[full_idx + 1]
                neighbors.append(is_target[next_ext])
            
            if neighbors:
                # If this slot is target_state, at least one neighbor must also be target_state
                self.model.Add(sum(neighbors) >= 1).OnlyEnforceIf(is_target[ext_t])
            else:
                # No neighbors means this is an edge slot - for min_block=2, 
                # we should NOT allow a single isolated target_state at the very edge
                # unless it's being continued to/from the next/previous month
                # If no neighbors (edge case), prevent isolated single at end of month
                self.model.Add(self.shifts[(resource, self._to_internal_index(ext_t))] != target_state)
    
    def _force_incomplete_block_continuation(self, resource: str, target_state: int, min_block: int, current_month_slots: List[int]):
        """
        If previous month ends with an incomplete block (1 to min_block-1 consecutive target_states
        at the end), force first days of current month to complete the block.
        
        Example: If min_block=2 and previous month ends with a single E (Oct 31=E, Oct 30=O),
        then Nov 1 MUST be E to make it a valid block of 2.
        
        Example: If min_block=3 and previous month ends with E E (Oct 30-31=E, Oct 29=O),
        then Nov 1 MUST be E to make it a valid block of 3.
        
        IMPORTANT: Only counts days where we have explicit previous month data.
        Missing data is treated as "unknown" and breaks the count to be conservative.
        """
        if not hasattr(self, '_previous_month_values'):
            return  # No previous month data tracked
        
        prev_values = self._previous_month_values.get(resource, {})
        if not prev_values:
            return
        
        # Count trailing consecutive target_states at the end of previous month
        # Start from the last day of previous month (internal index = previous_month_offset - 1)
        # and count backwards until we find a non-target OR missing data
        trailing_count = 0
        for int_t in range(self.previous_month_offset - 1, -1, -1):
            value = prev_values.get(int_t)
            if value is None:
                # Missing data - we can't be sure, so stop counting
                # This is conservative: we only force continuation if we KNOW
                # there's an incomplete block
                break
            if value == target_state:
                trailing_count += 1
            else:
                break  # Found a non-target state, stop counting
        
        # If there's no trailing block at all, nothing to continue
        if trailing_count == 0:
            return
        
        # If trailing block is already complete (>= min_block), no forcing needed
        if trailing_count >= min_block:
            return
        
        # Incomplete block found! Need to force (min_block - trailing_count) days in current month
        days_needed = min_block - trailing_count
        
        # Force first N days of current month to be target_state
        for i in range(min(days_needed, len(current_month_slots))):
            ext_t = current_month_slots[i]
            int_t = self._to_internal_index(ext_t)
            self.model.Add(self.shifts[(resource, int_t)] == target_state)

    def _apply_max_consecutive_constraint(self, constraint: MaxConsecutiveConstraint):
        """
        Apply max consecutive constraint: Maximum consecutive occurrences of a state.
        
        Example: Day shifts must not exceed 3 consecutive (prevents 7 7 7 7 patterns).
        For max_block=3: 7 7 7 O is OK, 7 7 7 7 is NOT allowed.
        
        Special case: If target_state=-1, applies to ALL non-OFF work states combined.
        Example: target_state=-1, max_block=3 prevents 7 7 E E E (5 consecutive work).
        
        Implementation: For every window of (max_block + 1) consecutive time slots,
        at least one must NOT be the target state (or any work state if target_state=-1).
        
        Note: External time slots converted to internal indices.
        Respects cross-month boundaries via previous_month_offset.
        """
        resource = constraint.resource
        ext_time_slots = constraint.time_slots
        target_state = constraint.target_state
        max_block = constraint.max_block
        
        # Sort time slots
        sorted_slots = sorted(ext_time_slots)
        n = len(sorted_slots)
        
        if n < max_block + 1:
            return  # Not enough slots to violate the max
        
        # Include previous month days in the slot range for cross-month constraints
        all_internal_slots = []
        
        # Add previous month slots (if any)
        for prev_int_t in range(self.previous_month_offset):
            all_internal_slots.append((-self.previous_month_offset + prev_int_t, prev_int_t))  # (virtual_ext, int_t)
        
        # Add current month slots
        for ext_t in sorted_slots:
            int_t = self._to_internal_index(ext_t)
            all_internal_slots.append((ext_t, int_t))
        
        # Create boolean vars for ALL slots (including previous month)
        is_target = {}
        for (ext_t, int_t) in all_internal_slots:
            b = self.model.NewBoolVar(f"max_consec_{resource}_{ext_t}_is_{target_state}")
            
            if target_state == -1:
                # Special case: Check if ANY work state (non-zero)
                # b is True if shift is ANY work state (state > 0)
                self.model.Add(self.shifts[(resource, int_t)] > 0).OnlyEnforceIf(b)
                self.model.Add(self.shifts[(resource, int_t)] == 0).OnlyEnforceIf(b.Not())
            else:
                # Normal case: Check specific state
                self.model.Add(self.shifts[(resource, int_t)] == target_state).OnlyEnforceIf(b)
                self.model.Add(self.shifts[(resource, int_t)] != target_state).OnlyEnforceIf(b.Not())
            
            is_target[ext_t] = b
        
        # For each window of (max_block + 1) consecutive slots, at least one must NOT be target
        all_ext_slots = [s[0] for s in all_internal_slots]
        
        for i in range(len(all_ext_slots) - max_block):
            window_slots = all_ext_slots[i:i + max_block + 1]
            window_bools = [is_target[ext_t] for ext_t in window_slots]
            
            # At least one of these (max_block + 1) slots must NOT be target_state
            # Equivalently: sum of target bools must be <= max_block
            self.model.Add(sum(window_bools) <= max_block)

    def _apply_night_block_gap_constraint(self, constraint: NightBlockGapConstraint):
        """
        Apply night block gap constraint: Minimum gap between night shift blocks.
        
        Example: After a night block ends, at least 7 days before starting another.
        Pattern: E E O O O O O O O E E is OK (7-day gap), E E O O E E is NOT (2-day gap).
        
        Cross-month handling: If previous month data exists, we include it to detect
        blocks that ended in previous month and enforce gap into current month.
        """
        resource = constraint.resource
        ext_time_slots = constraint.time_slots
        target_state = constraint.target_state
        min_gap_days = constraint.min_gap_days
        
        # Build combined slot list including previous month
        all_slots = []  # List of (external_slot, internal_index)
        
        # Add previous month slots
        for prev_int_t in range(self.previous_month_offset):
            virtual_ext = prev_int_t - self.previous_month_offset  # Negative values for prev month
            all_slots.append((virtual_ext, prev_int_t))
        
        # Add current month slots
        sorted_ext_slots = sorted(ext_time_slots)
        for ext_t in sorted_ext_slots:
            int_t = self._to_internal_index(ext_t)
            all_slots.append((ext_t, int_t))
        
        n = len(all_slots)
        if n < 2:
            return
        
        # Create boolean vars for all slots
        is_target = {}
        for (ext_t, int_t) in all_slots:
            b = self.model.NewBoolVar(f"gap_{resource}_{ext_t}_is_{target_state}")
            self.model.Add(self.shifts[(resource, int_t)] == target_state).OnlyEnforceIf(b)
            self.model.Add(self.shifts[(resource, int_t)] != target_state).OnlyEnforceIf(b.Not())
            is_target[ext_t] = b
        
        all_ext_slots = [s[0] for s in all_slots]
        
        # For each potential block-end at i and block-start at j with gap < min_gap_days:
        # Only enforce constraints where j is in current month (ext >= 0)
        for i in range(n - 1):
            end_ext = all_ext_slots[i]
            next_ext = all_ext_slots[i + 1]
            
            # Check slots within min_gap_days after end_ext
            for j in range(i + 2, n):
                start_ext = all_ext_slots[j]
                prev_ext = all_ext_slots[j - 1]
                
                # Only constrain if start is in current month
                if start_ext < 0:
                    continue
                
                gap = start_ext - end_ext - 1
                
                if gap >= min_gap_days:
                    break
                
                # Block ends at end_ext and starts at start_ext with insufficient gap
                self.model.AddBoolOr([
                    is_target[end_ext].Not(),
                    is_target[next_ext],
                    is_target[prev_ext],
                    is_target[start_ext].Not()
                ])

    def _apply_post_block_rest_constraint(self, constraint: PostBlockRestConstraint):
        """
        Apply post-block rest constraint: After any block of target_state ends, 
        require rest_days of OFF before any other shift.
        
        Example: After night block ends (E E -> O), need 2 days OFF before working.
        Pattern: E E O O 7 is OK, E E O 7 is NOT (only 1 rest day).
        
        Cross-month handling: Detects block ends in previous month and enforces
        rest days that extend into current month.
        """
        resource = constraint.resource
        target_state = constraint.target_state
        rest_days = constraint.rest_days
        
        total_internal_slots = self.previous_month_offset + self.time_slots
        current_month_start = self.previous_month_offset
        
        # Create boolean vars for each internal slot being in target state
        is_target = {}
        for int_t in range(total_internal_slots):
            b = self.model.NewBoolVar(f"post_rest_{resource}_{int_t}_is_{target_state}")
            self.model.Add(self.shifts[(resource, int_t)] == target_state).OnlyEnforceIf(b)
            self.model.Add(self.shifts[(resource, int_t)] != target_state).OnlyEnforceIf(b.Not())
            is_target[int_t] = b
        
        # For each slot t, if t is target_state AND t+1 is NOT target_state (block end),
        # then slots t+1 through t+rest_days must all be OFF (state 0)
        # But only enforce on slots that are in the CURRENT month (int_t >= current_month_start)
        for int_t in range(total_internal_slots - 1):
            # Block end detection: is_target[t] AND NOT is_target[t+1]
            block_end = self.model.NewBoolVar(f"block_end_{resource}_{int_t}")
            self.model.AddBoolAnd([is_target[int_t], is_target[int_t + 1].Not()]).OnlyEnforceIf(block_end)
            self.model.AddBoolOr([is_target[int_t].Not(), is_target[int_t + 1]]).OnlyEnforceIf(block_end.Not())
            
            # If block ends at t, then slots t+1 to t+rest_days must be OFF (state 0)
            # Only enforce on current month slots
            for d in range(1, rest_days + 1):
                rest_slot = int_t + d
                if rest_slot < total_internal_slots and rest_slot >= current_month_start:
                    self.model.Add(self.shifts[(resource, rest_slot)] == 0).OnlyEnforceIf(block_end)
    
    def _extract_schedule(self, solver: cp_model.CpSolver) -> Dict[str, List[int]]:
        """
        Extract the schedule from the solved model.
        
        Returns current month schedule only, excluding previous month overlap days.
        The returned indices are 0-based for the current month.
        
        Returns:
            Dictionary mapping resource_id to list of states per time slot
        """
        schedule = {}
        
        for resource in self.resources:
            schedule[resource] = []
            # Only extract current month time slots
            for ext_t in range(self.time_slots):
                int_t = self._to_internal_index(ext_t)
                state_value = solver.Value(self.shifts[(resource, int_t)])
                schedule[resource].append(state_value)
        
        return schedule


# Singleton instance
solver_service = SolverService()
