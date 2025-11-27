"""
API router for the solver endpoints.
"""

import time
from fastapi import APIRouter, HTTPException
from src.core.schemas import (
    SolveRequest, SolveResponse, ValidateRequest, ValidateResponse,
    ScheduleSummary, VerticalSummary, HorizontalSummary
)
from src.services.solver import solver_service
from src.services.validator import ConstraintValidator

router = APIRouter(prefix="/api/v1", tags=["solver"])


@router.post("/solve", response_model=SolveResponse)
async def solve(request: SolveRequest) -> SolveResponse:
    """
    Solve a scheduling problem with the given configuration and constraints.
    
    Args:
        request: SolveRequest containing config and constraints
        
    Returns:
        SolveResponse with status, schedule, and metadata
        
    Example:
        ```json
        {
          "config": {
            "resources": ["A", "B", "C"],
            "time_slots": 7,
            "states": [0, 1]
          },
          "constraints": [
            {
              "type": "point",
              "resource": "A",
              "time_slot": 0,
              "state": 0
            },
            {
              "type": "vertical_sum",
              "time_slot": "ALL",
              "target_state": 1,
              "operator": ">=",
              "value": 2
            }
          ]
        }
        ```
    """
    try:
        response = solver_service.solve(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during solving: {str(e)}"
        )


@router.post("/validate", response_model=ValidateResponse)
async def validate(request: ValidateRequest) -> ValidateResponse:
    """
    Validate an existing schedule against a set of constraints.
    
    Part of FN/BE/ENG/002 - Roster Validator (Audit Mode)
    
    Args:
        request: ValidateRequest containing config, schedule, and constraints
        
    Returns:
        ValidateResponse with per-constraint validation results
        
    Example:
        ```json
        {
          "config": {
            "resources": ["A", "B", "C"],
            "time_slots": 7,
            "states": [0, 1]
          },
          "schedule": {
            "A": [0, 1, 1, 0, 0, 1, 0],
            "B": [1, 1, 0, 0, 1, 1, 0],
            "C": [0, 0, 1, 1, 1, 0, 0]
          },
          "constraints": [
            {
              "type": "point",
              "resource": "A",
              "time_slot": 0,
              "state": 0
            },
            {
              "type": "vertical_sum",
              "time_slot": "ALL",
              "target_state": 1,
              "operator": ">=",
              "value": 2
            }
          ]
        }
        ```
    """
    try:
        start_time = time.time()
        
        validator = ConstraintValidator(request.config, request.schedule)
        
        results = []
        constraint_names = request.constraint_names or [None] * len(request.constraints)
        for idx, constraint in enumerate(request.constraints):
            name = constraint_names[idx] if idx < len(constraint_names) else None
            result = validator.validate_constraint(constraint, idx, name)
            results.append(result)
        
        passed = sum(1 for r in results if r.status == "PASS")
        failed = sum(1 for r in results if r.status == "FAIL")
        
        # Build schedule summary
        summary = _build_schedule_summary(
            request.config,
            request.schedule,
            request.ic_assignments,
            request.state_mapping
        )
        
        validation_time_ms = (time.time() - start_time) * 1000
        
        return ValidateResponse(
            overall_status="PASS" if failed == 0 else "FAIL",
            total_constraints=len(results),
            passed_constraints=passed,
            failed_constraints=failed,
            results=results,
            summary=summary,
            validation_time_ms=validation_time_ms
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Validation error: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Universal Scheduler Solver Engine"
    }


def _build_schedule_summary(
    config,
    schedule: dict,
    ic_assignments: dict | None,
    state_mapping: dict | None
) -> ScheduleSummary:
    """
    Build summary statistics from schedule.
    
    Args:
        config: Configuration with resources, time_slots, states
        schedule: Schedule matrix {resource_id: [state_per_timeslot]}
        ic_assignments: Optional IC matrix {resource_id: [time_slots]}
        state_mapping: Optional state name mapping {name: state_int}
        
    Returns:
        ScheduleSummary with vertical and horizontal breakdowns
    """
    time_slots = config.time_slots
    resources = config.resources
    states = config.states
    
    # Default state mapping if not provided
    if not state_mapping:
        state_mapping = {f"State_{s}": s for s in states}
    
    # Reverse mapping: state_int -> state_name
    state_to_name = {v: k for k, v in state_mapping.items()}
    
    # Convert ic_assignments to a lookup: (resource, time_slot) -> bool
    ic_lookup = set()
    if ic_assignments:
        for res_id, slots in ic_assignments.items():
            for slot in slots:
                ic_lookup.add((res_id, slot))
    
    # Build vertical summary (per time slot)
    vertical_summary = []
    for t in range(time_slots):
        counts = {name: 0 for name in state_mapping.keys()}
        ic_count = 0
        
        for res_id in resources:
            state = schedule[res_id][t]
            state_name = state_to_name.get(state, f"State_{state}")
            if state_name in counts:
                counts[state_name] += 1
            
            # Check IC
            if (res_id, t) in ic_lookup:
                ic_count += 1
        
        vertical_summary.append(VerticalSummary(
            time_slot=t,
            counts=counts,
            ic_count=ic_count
        ))
    
    # Build horizontal summary (per resource)
    horizontal_summary = []
    total_ic = 0
    
    for res_id in resources:
        counts = {name: 0 for name in state_mapping.keys()}
        ic_count = 0
        
        for t in range(time_slots):
            state = schedule[res_id][t]
            state_name = state_to_name.get(state, f"State_{state}")
            if state_name in counts:
                counts[state_name] += 1
            
            # Check IC
            if (res_id, t) in ic_lookup:
                ic_count += 1
        
        total_ic += ic_count
        horizontal_summary.append(HorizontalSummary(
            resource=res_id,
            counts=counts,
            ic_count=ic_count
        ))
    
    return ScheduleSummary(
        vertical_summary=vertical_summary,
        horizontal_summary=horizontal_summary,
        total_ic_count=total_ic
    )
