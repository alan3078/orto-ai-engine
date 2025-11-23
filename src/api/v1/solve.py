"""
API router for the solver endpoints.
"""

import time
from fastapi import APIRouter, HTTPException
from src.core.schemas import SolveRequest, SolveResponse, ValidateRequest, ValidateResponse
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
        
        validation_time_ms = (time.time() - start_time) * 1000
        
        return ValidateResponse(
            overall_status="PASS" if failed == 0 else "FAIL",
            total_constraints=len(results),
            passed_constraints=passed,
            failed_constraints=failed,
            results=results,
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
