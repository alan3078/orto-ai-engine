# Universal Scheduler - Solver Engine

Python FastAPI service that provides mathematical optimization using Google OR-Tools.

## Setup

```bash
# Install dependencies
uv sync

# Run development server
uv run uvicorn src.main:app --reload --port 8000
```

## API

- POST `/api/v1/solve` - Solve scheduling problem

## Documentation

See `/specs/FN/BE/ENG/001-Core_Universal_Solver/` for full specification.
