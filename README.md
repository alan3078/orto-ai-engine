# Universal Scheduler - Solver Engine

Python FastAPI service that provides mathematical optimization using Google OR-Tools.

## Setup

```bash
# Install dependencies
uv sync

# Run development server (hot reload)
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## Production / Railway

Railway sets the `PORT` environment variable; the container must bind to that value.
The root `main.py` now reads `PORT` and starts uvicorn accordingly. Default fallback: 8000.

```bash
# Local production-style run (simulates Railway)
PORT=8001 python main.py

# Railway automatically executes: python main.py
# (Ensure start command in Railway dashboard is just `python main.py`)
```

Health check endpoints:
- `GET /` (basic service info)
- `GET /api/v1/health` (returns `{"status":"healthy"}`)

Solve endpoint:
- `POST /api/v1/solve`

## API

- POST `/api/v1/solve` - Solve scheduling problem

## Documentation

See `/specs/FN/BE/ENG/001-Core_Universal_Solver/` for full specification.
