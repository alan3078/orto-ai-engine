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

### Python Version & OR-Tools

Google OR-Tools may not publish wheels immediately for brand new Python releases (e.g. 3.13). If the
platform auto-selects Python 3.13 you can see startup failures / 502 responses. We pin the runtime to
Python 3.12 via `pyproject.toml (>=3.12,<3.13)` and `.mise.toml` to ensure a compatible wheel.

If OR-Tools cannot be imported the API will still start, but `/api/v1/solve` returns an ERROR status
indicating the solver is unavailable.

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
