"""Railway / production entrypoint.

Starts the FastAPI solver engine using uvicorn. Railway provides the
listening port via the `PORT` environment variable; default to 8000 for
local runs if not set.
"""

import os
import uvicorn


def main():
    port = int(os.environ.get("PORT", "8000"))
    # Import inside main to avoid side effects during build phase
    from src.main import app  # noqa: WPS433
    print(f"[startup] Launching solver engine on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
