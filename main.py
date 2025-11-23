"""Railway / production entrypoint.

Starts the FastAPI solver engine using uvicorn. Railway provides the
listening port via the `PORT` environment variable; default to 8000 for
local runs if not set.
"""

import os
import sys
import traceback
import uvicorn


def main():
    port_env = os.environ.get("PORT")
    port = int(port_env) if port_env and port_env.isdigit() else 8000
    print(f"[startup] Detected PORT={port_env}; binding to {port}")
    print(f"[startup] Python: {sys.version.split()[0]} | cwd={os.getcwd()}")
    try:
        # Import inside main to avoid heavy imports during build; any failure will be logged.
        from src.main import app  # noqa: WPS433
    except Exception as import_error:  # pragma: no cover
        print("[error] Failed to import src.main.app:")
        traceback.print_exc()
        sys.exit(1)

    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as run_error:  # pragma: no cover
        print("[error] uvicorn failed to start:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
