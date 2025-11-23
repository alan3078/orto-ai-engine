"""Configuration for the solver service."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    app_name: str = "Universal Scheduler - Solver Engine"
    version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    
    # Solver settings
    max_solve_time_seconds: int = 300  # 5 minutes max
    
    class Config:
        env_file = ".env"


settings = Settings()
