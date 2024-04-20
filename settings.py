"""
Configuration Settings for the application
"""
from pathlib import Path

from modules.logger.console_log import setup_logging
from pydantic_settings import BaseSettings

logger = setup_logging()

# Load Environment File
env_file = Path(__file__).parent / "envs/.env.prod"
if not env_file.exists():
    logger.error(
        f"Environment File {env_file} not found. please create one based on envs/.env.dev. exiting..."
    )
    raise FileNotFoundError(f"Environment File {env_file} not found. exiting...")


class AppSettings(BaseSettings):
    """Settings for the application"""

    ENV: str = "DEV"

    # Trainer