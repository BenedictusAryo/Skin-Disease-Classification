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
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    MAX_EPOCHS: int = 10
    NUM_WORKERS: int = 4
    LOG_EVERY_N_STEPS: int = 10
    EARLY_STOPPING_PATIENCE: int = 3
    IMAGE_SIZE: int = 224
    
    # Model
    MODEL_DIR: str = "models"
    
    # Config File
    class Config:
        env_file = env_file
        env_file_encoding = "utf-8"