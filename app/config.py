import os

class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "postgresql://user:password@postgres:5432/experiments")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DAL_BASE_URL = os.getenv(
        "DAL_BASE_URL",
        "https://api.dal.extremexp-icom.intracom-telecom.com/api",
    )
    ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
    DAL_SYNC_ENABLED = os.getenv("DAL_SYNC_ENABLED", "false").lower() in ("1", "true", "yes")
    DAL_SYNC_INTERVAL_MINUTES = int(os.getenv("DAL_SYNC_INTERVAL_MINUTES", "15"))
