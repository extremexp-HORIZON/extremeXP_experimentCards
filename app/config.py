import os

from urllib.parse import urlparse, urlunparse

def _ensure_db_name(uri: str, default_db: str = "experiments") -> str:
    parsed = urlparse(uri)
    if parsed.path and parsed.path != "/":
        return uri
    return urlunparse(parsed._replace(path=f"/{default_db}"))
class Config:
    # SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "postgresql://user:password@postgres:5432/experiments")
    SQLALCHEMY_DATABASE_URI = _ensure_db_name(
        os.getenv("DATABASE_URL", "postgresql://root:password@postgres:5432/experiments")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DAL_BASE_URL = os.getenv(
        "DAL_BASE_URL",
        "https://api.dal.extremexp-icom.intracom-telecom.com/api",
    )
    ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
    DAL_SYNC_ENABLED = os.getenv("DAL_SYNC_ENABLED", "false").lower() in ("1", "true", "yes")
    DAL_SYNC_INTERVAL_MINUTES = int(os.getenv("DAL_SYNC_INTERVAL_MINUTES", "15"))



