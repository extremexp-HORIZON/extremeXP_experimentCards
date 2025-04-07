import os

class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "postgresql://user:password@postgres:5432/experiments")
    SQLALCHEMY_TRACK_MODIFICATIONS = False