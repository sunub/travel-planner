import logging

from fastapi import FastAPI

from travel_planner.api.router import api_router

metrics_logger = logging.getLogger("travel_planner")
metrics_logger.setLevel(logging.INFO)
if not metrics_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    metrics_logger.addHandler(handler)
metrics_logger.propagate = False

app = FastAPI(title="TripFit API")
app.include_router(api_router, prefix="/api/v1")
