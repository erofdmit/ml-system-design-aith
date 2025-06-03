from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles
from app.routers.data_router import data_router
from app.routers.trip_router import trip_router
from app.routers.signal_router import signal_router
from app.routers.metrics.metrics_router import metrics_router
from settings import settings
import os

path_static = './'
path_static = os.path.join(path_static, 'static')


def create_app():
    app = FastAPI(
        title='trainCV',
        version='0.0.0',
        openapi_version='3.1.0',
        docs_url='/docs',
        openapi_url='/docs/openapi.json'
    )
    app.mount("/static", StaticFiles(directory=path_static), name="static")
    main_routers: tuple[APIRouter, ...] = (
        data_router,
        trip_router,
        signal_router,
        metrics_router
    )

    for router in main_routers:
        app.include_router(
            router=router,
            prefix=settings.ROOT_PATH
        )

    return app

