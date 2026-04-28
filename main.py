from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.infrastructure.config.database import dispose_engine, init_engine, init_db
from src.infrastructure.config.dependency import (
    build_model_service,
    clear_model_service,
    set_model_service,
)
from src.infrastructure.config.settings import get_settings
from src.infrastructure.middleware import AccessLogMiddleware, configure_logging, RateLimitMiddleware, RequestIDMiddleware
from src.presentation.endpoints import inferences_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    await asyncio.to_thread(lambda: settings.data_dir.mkdir(parents=True, exist_ok=True))
    await asyncio.to_thread(lambda: settings.hf_cache_dir.mkdir(parents=True, exist_ok=True))

    init_engine(settings)
    await init_db()

    model_service = build_model_service(settings)
    set_model_service(model_service)

    logger = logging.getLogger("llm_inferance.boot")
    device = "gpu" if settings.model_provider == "cuda" else "cpu"
    logger.info("loading model on startup, this may take a while on %s...", device)
    await model_service.load()
    logger.info("startup complete")

    try:
        yield
    finally:
        await model_service.unload()
        clear_model_service()
        await dispose_engine()


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allow_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(AccessLogMiddleware)

    app.include_router(inferences_router)

    return app


app = create_app()


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
