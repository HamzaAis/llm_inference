from src.presentation.endpoints.health import router as health_router
from src.presentation.endpoints.inferences import router as inferences_router

__all__ = ["health_router", "inferences_router"]
