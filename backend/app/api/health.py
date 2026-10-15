import os
from fastapi import APIRouter
from backend.app.schemas.translation import HealthResponse
from backend.app.services.inference_service import InferenceService, get_default_checkpoint

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def check_health():
    default_ckpt = get_default_checkpoint()
    checkpoint_exists = os.path.exists(default_ckpt)
    device = InferenceService.get_system_device()

    return HealthResponse(
        status="healthy" if checkpoint_exists else "degraded",
        device=device,
        checkpoint_available=checkpoint_exists
    )
