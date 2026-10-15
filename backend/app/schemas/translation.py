from typing import Optional, Dict, Any
from pydantic import BaseModel


class TranslationResponse(BaseModel):
    translation: str
    confidence: Optional[float] = None
    processing_time_ms: float
    video_info: Dict[str, Any]
    checkpoint_used: str


class HealthResponse(BaseModel):
    status: str
    device: str
    checkpoint_available: bool
