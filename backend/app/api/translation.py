import os
import tempfile
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from backend.app.schemas.translation import TranslationResponse
from backend.app.services.inference_service import InferenceService

router = APIRouter(prefix="/api", tags=["Translation"])

ALLOWED_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}

@router.post("/translate", response_model=TranslationResponse)
async def translate_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename."
        )

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video file format '{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Save incoming upload to temporary file on disk
    temp_dir = tempfile.mkdtemp(prefix="signspeak_")
    temp_video_path = os.path.join(temp_dir, f"input_{os.path.basename(file.filename)}")

    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        translation, elapsed_ms, video_info, checkpoint_used = InferenceService.run_inference(temp_video_path)

        return TranslationResponse(
            translation=translation,
            confidence=None,  # Rule #4: Do not fabricate confidence scores
            processing_time_ms=elapsed_ms,
            video_info=video_info,
            checkpoint_used=checkpoint_used
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Video processing error: {str(val_err)}"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(exc)}"
        )
    finally:
        # Cleanup temp files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
