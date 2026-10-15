from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.translation import router as translation_router
from backend.app.api.health import router as health_router

app = FastAPI(
    title="SignSpeak API",
    description="Continuous Sign-Language Understanding API powered by PyTorch ML Engine",
    version="1.0.0"
)

# Configure CORS for local development and frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(translation_router)
app.include_router(health_router)


@app.get("/")
def read_root():
    return {
        "app": "SignSpeak API",
        "status": "online",
        "docs": "/docs"
    }
