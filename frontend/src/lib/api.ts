/**
 * lib/api.ts
 * Typed fetch wrapper for the SignSpeak FastAPI backend.
 * All calls are proxied via Next.js rewrites: /api/* → http://127.0.0.1:8000/api/*
 */

// ─── Data Models ─────────────────────────────────────────────────────────────

export interface VideoInfo {
  duration_sec: number;
  fps: number;
  width: number;
  height: number;
  frame_count: number;
}

export interface TranslationResponse {
  translation: string;
  confidence: number | null;
  processing_time_ms: number;
  video_info: VideoInfo;
  checkpoint_used: string;
}

export interface HealthResponse {
  status: string;
  device: string;
  checkpoint_available: boolean;
}

// ─── Error Type ───────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public detail?: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// ─── API Functions ────────────────────────────────────────────────────────────

/**
 * Upload a video file for sign-language translation.
 *
 * @param file - The video File to translate.
 * @returns TranslationResponse from the backend.
 * @throws ApiError on HTTP errors, network failures, or malformed responses.
 */
export async function translateVideo(file: File): Promise<TranslationResponse> {
  const formData = new FormData();
  formData.append("file", file);

  let response: Response;

  try {
    response = await fetch("/api/translate", {
      method: "POST",
      body: formData,
    });
  } catch (err) {
    // TypeError indicates a network failure (backend unreachable, CORS, etc.)
    if (err instanceof TypeError) {
      throw new ApiError(
        "Backend unavailable — is the server running?",
        0
      );
    }
    throw err;
  }

  if (!response.ok) {
    let detail: string = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) {
        detail = body.detail;
      }
    } catch {
      // JSON parse failed — keep statusText as the detail message
    }
    throw new ApiError(detail, response.status, detail);
  }

  try {
    const data = (await response.json()) as TranslationResponse;
    return data;
  } catch {
    throw new ApiError("Unexpected response format", response.status);
  }
}

/**
 * Check backend health and device availability.
 *
 * @returns HealthResponse from the backend.
 * @throws ApiError if the backend is unreachable or returns an error.
 */
export async function checkHealth(): Promise<HealthResponse> {
  let response: Response;

  try {
    response = await fetch("/api/health");
  } catch {
    throw new ApiError("Backend unreachable", 0);
  }

  if (!response.ok) {
    throw new ApiError("Backend unreachable", response.status);
  }

  try {
    const data = (await response.json()) as HealthResponse;
    return data;
  } catch {
    throw new ApiError("Unexpected response format", response.status);
  }
}
