"""Authenticated worker API for processing Supabase-backed ArtSpeak sessions."""

import os
import tempfile
import math
import traceback
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import Client, create_client

from services.inference_engine.process_session import process_session
from services.stable_diffusion_service import generate_artwork

app = FastAPI(title="ArtSpeak Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_admin_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
    return create_client(url, key)


def authenticated_user_id(client: Client, authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    try:
        user = client.auth.get_user(authorization.removeprefix("Bearer ").strip())
        return user.user.id
    except Exception as error:
        raise HTTPException(status_code=401, detail="Invalid access token") from error


def rows(data):
    return data.data if data and data.data else []


@app.post("/process/{session_id}")
def process(session_id: str, authorization: str | None = Header(default=None)):
    client = get_admin_client()
    user_id = authenticated_user_id(client, authorization)
    session_result = client.table("sessions").select("*").eq("id", session_id).single().execute()
    session = session_result.data
    if not session or session["caregiver_id"] != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        touch = rows(client.table("session_touch_events").select("*").eq("session_id", session_id).order("timestamp_ms").execute())
        video_rows = rows(client.table("session_video_recordings").select("storage_path").eq("session_id", session_id).execute())

        video_path = None
        with tempfile.TemporaryDirectory() as temp_dir:
            if video_rows:
                video_path = str(Path(temp_dir) / "session.mp4")
                video_bytes = client.storage.from_("session-videos").download(video_rows[0]["storage_path"])
                Path(video_path).write_bytes(video_bytes)

            report = process_session(
                session_id=session_id,
                video_path=video_path,
                touch_events=[{
                    "x": item["x"], "y": item["y"], "t": item["timestamp_ms"],
                    "pressure": item.get("pressure", 0.5),
                } for item in touch],
                e4_events=[],
                duration_minutes=max(1, math.ceil((session["video_duration_seconds"] + session["art_duration_seconds"]) / 60)),
            )

        affect = report.get("affect_vector", {})
        artwork_b64, artwork_prompt = generate_artwork(
            valence=affect.get("valence", 0),
            arousal=affect.get("arousal", 0),
            dominance=affect.get("dominance", 0),
            interest=session.get("interest_category", "nature"),
        )
        client.table("predictions_facial_emotion").insert({
            "session_id": session_id, "timestamp_sec": 0,
            "dominant_emotion": report.get("facial_expression", "Neutral"),
            "confidence": max(report.get("facial_expression_confidence", 0.0), 0.0),
            "valence": affect.get("valence", 0), "arousal": affect.get("arousal", 0),
            "dominance": affect.get("dominance", 0),
            "emotion_probabilities": report.get("facial_probabilities", {}),
        }).execute()
        gaze = report.get("gaze_attention", {})
        touch_prediction = report.get("touch_prediction", {})
        client.table("predictions_touch_rhythm").insert({
            "session_id": session_id, "window_start_sec": 0,
            "window_end_sec": session["video_duration_seconds"] + session["art_duration_seconds"],
            "behavioral_state": report["overall_state"], "state_probabilities": report.get("regulation_breakdown", {}),
            "touch_valence": affect.get("valence", 0), "touch_arousal": affect.get("arousal", 0),
            "touch_dominance": affect.get("dominance", 0),
            "stroke_speed_avg": touch_prediction.get("stroke_speed_avg"),
            "pressure_avg": touch_prediction.get("pressure_avg"),
            "stereotypy_detected": False,
        }).execute()
        client.table("session_reports").upsert({
            "session_id": session_id, "caregiver_id": session["caregiver_id"], "child_id": session["child_id"],
            "overall_state": report["overall_state"], "regulation_score_pct": report["regulation_score_pct"],
            "valence": affect.get("valence", 0), "arousal": affect.get("arousal", 0), "dominance": affect.get("dominance", 0),
            "insights": report.get("insights", []), "recommendations": report.get("recommendations", []),
            "distress_alerts": [], "report_markdown": report["markdown_report"], "raw_report_json": report,
            "artwork_base64": artwork_b64, "artwork_prompt": artwork_prompt,
        }, on_conflict="session_id").execute()
        client.table("sessions").update({"status": "completed"}).eq("id", session_id).execute()
        return {"session_id": session_id, "status": "completed"}
    except Exception as error:
        client.table("sessions").update({"status": "failed"}).eq("id", session_id).execute()
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Inference processing failed: {type(error).__name__}: {error}",
        ) from error
