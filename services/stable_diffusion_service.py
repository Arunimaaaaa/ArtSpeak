"""ArtSpeak Stable Diffusion artwork generation service."""

from dotenv import load_dotenv
load_dotenv('.env.inference')

import base64
import os

import requests

HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
HF_TOKEN = os.environ.get("HF_API_TOKEN", "")
TIMEOUT = 60

INTEREST_THEMES = {
    "trains": "soft railway tracks, gentle steam clouds, locomotive silhouette",
    "space": "soft stars, pastel nebula, gentle cosmic glow, planets",
    "nature": "gentle leaves, soft flowers, calm waves, forest light",
    "geometric": "soft geometric shapes, circles, flowing patterns, symmetry",
    "animals": "gentle bird shapes, butterflies, soft animal silhouettes",
}


def vad_to_prompt(valence: float, arousal: float, dominance: float, interest: str) -> tuple[str, str]:
    if valence > 0.3:
        mood = "warm, gentle, hopeful, soft, uplifting"
    elif valence > -0.3:
        mood = "calm, neutral, soft, balanced, serene"
    else:
        mood = "deeply soothing, peaceful, quiet, healing, restorative"
    if arousal > 0.7:
        energy, colors = "very gentle, minimal, soft edges, slow flowing", "muted blue, soft lavender, pale sage green, low contrast"
    elif arousal > 0.4:
        energy, colors = "flowing, smooth, gentle movement, gradual", "cool blue, soft teal, light purple, pastel"
    else:
        energy, colors = "light, airy, serene, open", "warm pastels, soft peach, gentle yellow, cream"
    composition = (
        "simple composition, minimal elements, lots of open space"
        if dominance < 0.3 else "balanced composition, gentle details"
    )
    theme = INTEREST_THEMES.get(interest, "soft abstract flowing shapes, gentle curves")
    prompt = (
        f"abstract therapeutic artwork, {mood}, {energy}, {colors}, {composition}, {theme}, "
        "watercolor style, soft brushstrokes, dreamy, no text, no faces, no people, no words, "
        "child-safe, sensory friendly, high quality art"
    )
    negative = (
        "realistic, photographic, dark, scary, violent, disturbing, text, watermark, signature, "
        "logo, blurry, low quality, high contrast, bright neon, harsh colors, sharp edges"
    )
    return prompt, negative


def _generate_placeholder(valence: float, arousal: float, interest: str) -> str:
    """
    Generates a simple colour-coded PIL image as base64 PNG.
    Used as fallback when Stable Diffusion API is unavailable.
    Colour reflects emotional state: calm=blue, dysregulated=warm orange.
    """
    from PIL import Image, ImageDraw, ImageFont
    import io

    # Pick background colour from VAD
    if arousal > 0.7:
        bg = (255, 220, 180)   # warm muted orange - high arousal
    elif arousal > 0.4:
        bg = (180, 210, 240)   # soft blue - moderate arousal
    else:
        bg = (200, 230, 210)   # soft green - calm

    img = Image.new("RGB", (512, 512), color=bg)
    draw = ImageDraw.Draw(img)

    # Draw simple shapes based on interest category
    shapes = {
        "trains":    [(100, 250, 400, 300), (80, 220, 420, 260)],
        "space":     [(230, 230, 280, 280), (200, 200, 310, 310)],
        "nature":    [(256, 150, 256, 400), (150, 300, 360, 400)],
        "geometric": [(150, 150, 360, 360), (180, 180, 330, 330)],
        "animals":   [(180, 180, 330, 330), (220, 140, 290, 200)],
    }
    outline_color = tuple(max(0, c - 40) for c in bg)
    for shape in shapes.get(interest, [(150, 150, 360, 360)]):
        draw.ellipse(shape, outline=outline_color, width=6)

    # Label
    draw.text((20, 480), f"ArtSpeak · VAD ({valence:+.1f}, {arousal:.1f})", fill=outline_color)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_artwork(
    valence: float, arousal: float, dominance: float, interest: str = "nature"
) -> tuple[str | None, str]:
    prompt, negative = vad_to_prompt(valence, arousal, dominance, interest)
    if not HF_TOKEN:
        print("[ArtworkService] HF_API_TOKEN not set — skipping generation.")
        print(f"[ArtworkService] Prompt would have been: {prompt[:100]}...")
        return _generate_placeholder(valence, arousal, interest), prompt
    try:
        response = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": prompt, "parameters": {
                "negative_prompt": negative, "num_inference_steps": 20,
                "guidance_scale": 7.5, "width": 512, "height": 512,
            }},
            timeout=TIMEOUT,
        )
        if response.status_code == 503:
            import time
            time.sleep(20)
            response = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": prompt, "parameters": {"num_inference_steps": 20, "guidance_scale": 7.5}},
                timeout=TIMEOUT,
            )
        if response.status_code == 200:
            return base64.b64encode(response.content).decode("utf-8"), prompt
        print(f"[ArtworkService] Generation failed: HTTP {response.status_code} — {response.text[:200]}")
        return _generate_placeholder(valence, arousal, interest), prompt
    except requests.exceptions.Timeout:
        print("[ArtworkService] Request timed out.")
    except Exception as e:
        print(f"[ArtworkService] Network error: {e}")
        print("[ArtworkService] Returning placeholder image.")
        return _generate_placeholder(valence, arousal, interest), prompt
    return None, prompt
