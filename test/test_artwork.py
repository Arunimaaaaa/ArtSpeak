import sys
sys.path.insert(0, '.')

from services.stable_diffusion_service import generate_artwork


def test_artwork_generation():
    print("Test 1: Calm state (nature)")
    b64, prompt = generate_artwork(
        valence=0.4,
        arousal=0.2,
        dominance=0.6,
        interest="nature",
    )
    print(f"Prompt: {prompt[:120]}...")
    print(f"Image generated: {b64 is not None}")
    if b64:
        print(f"Image size: {len(b64)} base64 chars")
    else:
        print("No image returned - check HF_API_TOKEN in .env.inference")

    print("\nTest 2: Dysregulated state (trains)")
    b64_2, prompt_2 = generate_artwork(
        valence=-0.7,
        arousal=0.9,
        dominance=0.1,
        interest="trains",
    )
    print(f"Prompt: {prompt_2[:120]}...")
    print(f"Image generated: {b64_2 is not None}")


if __name__ == "__main__":
    test_artwork_generation()
