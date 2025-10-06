import sys, json, base64, io
from pathlib import Path

try:
    from PIL import Image  # Pillow
except Exception as e:
    print(json.dumps({"ok": False, "error": f"Pillow import failure: {e}"}))
    sys.exit(1)


def resize_to_224(path: Path):
    with Image.open(path) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        else:
            im = im.convert("RGB")
        resized = im.resize((224, 224))
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {
            "ok": True,
            "steps": ["Resize to 224x224 (Pillow)"],
            "processed_base64": b64,
            "pet_type": "Unknown",
        }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "Usage: resize_image.py <image_path>"}))
        return
    p = Path(sys.argv[1])
    if not p.exists():
        print(json.dumps({"ok": False, "error": "Image path not found"}))
        return
    try:
        res = resize_to_224(p)
        print(json.dumps(res))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))


if __name__ == "__main__":
    main()
