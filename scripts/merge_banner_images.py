"""
Merge heimdall_logo.png with Waymo_Flotte.jpg at heimdall_banner.png dimensions.
Fades the fleet background and adds Bifrost aurora/rainbow overlay so Heimdall stands out.
Output: docs/assets/images/heimdall_waymo_banner.png
"""
from pathlib import Path

try:
    from PIL import Image, ImageEnhance
except ImportError:
    raise SystemExit("Install Pillow: pip install Pillow")

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO_ROOT / "docs" / "assets" / "images"
BANNER_PATH = IMAGES_DIR / "heimdall_banner.png"
WAYMO_PATH = IMAGES_DIR / "Waymo_Flotte.jpg"
LOGO_PATH = IMAGES_DIR / "heimdall_logo.png"
BIFROST_PATH = IMAGES_DIR / "bifrost_aurora.png"
OUT_PATH = IMAGES_DIR / "heimdall_waymo_banner.png"

# Fade fleet so Heimdall is clear; aurora strength over the background
BACKGROUND_BRIGHTNESS = 0.48
AURORA_OPACITY = 0.52


def _crop_to_fill(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scale and center-crop image to exact target size (no distortion)."""
    w, h = img.size
    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def main():
    banner = Image.open(BANNER_PATH).convert("RGBA")
    target_w, target_h = banner.size

    # Waymo base: crop to banner size, then fade so background doesn't compete with Heimdall
    waymo = Image.open(WAYMO_PATH).convert("RGBA")
    waymo = _crop_to_fill(waymo, target_w, target_h)
    waymo = ImageEnhance.Brightness(waymo).enhance(BACKGROUND_BRIGHTNESS)

    # Bifrost aurora/rainbow: same crop, overlay at partial opacity for glow
    if BIFROST_PATH.exists():
        bifrost = Image.open(BIFROST_PATH).convert("RGBA")
        bifrost = _crop_to_fill(bifrost, target_w, target_h)
        r, g, b, a = bifrost.split()
        a = a.point(lambda v: int(v * AURORA_OPACITY))
        bifrost = Image.merge("RGBA", (r, g, b, a))
        waymo = Image.alpha_composite(waymo, bifrost)

    # Heimdall on top: full height, centered, no bottom gap (clearly visible)
    logo = Image.open(LOGO_PATH).convert("RGBA")
    logo_h = target_h
    logo_w = int(logo.width * (logo_h / logo.height))
    logo = logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
    x = (target_w - logo_w) // 2
    y = 0

    merged = waymo.copy()
    merged.paste(logo, (x, y), logo)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.convert("RGB").save(OUT_PATH, "PNG", optimize=True)
    print(f"Saved: {OUT_PATH} ({target_w}x{target_h})")


if __name__ == "__main__":
    main()
