"""Generate Thalamus app icons.

Produces a portable base PNG (pure Python), then:
  - assets/icon.ico   for the Windows build (Pillow, any platform)
  - assets/icon.icns  for the macOS build (iconutil, macOS only)

icon.icns is what desktop-app/build.sh bundles into Thalamus.app; icon.ico is
what desktop-app/build.ps1 passes to PyInstaller. Run directly:

    python generate_icon.py
"""

import io
import struct
import zlib
import subprocess
import shutil
import sys
from pathlib import Path

from PIL import Image

ASSETS = Path(__file__).parent / "assets"

ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
ICNS_SIZES = [16, 32, 64, 128, 256, 512]


def make_png(w, h, pixels):
    def chunk(ctype, cdata):
        c = ctype + cdata
        return struct.pack(">I", len(cdata)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(bytes(pixels), 9))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def generate_icon_png(size=1024):
    W = H = size
    pixels = bytearray()

    for y in range(H):
        pixels.append(0)
        for x in range(W):
            cx, cy = W / 2, H / 2
            corner_r = int(W * 0.22)

            dx = max(0, abs(x - cx) - (cx - corner_r))
            dy = max(0, abs(y - cy) - (cy - corner_r))
            dist = (dx * dx + dy * dy) ** 0.5

            if dist > corner_r:
                pixels.extend([0, 0, 0, 0])
                continue

            t = (x + y) / (W + H)
            r = int(108 + (162 - 108) * t)
            g = int(92 + (155 - 92) * t)
            b = int(231 + (254 - 231) * t)

            nx = (x - cx) / (W * 0.35)
            ny = (y - cy) / (H * 0.4)
            bolt = False
            if -0.15 <= nx <= 0.25 and -1.0 <= ny <= 0.0:
                if nx <= 0.25 - 0.4 * (ny + 1.0):
                    bolt = True
            if -0.25 <= nx <= 0.15 and 0.0 <= ny <= 1.0:
                if nx >= -0.25 + 0.4 * ny:
                    bolt = True
            if bolt:
                r, g, b = 255, 255, 255

            pixels.extend([r, g, b, 255])

    return make_png(W, H, pixels)


def _base_image() -> Image.Image:
    """The 1024px master icon as a Pillow RGBA image."""
    return Image.open(io.BytesIO(generate_icon_png(1024))).convert("RGBA")


def write_ico(base: Image.Image) -> Path:
    """Windows multi-resolution .ico (works on any platform)."""
    out = ASSETS / "icon.ico"
    base.save(out, format="ICO", sizes=ICO_SIZES)
    return out


def write_icns(base: Image.Image) -> Path:
    """macOS .icns: build the .iconset with Pillow, assemble with iconutil."""
    iconset = ASSETS / "icon.iconset"
    iconset.mkdir(exist_ok=True)
    try:
        for size in ICNS_SIZES:
            base.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
            double = size * 2
            base.resize((double, double), Image.LANCZOS).save(iconset / f"icon_{size}x{size}@2x.png")
        out = ASSETS / "icon.icns"
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True)
        return out
    finally:
        shutil.rmtree(iconset, ignore_errors=True)


def main():
    ASSETS.mkdir(exist_ok=True)
    base = _base_image()

    print(f"Generated {write_ico(base)}")

    if sys.platform == "darwin" and shutil.which("iconutil"):
        print(f"Generated {write_icns(base)}")
    else:
        print("Skipped icon.icns (macOS/iconutil only; not needed for the Windows build)")


if __name__ == "__main__":
    main()
