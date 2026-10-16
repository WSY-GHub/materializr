#!/usr/bin/env python3
"""Bake a Radiance .hdr panorama into the viewport's reflection environment.

The output is src/viewport/EnvironmentImage.cpp: a JPEG atlas embedded as a
byte array, so every platform gets it without per-platform asset packaging
(Android and iOS glob src/, desktop lists the file).

The atlas holds the panorama pre-blurred for six roughness levels, which the
renderer uploads as the mip chain of one equirect texture and samples with
textureLod(roughness^2 * 5) - the same mapping as the reflection probe.

    level 0  1024x512  at (0, 512*0)    sharp
    level 1   512x256  at (0, 512)      then progressively blurrier
    level 2   256x128  at (512, 512)
    level 3   128x64   at (768, 512)
    level 4    64x32   at (896, 512)
    level 5    32x16   at (960, 512)

Radiance is stored compressed into [0, RANGE] and gamma-encoded; the shader
undoes it with RANGE * pow(texel, 2.2). Keep RANGE in step with
kEnvironmentRange in ShapeRenderer.cpp.

Needs numpy and Pillow. The source HDRI is not checked in; see
assets/ENVIRONMENT-CREDITS.md for where it comes from.

    bake_environment.py preview a.hdr b.hdr ...   tone-mapped PNGs to compare
    bake_environment.py bake in.hdr [--rotate DEG] [--out src/viewport/EnvironmentImage.cpp]
"""

import argparse
import io
import math
import os
import sys

import numpy as np
from PIL import Image

RANGE = 4.0
LEVELS = 6
TARGET_MEAN = 0.75   # average brightness after exposure, ~the old drawn studio's
# Added to every texel after exposure. A CAD camera mostly looks down, so top
# faces reflect the studio's ceiling, and a real one is dark enough to turn a
# steel part black. The lift keeps the room's structure but greys the darks.
LIFT = 0.30


def read_hdr(path):
    with open(path, "rb") as f:
        data = f.read()
    end = data.index(b"\n\n") + 2
    nl = data.index(b"\n", end)
    parts = data[end:nl].split()
    if parts[0] != b"-Y" or parts[2] != b"+X":
        sys.exit("unsupported orientation: %r" % data[end:nl])
    h, w = int(parts[1]), int(parts[3])
    pos = nl + 1
    out = np.zeros((h, w, 4), np.uint8)
    for y in range(h):
        if data[pos] != 2 or data[pos + 1] != 2:
            sys.exit("only RLE scanlines are supported")
        pos += 4
        for c in range(4):
            x = 0
            row = out[y, :, c]
            while x < w:
                n = data[pos]
                pos += 1
                if n > 128:
                    n -= 128
                    row[x:x + n] = data[pos]
                    pos += 1
                else:
                    row[x:x + n] = np.frombuffer(data, np.uint8, n, pos)
                    pos += n
                x += n
    e = out[..., 3].astype(np.int32)
    scale = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0)
    return (out[..., :3].astype(np.float64) + 0.5) * scale[..., None]


def downsample(img, w):
    """Box-filter halvings down to width w."""
    while img.shape[1] > w:
        img = 0.25 * (img[0::2, 0::2] + img[1::2, 0::2] + img[0::2, 1::2] + img[1::2, 1::2])
    return img


def directions(w, h):
    """Unit direction at each texel centre, matching the shader's lookup:
    u = atan(z, x) / 2pi + 0.5, v = acos(y) / pi (row 0 at the top)."""
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    phi = (u - 0.5) * 2.0 * math.pi
    theta = v * math.pi
    st = np.sin(theta)[:, None]
    d = np.stack([np.cos(phi)[None, :] * st,
                  np.repeat(np.cos(theta)[:, None], w, 1),
                  np.sin(phi)[None, :] * st], -1)
    return d.reshape(-1, 3), np.repeat(st, w, 1).reshape(-1)


def convolve(src, out_w, exponent):
    """Each output texel = cosine-power-lobe average of src around it."""
    sh, sw = src.shape[:2]
    sd, sa = directions(sw, sh)
    cols = src.reshape(-1, 3)
    od, _ = directions(out_w, out_w // 2)
    res = np.empty((od.shape[0], 3))
    for i in range(0, od.shape[0], 2048):
        dots = np.clip(od[i:i + 2048] @ sd.T, 0.0, None)
        wgt = dots ** exponent * sa[None, :] if exponent > 0 else (dots > 0) * sa[None, :]
        res[i:i + 2048] = (wgt @ cols) / wgt.sum(1, keepdims=True)
    return res.reshape(out_w // 2, out_w, 3)


def compress(x):
    """Linear to 1, then roll off toward RANGE so softboxes keep their shape."""
    k = RANGE - 1.0
    return np.where(x <= 1.0, x, 1.0 + k * (1.0 - np.exp(-(x - 1.0) / k)))


def encode(x):
    x = x + LIFT
    return np.clip(np.round(255.0 * (compress(x) / RANGE) ** (1.0 / 2.2)), 0, 255).astype(np.uint8)


def preview(paths):
    for p in paths:
        img = downsample(read_hdr(p), 1024)
        img *= TARGET_MEAN / img.mean()
        out = os.path.splitext(os.path.basename(p))[0] + "_preview.png"
        Image.fromarray(np.clip(255 * np.clip(img, 0, 1) ** (1 / 2.2), 0, 255).astype(np.uint8)).save(out)
        print(out, "max %.1f" % img.max())


def bake(path, rotate, out_path):
    full = read_hdr(path)
    if rotate:
        full = np.roll(full, int(round(rotate / 360.0 * full.shape[1])), axis=1)
    base = downsample(full, 1024)
    exposure = TARGET_MEAN / base.mean()
    levels = [base]
    for lvl in range(1, LEVELS):
        alpha = lvl / (LEVELS - 1)                  # = roughness^2 at this level
        exponent = max(2.0 / (alpha * alpha) - 2.0, 0.0)
        src = downsample(base, 256 if lvl == 1 else 128)
        levels.append(convolve(src, 1024 >> lvl, exponent))
        print("level", lvl, "exponent %.1f" % exponent, file=sys.stderr)

    atlas = np.zeros((768, 1024, 3), np.uint8)
    atlas[0:512, :] = encode(levels[0] * exposure)
    x = 0
    for lvl in range(1, LEVELS):
        h, w = levels[lvl].shape[:2]
        atlas[512:512 + h, x:x + w] = encode(levels[lvl] * exposure)
        x += w

    buf = io.BytesIO()
    Image.fromarray(atlas).save(buf, "JPEG", quality=92, subsampling=0, optimize=True)
    jpg = buf.getvalue()
    with open(out_path, "w") as f:
        f.write("// Generated by tools/bake_environment.py from %s - do not edit.\n"
                % os.path.basename(path))
        f.write("// The reflection environment: a CC0 studio panorama from Poly Haven\n"
                "// (assets/ENVIRONMENT-CREDITS.md), pre-blurred per roughness level.\n\n")
        f.write('#include "EnvironmentImage.h"\n\nnamespace materializr {\n\n')
        f.write("const unsigned char kEnvironmentJpeg[] = {\n")
        for i in range(0, len(jpg), 24):
            f.write(",".join(str(b) for b in jpg[i:i + 24]) + ",\n")
        f.write("};\nconst unsigned long kEnvironmentJpegSize = sizeof(kEnvironmentJpeg);\n\n"
                "} // namespace materializr\n")
    Image.fromarray(atlas).save(os.path.splitext(out_path)[0] + "_atlas.png") if os.environ.get("MZR_ATLAS_PNG") else None
    print("%s: %d bytes of JPEG, exposure %.4f" % (out_path, len(jpg), exposure))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preview")
    p.add_argument("hdr", nargs="+")
    b = sub.add_parser("bake")
    b.add_argument("hdr")
    b.add_argument("--rotate", type=float, default=0.0, help="spin the panorama, degrees")
    b.add_argument("--out", default="src/viewport/EnvironmentImage.cpp")
    a = ap.parse_args()
    if a.cmd == "preview":
        preview(a.hdr)
    else:
        bake(a.hdr, a.rotate, a.out)


if __name__ == "__main__":
    main()
