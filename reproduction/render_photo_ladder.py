"""Render figures/photo_ladder.svg from reproduction/photo_ladder_epoch29.json (standard library only).

    python reproduction/render_photo_ladder.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "photo_ladder_epoch29.json").read_text(encoding="utf-8"))
rows = d["per_L"]
L = [r["L"] for r in rows]
W, H, PAD_L, PAD_R, PAD_T, PAD_B = 920, 360, 64, 20, 46, 52
panel_w = (W - 3 * PAD_L) / 2
COL = {"r90": "#2563eb", "r95": "#ea580c", "r98": "#dc2626", "logit": "#16a34a"}


def x_of(v, x0):
    return x0 + v / 30 * panel_w


def y_of(v, lo, hi):
    return PAD_T + (1 - (v - lo) / (hi - lo)) * (H - PAD_T - PAD_B)


def axes(x0, lo, hi, ticks, title, ylabel):
    out = [f'<text x="{x0}" y="26" font-size="15" font-weight="600" fill="#1c1917">{title}</text>',
           f'<line x1="{x0}" y1="{H - PAD_B}" x2="{x0 + panel_w}" y2="{H - PAD_B}" stroke="#a8a29e"/>',
           f'<line x1="{x0}" y1="{PAD_T}" x2="{x0}" y2="{H - PAD_B}" stroke="#a8a29e"/>']
    for t in ticks:
        y = y_of(t, lo, hi)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_w}" y2="{y:.1f}" stroke="#e7e5e4"/>')
        out.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="#57534e">{t:g}</text>')
    for t in (0, 3, 6, 10, 16, 20, 24, 30):
        x = x_of(t, x0)
        out.append(f'<text x="{x:.1f}" y="{H - PAD_B + 18}" font-size="12" text-anchor="middle" fill="#57534e">{t}</text>')
    out.append(f'<text x="{x0 + panel_w / 2:.1f}" y="{H - 12}" font-size="12" text-anchor="middle" fill="#57534e">'
               f'added camera motion L (px)</text>')
    out.append(f'<text x="{x0 - 46}" y="{(PAD_T + H - PAD_B) / 2:.1f}" font-size="12" text-anchor="middle" fill="#57534e" '
               f'transform="rotate(-90 {x0 - 46} {(PAD_T + H - PAD_B) / 2:.1f})">{ylabel}</text>')
    return out


def line(x0, xs, ys, lo, hi, color, label, ly, kx=12):
    pts = " ".join(f"{x_of(x, x0):.1f},{y_of(y, lo, hi):.1f}" for x, y in zip(xs, ys))
    dots = "".join(f'<circle cx="{x_of(x, x0):.1f}" cy="{y_of(y, lo, hi):.1f}" r="3" fill="{color}"/>' for x, y in zip(xs, ys))
    key = (f'<line x1="{x0 + kx}" y1="{ly}" x2="{x0 + kx + 20}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>'
           f'<text x="{x0 + kx + 26}" y="{ly + 4}" font-size="12" fill="#1c1917">{label}</text>')
    return [f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{pts}"/>', dots, key]


svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui, sans-serif">',
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
x1 = PAD_L
svg += axes(x1, 0, 1, (0, 0.25, 0.5, 0.75, 1), f"Called blurred ({d['photos']} unseen photos, H.264 CRF 23)",
            "share called blurred")
for i, k in enumerate(("r90", "r95", "r98")):
    svg += line(x1, L, [r[f"blur_{k}"] for r in rows], 0, 1, COL[k], f"threshold {k} ({d['thresholds'][k]:.4f})",
                H - PAD_B - 66 + 18 * i, kx=panel_w * 0.52)
x2 = 2 * PAD_L + panel_w + PAD_L / 2
svg += axes(x2, -25, 25, (-20, -10, 0, 10, 20), "Median image logit", "logit")
svg += line(x2, L, [r["median_logit"] for r in rows], -25, 25, COL["logit"], "median over the photos", PAD_T + 14)
svg.append("</svg>")
out = HERE.parent / "figures" / "photo_ladder.svg"
out.write_text("\n".join(svg), encoding="utf-8")
print(out)
