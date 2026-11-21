"""Отрисовка истории поиска DALAIN из SQLite, без расчёта волн."""
from functools import lru_cache
from pathlib import Path

from PIL import GifImagePlugin, Image, ImageDraw, ImageFont
from shapely.geometry import box

from .geometry import geometry_parts


COLORS = (
    "#ffffff", "#dceef8", "#142a3b", "#878773", "#f4ead4", "#249467",
    "#db5b39", "#a82c55", "#726486", "#eac86d", "#3979bf", "#56baa6",
    "#984ec2", "#ef9260", "#92b54e", "#59798d",
)
STAGES = (10, 11, 12, 13, 14, 9)
STATUS = {
    "baseline": ("BASELINE", 15),
    "trial": ("STAGE EVALUATED", 10),
    "leader": ("NEW BEST", 5),
    "worse": ("NOT BETTER", 6),
    "duplicate": ("DUPLICATE", 8),
    "decode": ("NO FEASIBLE CUT", 7),
    "invalid": ("GEOMETRY REJECTED", 7),
    "checkpoint": ("CHECKPOINT SAVED", 5),
    "final": ("ACCEPTED CONSTRUCTION", 5),
}


def animation(path, events, geometry, target, *, fps, width, height):
    """Потоковая запись: в памяти только один кадр и небольшой кеш контуров."""
    path = Path(path)
    tmp = path.with_suffix(".gif.tmp")
    palette = bytes(int(color[i:i + 2], 16) for color in COLORS for i in (1, 3, 5))
    font = ImageFont.load_default(size=15)
    title = ImageFont.load_default(size=22)
    cfg = geometry.cfg
    contour = geometry._fill(target)
    radius = max(abs(x) for x in contour.bounds) + 2 * cfg.cell_side_m
    scale = min(width - 44, height - 168) / (2 * radius)
    cy = 104 + (height - 150) / 2

    def xy(x, y):
        return round(width / 2 + scale * x), round(cy - scale * y)

    def stroke(draw, geom, color, weight=1):
        for line in geometry_parts(geom, {"LineString", "LinearRing"}):
            draw.line([xy(x, y) for x, y, *_ in line.coords], fill=color, width=weight)

    base = Image.new("P", (width, height), 0)
    base.putpalette(palette)
    draw = ImageDraw.Draw(base)
    draw.rectangle((18, 100, width - 18, height - 43), fill=1)
    viewport = box(-radius, -radius, radius, radius)
    for poly in geometry_parts(geometry.local_land.intersection(viewport), {"Polygon"}):
        draw.polygon([xy(x, y) for x, y in poly.exterior.coords], fill=4)
        for hole in poly.interiors:
            draw.polygon([xy(x, y) for x, y in hole.coords], fill=1)
    shore = geometry.original_coast.intersection(viewport)
    stroke(draw, shore, 3, 2)

    @lru_cache(maxsize=96)
    def polygons(part):
        return geometry_parts(geometry._fill(frozenset(part)), {"Polygon"})

    def paint(draw, part, color):
        for poly in polygons(tuple(map(tuple, part))):
            edge = [xy(x, y) for x, y in poly.exterior.coords]
            # Маска сохраняет отверстия заходки, не стирая предыдущие этапы.
            if poly.interiors:
                mask = Image.new("1", base.size)
                m = ImageDraw.Draw(mask)
                m.polygon(edge, fill=1)
                for hole in poly.interiors:
                    m.polygon([xy(x, y) for x, y in hole.coords], fill=0)
                draw.bitmap((0, 0), mask, fill=color)
            else:
                draw.polygon(edge, fill=color)
            stroke(draw, poly.boundary, 2)

    def render(event, index):
        label, color = STATUS[event["status"]]
        im = base.copy()
        draw = ImageDraw.Draw(im)
        parts = event.get("plan") or []
        for i, part in enumerate(parts):
            paint(draw, part, STAGES[i % len(STAGES)])
        stroke(draw, contour.boundary, 2, 2)
        if parts and event["status"] != "checkpoint":
            active = min(event.get("active") or len(parts), len(parts)) - 1
            for poly in polygons(tuple(map(tuple, parts[active]))):
                stroke(draw, poly.boundary, color, 3)
        if event.get("seed") is not None:
            x, y = xy(*(v * cfg.cell_side_m for v in event["seed"]))
            draw.ellipse((x-5, y-5, x+5, y+5), fill=color, outline=2)
        step, attempt = event["step"], event.get("attempt")
        draw.rectangle((0, 0, width, 39), fill=color)
        draw.text((18, 8), label, fill=0, font=title)
        draw.text((18, 48), f"{event['mode'].upper()} | step {step} | "
                  f"attempt {attempt if attempt is not None else '-'} | frame {index}",
                  fill=2, font=font)
        score = event.get("score")
        best = event.get("best")
        kind = event.get("score_kind", "State score")
        v = "-" if score is None else f"{score:.7g}"
        b = "-" if best is None else f"{best:.7g}"
        draw.text((18, 73), f"{kind}: {v}    Best: {b}", fill=2, font=font)
        note = event.get("note") or "Dark outline: target  |  Colors: construction stages"
        draw.text((18, height - 32), note, fill=2, font=font)
        return im

    duration = max(10, round(100 / fps) * 10)
    try:
        with tmp.open("wb") as stream:
            count = 0
            for count, event in enumerate(events, 1):
                frame = render(event, count)
                if count == 1:
                    header, _ = GifImagePlugin.getheader(frame, info={"loop": 0, "optimize": False})
                    stream.writelines(header)
                stream.writelines(GifImagePlugin.getdata(frame, duration=duration, disposal=1))
            stream.write(b";")
        if count:
            tmp.replace(path)
        return count
    finally:
        tmp.unlink(missing_ok=True)
