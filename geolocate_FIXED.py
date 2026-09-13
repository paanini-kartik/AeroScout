"""AeroScout: real detection timestamps, entirely simulated camera coordinates.

Run: python geolocate.py
Requires: pip install folium
This does NOT estimate fire coordinates or parse real drone telemetry.
"""
import json
import math
from pathlib import Path
import folium

BASE = Path(__file__).resolve().parent


def load_groups(path):
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list) or not records:
        raise ValueError("detections.json must contain a nonempty list of detections.")
    groups = {}
    times = []
    for index, record in enumerate(records):
        timestamp = record.get("timestamp_ms")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise ValueError(f"Detection {index}: timestamp_ms must be a number.")
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError(f"Detection {index}: invalid timestamp_ms.")
        second = int(timestamp / 1000)
        groups.setdefault(second, []).append(record)
        times.append(timestamp)
    return records, groups, min(times), max(times)


def position(timestamp, first, last):
    # Arbitrary fictional route near 49 N, 123 W; unrelated to the footage.
    fraction = (timestamp - first) / max(last - first, 1)
    return [49.40 + fraction * 0.00030, -123.15 + fraction * 0.00055]


def main():
    records, groups, first, last = load_groups(BASE / "detections.json")
    route = [position(first + (last - first) * i / 20, first, last)
             for i in range(21)]
    # Use Esri's public street-map layer because direct OpenStreetMap tiles may
    # reject maps opened locally from a file:// URL with a 403 response.
    map_view = folium.Map(location=route[10], zoom_start=18, tiles=None,
                          control_scale=True)
    folium.TileLayer(
        tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/"
               "World_Street_Map/MapServer/tile/{z}/{y}/{x}"),
        attr="Tiles &copy; Esri",
        name="Esri World Street Map",
        overlay=False,
        control=False,
    ).add_to(map_view)
    folium.PolyLine(route, color="#ad8539", weight=5,
                    tooltip="SIMULATED camera route, unrelated to footage").add_to(map_view)
    for second, detections in sorted(groups.items()):
        timestamp = sum(d["timestamp_ms"] for d in detections) / len(detections)
        frames = len({d.get("frame") for d in detections})
        popup = (f"<b>SIMULATED camera position</b><br>"
                 f"Video interval: {second}–{second + 1} seconds<br>"
                 f"Box observations: {len(detections)} across {frames} frames<br>"
                 "Repeated observations, not a count of unique fires.<br>"
                 "These coordinates do not locate the fire.")
        folium.CircleMarker(position(timestamp, first, last), radius=9,
                            color="#9c321f", fill=True, fill_color="#db6949",
                            fill_opacity=0.9, popup=folium.Popup(popup, max_width=310),
                            tooltip=f"{second}–{second + 1}s: click for detections").add_to(map_view)
    panel = f"""
    <div style="position:fixed;top:18px;left:55px;z-index:9999;background:white;
    padding:18px 22px;border-left:6px solid #ad8539;max-width:390px;
    font-family:Arial,sans-serif;box-shadow:0 2px 12px #0003">
    <div style="font-size:25px;font-weight:bold;color:#414141">AeroScout</div>
    <div style="font-size:16px;margin:6px 0">Detection timestamp mapping demo</div>
    <strong style="color:#9c321f">SIMULATED TELEMETRY</strong>
    <p style="font-size:14px;line-height:1.5;margin-bottom:0">
    Real detection log. Fictional camera route.<br>
    {len(records):,} box observations grouped into {len(groups)} time intervals.<br>
    Points show assumed camera positions, <b>not fire locations</b>.
    No real GPS or SRT data was used.</p></div>"""
    map_view.get_root().html.add_child(folium.Element(panel))
    map_view.fit_bounds(route, padding=(70, 70))
    output = BASE / "AeroScout_Map_FIXED.html"
    map_view.save(str(output))
    print(f"Created {output}")
    print(f"{len(records)} observations; {len(groups)} one-second groups.")
    print("Open AeroScout_Map_FIXED.html in your browser. Internet is needed for the map.")


if __name__ == "__main__":
    main()
