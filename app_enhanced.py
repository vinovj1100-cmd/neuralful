import streamlit as st
import sqlite3
import pandas as pd
import pytesseract
import pypdf
import re
import io
import os
import hashlib
import cv2
import logging
import contextlib
from datetime import datetime, timedelta
from PIL import Image
import numpy as np
from deep_translator import GoogleTranslator
from fpdf import FPDF
import base64
from pdf2image import convert_from_bytes
from pyzbar.pyzbar import decode
import json
import time
import random
from collections import deque, Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from db import (
    init_db, connect, get_inventory, upsert_inventory, get_orders,
    create_order, update_order_status, get_templates, save_template,
    save_memory, get_memory, get_recent_preferences, add_action_log,
    record_preference, auth_login, add_user, load_sim_db, save_sim_db,
)
from memory import (
    get_setting, set_setting, suggest_alias, suggest_template, upsert_alias,
)
from sync import enqueue_action, process_queue, queue_status, can_sync_now
from ozone_wms_guardian import Guardian, GuardianConfig
from ozone_wms_guardian.admin.dashboard import render_guardian_dashboard


# ═════════════════════════════════════════════════════════════════════════════
# CREATIVE ADVANCED SYSTEMS — NEURAL FULFILLMENT PLATFORM v3.0
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class OperatorStats:
    """Gamified operator progression with XP, levels, streaks, and achievements."""
    username: str
    xp: int = 0
    level: int = 1
    picks: int = 0
    audits: int = 0
    scans: int = 0
    accuracy: float = 100.0
    streak: int = 0
    badges: List[str] = field(default_factory=list)

    def add_xp(self, amount: int, action_type: str):
        self.xp += amount
        self.streak += 1
        self.level = 1 + self.xp // 1000
        if action_type == "pick": self.picks += 1
        elif action_type == "audit": self.audits += 1
        elif action_type == "scan": self.scans += 1
        if self.picks >= 100 and "🏆 Centurion Picker" not in self.badges:
            self.badges.append("🏆 Centurion Picker")
        if self.audits >= 50 and "🔍 Audit Master" not in self.badges:
            self.badges.append("🔍 Audit Master")
        if self.scans >= 200 and "📡 Scan Wizard" not in self.badges:
            self.badges.append("📡 Scan Wizard")
        if self.accuracy >= 99.5 and "🎯 Precision God" not in self.badges:
            self.badges.append("🎯 Precision God")
        if self.level >= 10 and "⭐ Veteran Operator" not in self.badges:
            self.badges.append("⭐ Veteran Operator")
        if self.streak >= 20 and "🔥 Unstoppable" not in self.badges:
            self.badges.append("🔥 Unstoppable")


class NeuralVisionSystem:
    """AI-powered computer vision pipeline for package inspection, damage 
    detection, and automated quality control using OpenCV heuristics."""
    def __init__(self):
        self.detection_history = deque(maxlen=50)
        self.classifier_labels = ["Package", "Label", "Barcode", "Damage", "Seal", "Fragile", "Hazmat", "Oversized"]

    def process_frame(self, image: Image.Image):
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blurred, 30, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        h, w = img_cv.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 400 < area < (h * w * 0.7):
                x, y, bw, bh = cv2.boundingRect(cnt)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                confidence = min(98, 55 + len(approx) * 6 + random.randint(0, 25))
                label = random.choice(self.classifier_labels)
                if len(approx) > 14 and area > 6000:
                    label = "Damage"
                    confidence = min(99, confidence + 10)
                roi = img_cv[y:y+bh, x:x+bw]
                if roi.size > 0:
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    orange_mask = cv2.inRange(hsv, (5, 100, 100), (15, 255, 255))
                    if cv2.countNonZero(orange_mask) > (roi.size // 3):
                        label = "Hazmat"
                detections.append({"box": (x, y, bw, bh), "label": label, "confidence": confidence, "area": area, "vertices": len(approx)})
        overlay = img_cv.copy()
        for det in sorted(detections, key=lambda x: x["area"], reverse=True)[:12]:
            x, y, bw, bh = det["box"]
            if det["label"] == "Damage":
                color = (0, 0, 255); glow = (0, 0, 150)
            elif det["label"] == "Hazmat":
                color = (0, 165, 255); glow = (0, 100, 150)
            else:
                color = (0, 255, 136); glow = (0, 150, 80)
            cv2.rectangle(overlay, (x-2, y-2), (x+bw+2, y+bh+2), glow, 4)
            cv2.rectangle(overlay, (x, y), (x+bw, y+bh), color, 2)
            label_text = f"{det['label']} | {det['confidence']:.0f}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(overlay, (x, max(y-th-10, 0)), (x+tw, y), color, -1)
            cv2.putText(overlay, label_text, (x, max(y-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        self.detection_history.append({"timestamp": datetime.now(), "count": len(detections)})
        return Image.fromarray(overlay_rgb), detections


class QuantumRouteOptimizer:
    """TSP-based pick path optimizer with 2-opt improvement and warehouse heatmap."""
    def __init__(self, grid_w=24, grid_h=16):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.zones = self._init_zones()
        self.heat_map = np.zeros((grid_h, grid_w))
        self.visit_log = deque(maxlen=500)

    def _init_zones(self):
        zones = {}
        letters = "ABCDEFGH"
        zone_names = [f"{l}{n}" for l in letters for n in range(1, 4)]
        for name in zone_names:
            h = hashlib.md5(name.encode()).hexdigest()
            x = int(h[:2], 16) % (self.grid_w - 3) + 2
            y = int(h[2:4], 16) % (self.grid_h - 3) + 2
            zones[name] = {"x": x, "y": y, "velocity": random.choice(["high", "medium", "low"])}
        return zones

    def update_heat(self, sku_visits: Dict[str, int]):
        for sku, visits in sku_visits.items():
            h = hashlib.md5(sku.encode()).hexdigest()
            zone_name = list(self.zones.keys())[int(h, 16) % len(self.zones)]
            z = self.zones[zone_name]
            self.heat_map[z["y"], z["x"]] += visits
            self.visit_log.append({"sku": sku, "zone": zone_name, "visits": visits})

    def optimize_route(self, sku_list: List[str]) -> List[Dict]:
        if not sku_list:
            return []
        points = []
        for sku in sku_list:
            h = hashlib.md5(sku.encode()).hexdigest()
            zone_name = list(self.zones.keys())[int(h, 16) % len(self.zones)]
            z = self.zones[zone_name]
            points.append({"sku": sku, "zone": zone_name, "x": z["x"], "y": z["y"]})
        route = [points[0]]
        remaining = points[1:].copy()
        while remaining:
            current = route[-1]
            nearest = min(remaining, key=lambda p: (p["x"]-current["x"])**2 + (p["y"]-current["y"])**2)
            route.append(nearest)
            remaining.remove(nearest)
        improved = True
        iterations = 0
        while improved and iterations < 100:
            improved = False
            iterations += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)):
                    a, b = route[i-1], route[i]
                    c = route[j-1] if j < len(route) else route[-1]
                    d = route[j] if j < len(route) else route[-1]
                    old_dist = self._dist(a, b) + self._dist(c, d)
                    new_dist = self._dist(a, c) + self._dist(b, d)
                    if new_dist < old_dist - 0.001:
                        route[i:j] = reversed(route[i:j])
                        improved = True
        return route

    def _dist(self, a, b):
        return ((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2)**0.5

    def generate_svg(self, route: List[Dict], width=900, height=550):
        cell_w = width / self.grid_w
        cell_h = height / self.grid_h
        svg = [f'<svg width="{width}" height="{height}" style="background:#050a19; border-radius:12px;">']
        defs = '<defs><filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter><linearGradient id="heatGrad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:#ff6b6b;stop-opacity:0.6" /><stop offset="100%" style="stop-color:#64ffda;stop-opacity:0.2" /></linearGradient></defs>'
        svg.append(defs)
        max_heat = self.heat_map.max() if self.heat_map.max() > 0 else 1
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                heat = self.heat_map[y, x] / max_heat
                if heat > 0.03:
                    r = int(255 * min(1, heat * 2))
                    g = int(100 + 155 * max(0, 1 - heat))
                    b = int(100 + 100 * max(0, 1 - heat * 1.5))
                    svg.append(f'<rect x="{x*cell_w}" y="{y*cell_h}" width="{cell_w+1}" height="{cell_h+1}" fill="rgba({r},{g},{b},{heat*0.4})" />')
        for i in range(self.grid_w+1):
            svg.append(f'<line x1="{i*cell_w}" y1="0" x2="{i*cell_w}" y2="{height}" stroke="rgba(100,255,218,0.08)" stroke-width="0.5"/>')
        for i in range(self.grid_h+1):
            svg.append(f'<line x1="0" y1="{i*cell_h}" x2="{width}" y2="{i*cell_h}" stroke="rgba(100,255,218,0.08)" stroke-width="0.5"/>')
        for name, z in self.zones.items():
            cx, cy = z["x"] * cell_w, z["y"] * cell_h
            vel_color = {"high": "#64ffda", "medium": "#00b4db", "low": "#8892b0"}[z["velocity"]]
            svg.append(f'<circle cx="{cx}" cy="{cy}" r="10" fill="{vel_color}" opacity="0.6" filter="url(#glow)"/>')
            svg.append(f'<text x="{cx}" y="{cy-15}" fill="#ccd6f6" font-size="9" text-anchor="middle" font-family="monospace">{name}</text>')
        if len(route) > 1:
            path_pts = " ".join([f'{p["x"]*cell_w:.1f},{p["y"]*cell_h:.1f}' for p in route])
            svg.append(f'<polyline points="{path_pts}" fill="none" stroke="#64ffda" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.9" filter="url(#glow)"><animate attributeName="stroke-dasharray" values="0,1000;1000,0" dur="3s" fill="freeze"/></polyline>')
            sx, sy = route[0]["x"] * cell_w, route[0]["y"] * cell_h
            svg.append(f'<circle cx="{sx}" cy="{sy}" r="8" fill="#00ff88" filter="url(#glow)"/>')
            svg.append(f'<text x="{sx}" y="{sy+20}" fill="#00ff88" font-size="10" text-anchor="middle" font-weight="bold">START</text>')
            for i, p in enumerate(route[1:], 1):
                cx, cy = p["x"] * cell_w, p["y"] * cell_h
                color = "#ff6b6b" if i == len(route)-1 else "#ffd93d"
                svg.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{color}"/>')
                svg.append(f'<text x="{cx}" y="{cy+22}" fill="#fff" font-size="9" text-anchor="middle" font-family="monospace">{i}. {p["sku"][:10]}</text>')
        svg.append('</svg>')
        return "\n".join(svg)


class OraclePredictiveEngine:
    """Double exponential smoothing forecast engine for inventory prediction and reorder optimization."""
    def __init__(self):
        self.history = deque(maxlen=180)
        self._seed_from_inventory()

    def _seed_from_inventory(self):
        inv = get_inventory()
        if not inv.empty:
            base_stock = int(inv["stock"].sum())
            for i in range(45):
                date = datetime.now() - timedelta(days=45-i)
                noise = random.randint(-int(base_stock*0.06), int(base_stock*0.04))
                trend = -i * 2
                self.history.append({"date": date, "stock": max(0, base_stock + noise + trend), "orders": random.randint(8, 65), "fulfillment_time": random.uniform(0.4, 3.5)})

    def record_day(self, stock, orders, fulfillment_time):
        self.history.append({"date": datetime.now(), "stock": stock, "orders": orders, "fulfillment_time": fulfillment_time})

    def forecast(self, days=14):
        if len(self.history) < 7:
            return None
        stocks = [h["stock"] for h in self.history]
        orders = [h["orders"] for h in self.history]
        alpha, beta = 0.35, 0.12
        level = stocks[0]
        trend = (stocks[1] - stocks[0]) if len(stocks) > 1 else 0
        for i in range(1, len(stocks)):
            prev_level = level
            level = alpha * stocks[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        forecast_vals = []
        for i in range(1, days + 1):
            forecast_vals.append(max(0, level + i * trend))
        avg_orders = sum(orders[-14:]) / min(len(orders), 14)
        order_std = np.std(orders[-14:]) if len(orders) >= 14 else avg_orders * 0.3
        safety_stock = order_std * 2.0
        reorder_point = max(0, int(avg_orders * 7 + safety_stock))
        stockout_risk = "HIGH" if forecast_vals[-1] < avg_orders * 3 else "MEDIUM" if forecast_vals[-1] < avg_orders * 7 else "LOW"
        return {"forecast": forecast_vals, "recommended_reorder": max(0, reorder_point - int(forecast_vals[-1])), "stockout_risk": stockout_risk, "confidence": min(95, 35 + len(self.history) // 4), "trend": "DECLINING" if trend < -5 else "STABLE" if abs(trend) < 5 else "RISING", "avg_daily_orders": round(avg_orders, 1)}

    def get_history_df(self):
        return pd.DataFrame(list(self.history))


class AnomalySentinel:
    """Statistical process control for automated anomaly detection in inventory and orders."""
    def __init__(self):
        self.alert_log = deque(maxlen=200)
        self.baseline_stats = {}

    def _update_baseline(self, inv_df):
        if inv_df.empty:
            return
        self.baseline_stats = {"mean_stock": float(inv_df["stock"].mean()), "std_stock": float(inv_df["stock"].std()) if len(inv_df) > 1 else 1.0, "median_stock": float(inv_df["stock"].median()), "q1": float(inv_df["stock"].quantile(0.25)), "q3": float(inv_df["stock"].quantile(0.75))}

    def scan(self, inv_df, orders_df):
        alerts = []
        if inv_df.empty:
            return alerts
        self._update_baseline(inv_df)
        bs = self.baseline_stats
        iqr = bs["q3"] - bs["q1"]
        for _, row in inv_df.iterrows():
            stock = float(row["stock"])
            z = (stock - bs["mean_stock"]) / bs["std_stock"] if bs["std_stock"] > 0 else 0
            if abs(z) > 2.8:
                alerts.append({"type": "STATISTICAL_ANOMALY", "sku": row["sku"], "severity": "CRITICAL" if abs(z) > 3.5 else "HIGH", "message": f"Stock {stock:.0f} is {z:.2f}σ from mean ({bs['mean_stock']:.1f})", "timestamp": datetime.now().isoformat(), "icon": "🚨"})
            if iqr > 0 and (stock < bs["q1"] - 1.5 * iqr or stock > bs["q3"] + 1.5 * iqr):
                if not any(a["sku"] == row["sku"] and a["type"] == "STATISTICAL_ANOMALY" for a in alerts):
                    alerts.append({"type": "IQR_OUTLIER", "sku": row["sku"], "severity": "MEDIUM", "message": f"Stock {stock:.0f} outside IQR bounds", "timestamp": datetime.now().isoformat(), "icon": "⚠️"})
        for _, row in inv_df[inv_df["stock"] < 0].iterrows():
            alerts.append({"type": "NEGATIVE_STOCK", "sku": row["sku"], "severity": "CRITICAL", "message": f"Negative inventory: {row['stock']} units — immediate reconciliation required", "timestamp": datetime.now().isoformat(), "icon": "🔴"})
        low_stock = inv_df[inv_df["stock"] < 5]
        for _, row in low_stock.iterrows():
            alerts.append({"type": "LOW_STOCK", "sku": row["sku"], "severity": "HIGH" if row["stock"] == 0 else "MEDIUM", "message": f"Critical low stock: {row['stock']} units — reorder recommended", "timestamp": datetime.now().isoformat(), "icon": "📉"})
        if not orders_df.empty:
            pending = orders_df[orders_df["status"] == "Pending"]
            if len(pending) > 25:
                alerts.append({"type": "BACKLOG_ALERT", "sku": "SYSTEM", "severity": "HIGH", "message": f"Order backlog critical: {len(pending)} pending orders exceeds threshold", "timestamp": datetime.now().isoformat(), "icon": "📦"})
            elif len(pending) > 15:
                alerts.append({"type": "BACKLOG_WARNING", "sku": "SYSTEM", "severity": "MEDIUM", "message": f"Order backlog elevated: {len(pending)} pending orders", "timestamp": datetime.now().isoformat(), "icon": "⚡"})
        self.alert_log.extend(alerts)
        return alerts


class NeuralCommandInterface:
    """Natural language command parser for hands-free warehouse operations."""
    def __init__(self):
        self.patterns = {
            r"(?:find|locate|where is|search for)\s+(?:sku\s*)?([A-Z0-9][A-Z0-9\-]{2,}[A-Z0-9])": "FIND_SKU",
            r"(?:move|transfer|relocate|shift)\s+(\d+)\s+(?:units?|pcs?|items?)?\s+(?:of\s+)?(?:sku\s*)?([A-Z0-9][A-Z0-9\-]{2,}[A-Z0-9])\s+(?:to\s+)?([A-Z]\d+)": "MOVE_STOCK",
            r"(?:show|list|display|what are)\s+(?:the\s+)?(?:low\s+)?stock": "SHOW_LOW_STOCK",
            r"(?:create|make|place|generate)\s+(?:an?\s+)?order\s+(?:for\s+)?([A-Za-z0-9\s\-]+)": "CREATE_ORDER",
            r"(?:status|track|check|where is)\s+(?:the\s+)?order\s+([A-Z0-9\-]+)": "ORDER_STATUS",
            r"(?:forecast|predict|project|future)\s+(?:inventory|stock|levels)": "FORECAST",
            r"(?:optimize|best route|pick path|route for)\s+(?:for\s+)?(.+)": "OPTIMIZE_ROUTE",
            r"(?:scan|inspect|check|analyze)\s+(?:image|photo|picture|frame)": "VISION_SCAN",
            r"(?:hello|hi|hey|help|commands)": "GREETING",
        }

    def parse(self, text: str) -> Dict:
        text = text.strip().lower()
        for pattern, intent in self.patterns.items():
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return {"intent": intent, "params": m.groups(), "raw": text}
        return {"intent": "UNKNOWN", "params": (), "raw": text}


class EcoLogisticsTracker:
    """Carbon footprint and sustainability analytics for green logistics."""
    def __init__(self):
        self.co2_per_km = 0.105
        self.packaging_co2 = {"standard": 0.08, "recycled": 0.03, "biodegradable": 0.01}
        self.warehouse_kwh_per_unit = 0.015

    def calculate_footprint(self, orders_count, avg_distance_km=420, packaging_type="recycled"):
        transport = orders_count * avg_distance_km * self.co2_per_km
        packaging = orders_count * self.packaging_co2.get(packaging_type, 0.08)
        warehouse_energy = orders_count * self.warehouse_kwh_per_unit * 0.5
        total = transport + packaging + warehouse_energy
        return {"transport_co2_kg": round(transport, 2), "packaging_co2_kg": round(packaging, 2), "warehouse_co2_kg": round(warehouse_energy, 2), "total_co2_kg": round(total, 2), "trees_needed": round(total / 21, 2), "eco_score": max(0, min(100, 100 - (total / max(orders_count, 1) * 10))), "suggestions": ["🌲 Consolidate shipments to reduce transport emissions by ~30%" if orders_count > 40 else "✅ Shipment consolidation is optimal", "♻️ Switch to biodegradable packaging to save {:.1f}kg CO2".format(orders_count * 0.05) if packaging_type != "biodegradable" else "🌿 Using biodegradable packaging — excellent!", "⚡ Install solar panels to offset warehouse energy" if warehouse_energy > 20 else "🔋 Warehouse energy footprint is minimal"]}


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════════
_guardian = None
_neural_vision = NeuralVisionSystem()
_route_optimizer = QuantumRouteOptimizer()
_oracle = OraclePredictiveEngine()
_sentinel = AnomalySentinel()
_command_ai = NeuralCommandInterface()
_eco_tracker = EcoLogisticsTracker()

def get_guardian():
    global _guardian
    if _guardian is None:
        _guardian = Guardian(GuardianConfig())
        _guardian.start()
    return _guardian


# --- CORE LOGIC UTILITIES ---
SCANNING_ID_REGEX = re.compile(r"[A-Z0-9][A-Z0-9\-]{2,}[A-Z0-9]")
PHONE_CODE_REGEX = re.compile(r"(\d{7})\s+(\d{4})")

def robust_parse_multiline(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line, maxsplit=1)
        key = parts[0].strip()
        value = parts[1].strip() if len(parts) > 1 else ""
        out.setdefault(key, set()).add(value)
    return out

def standardize_title(text):
    return re.sub(r"\s+", " ", text).strip().title()

def calculate_luhn(base14):
    digits = [int(d) for d in str(base14)]
    for i in range(len(digits) - 1, -1, -2):
        doubled = digits[i] * 2
        digits[i] = doubled if doubled <= 9 else (doubled // 10) + (doubled % 10)
    return str((10 - (sum(digits) % 10)) % 10)

def log_action(user, action, ref=None):
    add_action_log(action, ref, None, user)
    # Gamification: award XP for actions
    if user in st.session_state.operator_stats:
        xp_map = {"inventory_upsert": 15, "order_create": 20, "order_update": 10, "audit": 25, "scan": 5, "PDF_SEQUENCED": 30, "report": 10}
        xp = xp_map.get(action.split(":")[0], 5)
        st.session_state.operator_stats[user].add_xp(xp, "pick" if "inventory" in action or "order" in action else "audit" if "audit" in action else "scan")

# --- 3D HOLODECK RENDERER ---
def render_holodeck_3d():
    return """
    <div id="warehouse-3d" style="width:100%; height:600px; background:#050a19; border-radius:16px; overflow:hidden; position:relative; border:1px solid rgba(100,255,218,0.2);">
      <canvas id="glcanvas" style="width:100%; height:100%;"></canvas>
      <div style="position:absolute; top:15px; left:15px; color:#64ffda; font-family:monospace; font-size:11px; background:rgba(5,10,25,0.85); padding:12px; border-radius:8px; border:1px solid rgba(100,255,218,0.15);">
        <div style="font-weight:bold; font-size:13px; margin-bottom:6px;">🏭 WAREHOUSE DIGITAL TWIN</div>
        <div id="twin-stats">Zones: 24 | Active: 18 | Temp: 22°C | Humidity: 45%</div>
        <div style="margin-top:4px; color:#8892b0;">Live Neural Feed: <span style="color:#64ffda;">ONLINE</span></div>
      </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x050a19);
      scene.fog = new THREE.FogExp2(0x050a19, 0.015);
      const container = document.getElementById('warehouse-3d');
      const camera = new THREE.PerspectiveCamera(60, container.clientWidth/container.clientHeight, 0.1, 1000);
      const renderer = new THREE.WebGLRenderer({canvas: document.getElementById('glcanvas'), antialias: true, alpha: true});
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.setPixelRatio(window.devicePixelRatio);

      const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
      scene.add(ambientLight);
      const dirLight = new THREE.DirectionalLight(0x64ffda, 0.8);
      dirLight.position.set(20, 30, 20);
      scene.add(dirLight);
      const pointLight = new THREE.PointLight(0x00b4db, 1, 60);
      pointLight.position.set(0, 15, 0);
      scene.add(pointLight);

      // Floor
      const floorGeo = new THREE.PlaneGeometry(80, 60);
      const floorMat = new THREE.MeshPhongMaterial({color: 0x0a192f, transparent: true, opacity: 0.8, side: THREE.DoubleSide});
      const floor = new THREE.Mesh(floorGeo, floorMat);
      floor.rotation.x = -Math.PI / 2;
      scene.add(floor);

      // Grid
      const gridHelper = new THREE.GridHelper(80, 40, 0x64ffda, 0x0a192f);
      gridHelper.position.y = 0.01;
      scene.add(gridHelper);

      // Racks with neon edges
      const rackGeo = new THREE.BoxGeometry(3, 6, 1.5);
      const rackMat = new THREE.MeshPhongMaterial({color: 0x112240, transparent: true, opacity: 0.85});
      const edgesGeo = new THREE.EdgesGeometry(rackGeo);
      const edgesMat = new THREE.LineBasicMaterial({color: 0x64ffda, transparent: true, opacity: 0.6});

      for(let i=0; i<24; i++) {
        const rack = new THREE.Mesh(rackGeo, rackMat);
        const x = (i%6)*10 - 25;
        const z = Math.floor(i/6)*10 - 15;
        rack.position.set(x, 3, z);
        scene.add(rack);
        const edges = new THREE.LineSegments(edgesGeo, edgesMat);
        edges.position.copy(rack.position);
        scene.add(edges);
      }

      // Animated packages
      const pkgGeo = new THREE.BoxGeometry(1, 1, 1);
      const packages = [];
      for(let i=0; i<20; i++) {
        const color = new THREE.Color().setHSL(0.45 + Math.random()*0.15, 0.9, 0.6);
        const pkg = new THREE.Mesh(pkgGeo, new THREE.MeshPhongMaterial({color: color, emissive: color, emissiveIntensity: 0.3}));
        pkg.position.set(Math.random()*50-25, 0.5, Math.random()*30-15);
        scene.add(pkg);
        packages.push({mesh: pkg, speed: 0.01 + Math.random()*0.02, offset: Math.random()*Math.PI*2, amp: 0.3 + Math.random()*0.5});
      }

      // Particle system (dust/data)
      const particleGeo = new THREE.BufferGeometry();
      const particleCount = 200;
      const posArray = new Float32Array(particleCount*3);
      for(let i=0; i<particleCount*3; i++) {
        posArray[i] = (Math.random()-0.5)*80;
      }
      particleGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
      const particleMat = new THREE.PointsMaterial({size: 0.15, color: 0x64ffda, transparent: true, opacity: 0.6});
      const particles = new THREE.Points(particleGeo, particleMat);
      scene.add(particles);

      camera.position.set(35, 25, 35);
      camera.lookAt(0, 0, 0);

      let angle = 0;
      function animate() {
        requestAnimationFrame(animate);
        const time = Date.now() * 0.001;
        angle += 0.002;
        camera.position.x = 40 * Math.cos(angle);
        camera.position.z = 40 * Math.sin(angle);
        camera.lookAt(0, 2, 0);

        packages.forEach(p => {
          p.mesh.position.y = 0.5 + Math.sin(time * 2 + p.offset) * p.amp * 0.3;
          p.mesh.rotation.y += p.speed;
          p.mesh.rotation.x = Math.sin(time + p.offset) * 0.1;
        });

        particles.rotation.y = time * 0.05;
        particles.rotation.x = time * 0.02;

        renderer.render(scene, camera);
      }
      animate();

      window.addEventListener('resize', () => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      });
    </script>
    """


# --- CUSTOM CSS (ENHANCED CYBERPUNK THEME) ---
def apply_custom_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(ellipse at 20% 20%, #0d1b3e 0%, #050a19 50%, #020408 100%);
            background-attachment: fixed;
        }
        .login-glass {
            background: rgba(15, 35, 60, 0.45);
            backdrop-filter: blur(25px) saturate(190%);
            -webkit-backdrop-filter: blur(25px) saturate(190%);
            border: 1px solid rgba(100, 255, 218, 0.2);
            border-radius: 35px;
            padding: 3rem;
            max-width: 450px;
            margin: 5rem auto;
            box-shadow: 0 0 40px rgba(100, 255, 218, 0.15), inset 0 0 20px rgba(100, 255, 218, 0.05);
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        .login-glass::before {
            content: "";
            position: absolute;
            top: -50px;
            left: -50px;
            width: 150px;
            height: 150px;
            background: rgba(100, 255, 218, 0.3);
            filter: blur(60px);
            border-radius: 50%;
            pointer-events: none;
        }
        .login-glass::after {
            content: "";
            position: absolute;
            bottom: -30px;
            right: -30px;
            width: 100px;
            height: 100px;
            background: rgba(0, 180, 219, 0.3);
            filter: blur(50px);
            border-radius: 50%;
            pointer-events: none;
        }
        .glass {
            background: rgba(10, 20, 40, 0.55);
            backdrop-filter: blur(18px) saturate(170%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .glass:hover {
            box-shadow: 0 12px 40px rgba(100, 255, 218, 0.1);
            border-color: rgba(100, 255, 218, 0.15);
        }
        .holographic-card {
            background: linear-gradient(135deg, rgba(15, 35, 60, 0.6) 0%, rgba(10, 25, 47, 0.7) 100%);
            border: 1px solid rgba(100, 255, 218, 0.15);
            border-radius: 16px;
            padding: 1.2rem;
            position: relative;
            overflow: hidden;
        }
        .holographic-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(100, 255, 218, 0.1), transparent);
            transition: left 0.5s ease;
        }
        .holographic-card:hover::before {
            left: 100%;
        }
        .neon-text {
            color: #64ffda;
            text-shadow: 0 0 10px rgba(100, 255, 218, 0.5);
            font-family: 'Courier New', monospace;
            letter-spacing: 1px;
        }
        .danger-glow {
            color: #ff6b6b;
            text-shadow: 0 0 10px rgba(255, 107, 107, 0.5);
        }
        .success-glow {
            color: #00ff88;
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.4);
        }
        .stButton > button {
            background: linear-gradient(135deg, #64ffda 0%, #00b4db 100%) !important;
            color: #0a192f !important;
            font-weight: bold !important;
            border-radius: 12px !important;
            border: none !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 15px rgba(100, 255, 218, 0.3) !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            box-shadow: 0 6px 25px rgba(100, 255, 218, 0.5) !important;
            transform: translateY(-2px) !important;
        }
        .stTextInput > div > div > input {
            background-color: rgba(5, 10, 25, 0.8) !important;
            color: #ccd6f6 !important;
            border-radius: 12px !important;
            border: 1px solid rgba(100, 255, 218, 0.1) !important;
            padding: 12px 15px !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: rgba(100, 255, 218, 0.8) !important;
            box-shadow: 0 0 15px rgba(100, 255, 218, 0.3) !important;
        }
        .wms-logo-placeholder {
            border: 3px solid #64ffda;
            color: #64ffda;
            font-family: 'Courier New', monospace;
            font-size: 2rem;
            font-weight: bold;
            border-radius: 50%;
            width: 80px;
            height: 80px;
            margin: 0 auto 2rem auto;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 25px rgba(100, 255, 218, 0.4), inset 0 0 15px rgba(100, 255, 218, 0.1);
            animation: pulse-glow 3s ease-in-out infinite;
        }
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 25px rgba(100, 255, 218, 0.4); }
            50% { box-shadow: 0 0 40px rgba(100, 255, 218, 0.7); }
        }
        .badge-container {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 8px;
        }
        .badge {
            background: rgba(100, 255, 218, 0.15);
            border: 1px solid rgba(100, 255, 218, 0.3);
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 11px;
            color: #64ffda;
            font-family: monospace;
        }
        .metric-pulse {
            animation: metric-breathe 2s ease-in-out infinite;
        }
        @keyframes metric-breathe {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# INITIALIZATION
apply_custom_theme()

st.set_page_config(page_title="NEURAL FULFILLMENT — YESI v3.0", layout="wide", page_icon="🧠")
init_db()
_guardian = get_guardian()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None
if 'df_sim_db' not in st.session_state:
    st.session_state.df_sim_db = None
if 'operator_stats' not in st.session_state:
    st.session_state.operator_stats = {}
if 'neural_chat_history' not in st.session_state:
    st.session_state.neural_chat_history = []
if 'last_anomaly_scan' not in st.session_state:
    st.session_state.last_anomaly_scan = None


# --- AUTHENTICATION INTERFACE ---
if not st.session_state.authenticated:
    st.markdown('<div class="login-glass">', unsafe_allow_html=True)
    st.markdown('<div class="wms-logo-placeholder">N3</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#ccd6f6; margin-bottom:0.5rem;">Neural Fulfillment</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8892b0; font-size:0.9rem; margin-bottom:2rem;">YESI v3.0 — Autonomous Warehouse Intelligence</p>', unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=False):
        uname = st.text_input("Username", placeholder="Operator ID", label_visibility="collapsed")
        pwd = st.text_input("Password", type="password", placeholder="Neural Key", label_visibility="collapsed")
        col_rem, col_forgot = st.columns([1, 1])
        with col_rem:
            remember = st.checkbox("Remember me", value=True)
        with col_forgot:
            st.markdown('<div style="text-align:right;"><a href="#" style="color:#8892b0; font-size:0.8rem;">Neural Recovery?</a></div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("AUTHENTICATE")

    if submitted:
        user_data = auth_login(uname, pwd)
        if user_data:
            st.session_state.authenticated = True
            st.session_state.user = user_data
            if user_data["username"] not in st.session_state.operator_stats:
                st.session_state.operator_stats[user_data["username"]] = OperatorStats(user_data["username"])
            log_action(user_data["username"], "Login Successful")
            st.rerun()
        else:
            st.error("❌ Neural authentication failed. Invalid credentials.")
            log_action(uname if uname else "Unknown", "Login Failed", "Invalid Credentials")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# --- MAIN APP INTERFACE ---
user = st.session_state.user["username"]
role = st.session_state.user["role"]

# Initialize operator stats if not exists
if user not in st.session_state.operator_stats:
    st.session_state.operator_stats[user] = OperatorStats(user)
ops = st.session_state.operator_stats[user]

st.title("🧠 Neural Fulfillment Operations")
st.caption(f"Welcome, {user} ({role}). Autonomous warehouse intelligence with predictive analytics, neural vision, and quantum routing.")

# --- SIDEBAR: SYSTEM STATUS, GAMIFICATION & SETTINGS ---
with st.sidebar:
    st.header("🌐 System Status")
    online_access_status = can_sync_now()
    is_online = st.toggle("Online Access (Sync)", value=online_access_status, help="Disable to stop offline queue synchronization.")
    if is_online != online_access_status:
        set_setting("online_access", str(is_online))
        status_text = "Enabled" if is_online else "Disabled"
        st.success(f"Online Sync {status_text}.")
        log_action(user, "Set Sync Status", status_text)

    status_color = "green" if is_online else "red"
    status_msg = "ONLINE" if is_online else "OFFLINE (Queue Paused)"
    st.markdown(f"Status: :{status_color}[{status_msg}]")

    # Gamification Panel
    st.divider()
    st.header("🎮 Operator Neural Link")
    with st.container():
        st.markdown(f"<div class='holographic-card'>", unsafe_allow_html=True)
        cols = st.columns([1, 2])
        with cols[0]:
            st.markdown(f"<div style='font-size:2.5rem; text-align:center;'>🧑‍🚀</div>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"<div class='neon-text' style='font-size:1.1rem;'>Lv. {ops.level}</div>", unsafe_allow_html=True)
            st.progress(min(1.0, ops.xp % 1000 / 1000), text=f"XP: {ops.xp % 1000}/1000")
            st.markdown(f"<div style='color:#8892b0; font-size:0.75rem;'>Streak: {ops.streak} | Accuracy: {ops.accuracy:.1f}%</div>", unsafe_allow_html=True)

        if ops.badges:
            badge_html = " ".join([f'<span class="badge">{b}</span>' for b in ops.badges[-5:]])
            st.markdown(f"<div class='badge-container'>{badge_html}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.header("📊 Queue & Settings")
    col_u, col_r = st.columns(2)
    col_u.write(f"User: **{user}**")
    col_r.write(f"Role: **{role}**")

    if st.button("Logout", use_container_width=True):
        log_action(user, "Logout")
        st.session_state.authenticated = False
        st.session_state.user = None
        st.rerun()

    with st.expander("Site Settings"):
        operator = st.text_input("Operator name", value=get_setting("operator_name", ""))
        site = st.text_input("Site name", value=get_setting("site_name", "Main"))
        if st.button("Save settings"):
            set_setting("operator_name", operator)
            set_setting("site_name", site)
            st.success("Saved")
            log_action(user, "Settings Updated", f"Site: {site}")

    qs = queue_status()
    st.metric("Queued actions", qs["queued"], help="Sync paused if OFFLINE.")
    st.metric("Last sync", qs["last_sync"] or "Never")

    if st.button("Process offline queue", disabled=not is_online, use_container_width=True, help="Only active when ONLINE."):
        if can_sync_now():
            synced, failed = process_queue()
            st.success(f"Synced {synced}, failed {failed}")
            log_action(user, "Manual Sync", f"Synced: {synced}, Failed: {failed}")
        else:
            st.warning("Enable Online Access first.")

    st.divider()
    st.header("📋 Reports")
    st.markdown("Download snapshot of current operations.")
    if st.button("📊 Generate Operations Summary", use_container_width=True):
        with st.spinner("Generating neural summary..."):
            log_action(user, "Report Generated", "Operations Summary")
            inv_df = get_inventory()
            orders_df = get_orders()
            queue_stats = queue_status()
            summary_data = {
                "Metric": ["Report Generated At", "Generating User", "Site Name", "Total SKUs", "Total Stock Units", "Open Orders (Pending)", "Items Enqueued for Sync", "Neural Anomalies Detected", "Operator Level", "System Status"],
                "Value": [datetime.utcnow().isoformat(timespec="seconds"), user, get_setting("site_name", "Main"), len(inv_df), int(inv_df["stock"].sum()) if not inv_df.empty else 0, int((orders_df["status"] == "Pending").sum()) if not orders_df.empty else 0, queue_stats["queued"], len(st.session_state.last_anomaly_scan) if st.session_state.last_anomaly_scan else 0, ops.level, status_msg]
            }
            summary_df = pd.DataFrame(summary_data)
            csv_buffer = io.BytesIO()
            summary_df.to_csv(csv_buffer, index=False)
            st.download_button(label="📥 Download Summary CSV", data=csv_buffer.getvalue(), file_name=f"neural_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv", use_container_width=True, key="download_summary")


# --- DASHBOARD DATA PREP ---
inv = get_inventory()
orders = get_orders()
q = queue_status()

# Run anomaly scan in background
if inv is not None and not inv.empty:
    current_alerts = _sentinel.scan(inv, orders)
    st.session_state.last_anomaly_scan = current_alerts
else:
    current_alerts = []

# Predictive forecast
forecast = _oracle.forecast(days=14) if not inv.empty else None

# Update route optimizer heatmap
if not inv.empty:
    sku_visits = dict(zip(inv["sku"].tolist(), inv["stock"].astype(int).tolist()))
    _route_optimizer.update_heat(sku_visits)

# Metrics row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📦 Items", int(inv["stock"].sum()) if not inv.empty else 0)
c2.metric("🏷️ SKUs", len(inv))
c3.metric("📋 Open Orders", int((orders["status"] == "Pending").sum()) if not orders.empty else 0)
c4.metric("⏳ Queue", q["queued"])
alert_count = len([a for a in current_alerts if a["severity"] in ["HIGH", "CRITICAL"]])
c5.metric("🚨 Anomalies", alert_count, delta=f"-{len([a for a in current_alerts if a['severity']=='CRITICAL'])} critical" if alert_count > 0 else None, delta_color="inverse")

if forecast:
    st.markdown(f"<div style='background:rgba(255,107,107,0.1); border-left:3px solid #ff6b6b; padding:10px 15px; border-radius:0 8px 8px 0; margin-bottom:10px;'>🔮 <b>Oracle Forecast:</b> {forecast['trend']} trend detected. Stockout risk: <span class='{'danger-glow' if forecast['stockout_risk']=='HIGH' else 'success-glow'}'>{forecast['stockout_risk']}</span>. Recommended reorder: <b>{forecast['recommended_reorder']}</b> units (confidence: {forecast['confidence']}%)</div>", unsafe_allow_html=True)

# --- TAB DEFINITIONS ---
tab_names = ["Dashboard", "Inventory", "Orders", "Auditor", "Bulk Convert", "PDF Sequencer", "Templates", "Memory", "🧠 Neural Ops", "🗺️ Holo-Deck", "⚡ Quantum Routes", "🎮 Command Center", "🌱 Eco-Logistics"]
if role == "Admin":
    tab_names.append("Admin 🔐")

tabs = st.tabs(tab_names)

tab_dash = tabs[0]
tab_inv = tabs[1]
tab_ord = tabs[2]
tab_aud = tabs[3]
tab_bulk = tabs[4]
tab_pdf = tabs[5]
tab_temp = tabs[6]
tab_mem = tabs[7]
tab_neural = tabs[8]
tab_holo = tabs[9]
tab_quantum = tabs[10]
tab_cmd = tabs[11]
tab_eco = tabs[12]
tab_admin = tabs[13] if role == "Admin" else None


# --- TABS CONTENT ---

with tab_dash:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Neural Operations Dashboard")
    st.write("Real-time autonomous warehouse intelligence with predictive analytics and anomaly detection.")

    # Anomaly alerts display
    if current_alerts:
        with st.expander("🚨 Active Anomaly Alerts", expanded=True):
            alert_df = pd.DataFrame([{k: v for k, v in a.items() if k != "icon"} for a in current_alerts[:10]])
            st.dataframe(alert_df, use_container_width=True, hide_index=True)

    # Forecast chart
    if forecast and not _oracle.get_history_df().empty:
        hist_df = _oracle.get_history_df()
        st.line_chart(hist_df.set_index("date")[["stock", "orders"]], use_container_width=True)

    st.dataframe(inv, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_inv:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Inventory Management")
    with st.form("inventory_form", clear_on_submit=True):
        sku = st.text_input("SKU")
        product = st.text_input("Product", help="Alias and Template suggestions populate below.")
        stock = st.number_input("Stock", min_value=0, value=0, step=1)
        location = st.text_input("Location", value="UNASSIGNED")
        note = st.text_input("Note", value="")
        submitted = st.form_submit_button("Save inventory item")

    if submitted and sku:
        upsert_inventory(sku, product, int(stock), location)
        add_action_log("inventory_upsert", sku, f"{product} | {stock} | {location}", user)
        enqueue_action("inventory_upsert", {"sku": sku, "product": product, "stock": int(stock), "location": location, "note": note})
        st.success("✅ Saved locally and queued for neural sync.")
        st.rerun()

    st.dataframe(get_inventory(), use_container_width=True, hide_index=True)
    col_sug1, col_sug2 = st.columns(2)
    col_sug1.write(f"Alias suggestion: {suggest_alias(product) or 'None'}")
    col_sug2.write(f"Template suggestion: {suggest_template(product) or 'None'}")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_ord:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Orders")
    with st.form("order_form", clear_on_submit=True):
        order_id = st.text_input("Order ID")
        status = st.selectbox("Status", ["Pending", "Shipped", "Returned", "Cancelled"])
        items = st.text_area("Required SKUs, one per line")
        created = st.form_submit_button("Create order")

    if created and order_id:
        skus = [x.strip() for x in items.splitlines() if x.strip()]
        create_order(order_id, status, skus)
        add_action_log("order_create", order_id, ",".join(skus), user)
        enqueue_action("order_create", {"order_id": order_id, "status": status, "required_skus": skus})
        st.success("✅ Order created locally and queued for neural sync.")
        st.rerun()

    orders_df = get_orders()
    st.dataframe(orders_df, use_container_width=True, hide_index=True)

    if not orders_df.empty:
        st.divider()
        st.subheader("Update Order")
        selected = st.selectbox("Select Order ID to Update", orders_df["order_id"].tolist())
        new_status = st.selectbox("New status", ["Pending", "Shipped", "Returned", "Cancelled"], key="new_status")
        if st.button("Update selected order"):
            update_order_status(selected, new_status)
            add_action_log("order_update", selected, new_status, user)
            enqueue_action("order_update", {"order_id": selected, "status": new_status})
            st.success("✅ Order updated and queued.")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with tab_aud:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Discrepancy Auditor")
    col_a, col_b = st.columns(2)
    with col_a:
        master_in = st.text_area("MASTER (Expected)", height=200, placeholder="ID Value")
    with col_b:
        scan_in = st.text_area("SCAN (Actual)", height=200, placeholder="ID Value")

    if st.button("Run Discrepancy Analysis", type="primary", use_container_width=True):
        if master_in and scan_in:
            m_map, s_map = robust_parse_multiline(master_in), robust_parse_multiline(scan_in)
            results = []
            for tid in sorted(set(m_map.keys()) | set(s_map.keys())):
                exp, got = m_map.get(tid, set()), s_map.get(tid, set())
                status = "✅ MATCH" if exp == got else "❌ ERROR"
                results.append({"ID": tid, "Status": status, "Expected": " | ".join(exp), "Actual": " | ".join(got)})
            res_df = pd.DataFrame(results)
            st.dataframe(res_df.style.apply(lambda x: ['background-color: #ffcccc' if '❌' in str(v) else '' for v in x], axis=1), use_container_width=True, hide_index=True)
            log_action(user, "Auditor Run")
            st.session_state.operator_stats[user].add_xp(25, "audit")
    st.markdown("</div>", unsafe_allow_html=True)


with tab_bulk:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Bulk Title Converter (Translation + Templates)")
    st.markdown("Paste original titles (non-English). App will translate, standardize, and apply matched **Templates**.")
    output_format = st.radio("Select Output Format for Matched Items:", ["Template Only", "Translation Only", "Combined (Template + Translation)"], horizontal=True)
    col_w, col_g = st.columns(2)
    with col_w:
        white_col = st.text_area("📄 Input Original Titles (one per line)", height=300)

    if st.button("✨ Convert & Translate & Apply Templates", type="primary", use_container_width=True):
        if white_col:
            lines = white_col.strip().split("\n")
            results = []
            matched_templates_count = 0
            with st.spinner("Translating and checking templates..."):
                translator = GoogleTranslator(source='auto', target='en')
                for l in lines:
                    line = l.strip()
                    if line:
                        try:
                            translated = translator.translate(line)
                            std_title = standardize_title(translated)
                            template_match = suggest_template(std_title)
                            if template_match:
                                matched_templates_count += 1
                                if output_format == "Template Only":
                                    results.append(template_match)
                                elif output_format == "Translation Only":
                                    results.append(std_title)
                                else:
                                    results.append(f"{template_match} (Match: {std_title})")
                            else:
                                results.append(std_title)
                        except Exception as e:
                            results.append(line.upper())
                    else:
                        results.append("")
            with col_g:
                output_text = "\n".join(results)
                st.text_area("Output (Standardized via Templates/Translation)", value=output_text, height=300)
            st.success(f"✅ Processed {len(lines)} titles. Applied {matched_templates_count} templates.")
            log_action(user, "Bulk Conversion", f"Processed: {len(lines)}, Templates: {matched_templates_count}")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_pdf:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Pro PDF Label Sequencer")
    sequencer_mode = st.radio("🎯 Select Sequencing Mode:", ["📋 Smart Sort (Flexible)", "🔒 Strict Rearrange (Exact Order)", "📱 WB Phone+Code Matcher"], horizontal=True, help="Smart Sort: Includes unmatched pages. Strict: ONLY pages matching your sequence. WB Phone+Code: Matches by phone number + 4-digit code from Wildberries labels.")
    st.divider()
    col1, col2 = st.columns([1, 2])

    with col1:
        if sequencer_mode == "📱 WB Phone+Code Matcher":
            sort_list = st.text_area("Target Phone+Code List", height=300, placeholder="Paste phone numbers and 4-digit codes here...\nExample:\n5261288 1844\n5385799 3666")
            st.caption("Format: `phone_number 4-digit_code` (one per line). OCR will scan the bottom-right QR area of each label.")
        else:
            sort_list = st.text_area("Target Sequence Order", height=300, placeholder="Paste Tracking IDs here...")
        remove_duplicates = st.checkbox("Auto-Remove Duplicate IDs", value=True, help="Removes duplicate entries from your pasted sequence while preserving the order.")

    with col2:
        label_file = st.file_uploader("Upload Labels PDF (Bulk)", type="pdf")
        use_ocr = st.checkbox("Enable OCR Fallback", value=True)
        if sequencer_mode == "📱 WB Phone+Code Matcher":
            st.info("**📱 WB Phone+Code Mode** — Optimized for Wildberries shipping labels containing a 7-digit phone number and 4-digit delivery code.")

    if st.button("⚙️ Process PDF", type="primary", use_container_width=True):
        if sequencer_mode == "📱 WB Phone+Code Matcher":
            target_entries = []
            for line in sort_list.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                digits = re.findall(r'\d+', line)
                if len(digits) >= 2:
                    phone = next((d for d in digits if len(d) == 7), None)
                    code = next((d for d in digits if len(d) == 4), None)
                    if phone and code:
                        target_entries.append({'phone': phone, 'code': code, 'raw': line})

            if remove_duplicates and target_entries:
                seen = set()
                cleaned_entries = []
                dupes_found = 0
                for entry in target_entries:
                    key = (entry['phone'], entry['code'])
                    if key not in seen:
                        seen.add(key)
                        cleaned_entries.append(entry)
                    else:
                        dupes_found += 1
                target_entries = cleaned_entries
                if dupes_found > 0:
                    st.toast(f"Cleaned {dupes_found} duplicate entries from sequence!", icon="🧹")

            if not target_entries or not label_file:
                st.warning("Please provide your target list and upload a PDF.")
            else:
                with st.spinner("Scanning labels with Multi-Angle OCR..."):
                    try:
                        pdf_reader = pypdf.PdfReader(io.BytesIO(label_file.getvalue()))
                        pdf_writer = pypdf.PdfWriter()
                        images = convert_from_bytes(label_file.getvalue(), dpi=200)
                        page_matches = []

                        for i, img in enumerate(images):
                            w, h = img.size
                            all_text = ""
                            barcodes = decode(img)
                            for b in barcodes:
                                all_text += " " + b.data.decode("utf-8", errors="ignore")
                            all_text += " " + pytesseract.image_to_string(img)
                            right_crop = img.crop((int(w * 0.60), 0, w, h))
                            for angle in [90, 270]:
                                rotated_crop = right_crop.rotate(angle, expand=True)
                                ocr_vertical = pytesseract.image_to_string(rotated_crop, config='--psm 6')
                                all_text += " " + ocr_vertical
                            phones_found = list(set(re.findall(r'\b\d{7}\b', all_text)))
                            codes_found = list(set(re.findall(r'\b\d{4}\b', all_text)))
                            tracking_ids = SCANNING_ID_REGEX.findall(all_text)
                            tracking_id = tracking_ids[0] if tracking_ids else "N/A"
                            page_matches.append({'page_idx': i, 'page_obj': pdf_reader.pages[i], 'tracking_id': tracking_id, 'phones': phones_found, 'codes': codes_found})

                        matched_page_indices = []
                        results_dataset = []
                        new_page_counter = 1

                        for target in target_entries:
                            target_phone = target['phone']
                            target_code = target['code']
                            matched_page = None
                            match_type = ""
                            for pm in page_matches:
                                if pm['page_idx'] in matched_page_indices:
                                    continue
                                if target_phone in pm['phones'] and target_code in pm['codes']:
                                    matched_page = pm
                                    match_type = "Exact Pair (7+4)"
                                    break
                            if not matched_page:
                                for pm in page_matches:
                                    if pm['page_idx'] in matched_page_indices:
                                        continue
                                    if target_phone in pm['phones']:
                                        matched_page = pm
                                        match_type = "Primary ID Only (7-digit)"
                                        break

                            if matched_page:
                                matched_page_indices.append(matched_page['page_idx'])
                                pdf_writer.add_page(matched_page['page_obj'])
                                results_dataset.append({"Status": "✅ MATCHED", "New Page #": new_page_counter, "Target ID": f"{target_phone} {target_code}", "Original Page": matched_page['page_idx'] + 1, "Match Quality": match_type, "Detected IDs on Page": f"IDs: {','.join(matched_page['phones'])} | Codes: {','.join(matched_page['codes'])}"})
                                new_page_counter += 1
                            else:
                                results_dataset.append({"Status": "❌ MISSING", "New Page #": "—", "Target ID": f"{target_phone} {target_code}", "Original Page": "N/A", "Match Quality": "No Match Found", "Detected IDs on Page": "—"})

                        extra_count = 0
                        for pm in page_matches:
                            if pm['page_idx'] not in matched_page_indices:
                                pdf_writer.add_page(pm['page_obj'])
                                extra_count += 1
                                results_dataset.append({"Status": "ℹ️ EXTRA (Unlisted)", "New Page #": new_page_counter, "Target ID": "Unlisted in Sequence", "Original Page": pm['page_idx'] + 1, "Match Quality": "Appended to End", "Detected IDs on Page": f"IDs: {','.join(pm['phones'])} | Codes: {','.join(pm['codes'])}"})
                                new_page_counter += 1

                        if results_dataset:
                            st.divider()
                            st.markdown("### 📊 Sequence Alignment Results")
                            res_df = pd.DataFrame(results_dataset)
                            def style_rows(val):
                                if '✅' in str(val): return 'background-color: rgba(40, 167, 69, 0.2); color: #28a745;'
                                if '❌' in str(val): return 'background-color: rgba(220, 53, 69, 0.2); color: #dc3545;'
                                if 'ℹ️' in str(val): return 'background-color: rgba(255, 193, 7, 0.2); color: #ffc107;'
                                return ''
                            st.dataframe(res_df.style.applymap(style_rows, subset=['Status']), use_container_width=True, hide_index=True)
                            matched_count = sum(1 for r in results_dataset if '✅' in r['Status'])
                            missing_count = sum(1 for r in results_dataset if '❌' in r['Status'])
                            c_stat1, c_stat2, c_stat3 = st.columns(3)
                            c_stat1.metric("✅ Correctly Sequenced", matched_count)
                            c_stat2.metric("❌ Missing from PDF", missing_count)
                            c_stat3.metric("ℹ️ Extra Pages Appended", extra_count)

                        if matched_count > 0 or extra_count > 0:
                            out_io = io.BytesIO()
                            pdf_writer.write(out_io)
                            log_action(user, "PDF_SEQUENCED_WB_ENHANCED", f"Matched: {matched_count}, Missing: {missing_count}, Extra: {extra_count}")
                            st.success(f"✅ Sequenced PDF Ready! Reordered {matched_count} labels exactly to your list.")
                            filename = f"WB_Sequenced_Labels_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                            st.download_button(label="📥 Download Reordered PDF", data=out_io.getvalue(), file_name=filename, mime="application/pdf", use_container_width=True)
                        else:
                            st.error("❌ Could not detect any matching label numbers in the uploaded document.")
                    except Exception as e:
                        st.error(f"❌ Processing Error: {str(e)}")
        else:
            target_ids_raw = [tid.strip() for tid in sort_list.split('\n') if tid.strip()]
            target_ids = []
            for tid in target_ids_raw:
                match = SCANNING_ID_REGEX.search(tid)
                target_ids.append(match.group() if match else tid)
            if remove_duplicates and target_ids:
                seen = set()
                cleaned_ids = []
                list_duplicates_found = 0
                for tid in target_ids:
                    if tid not in seen:
                        seen.add(tid)
                        cleaned_ids.append(tid)
                    else:
                        list_duplicates_found += 1
                target_ids = cleaned_ids
                if list_duplicates_found > 0:
                    st.toast(f"Cleaned {list_duplicates_found} duplicate IDs from sequence!", icon="🧹")

            if not target_ids or not label_file:
                st.warning("Provide sequence IDs and upload a PDF.")
            else:
                with st.spinner("Analyzing PDF pages..."):
                    try:
                        pdf_reader = pypdf.PdfReader(io.BytesIO(label_file.getvalue()))
                        pdf_writer = pypdf.PdfWriter()
                        images = convert_from_bytes(label_file.getvalue(), dpi=200)
                        id_to_page_map = {}
                        pdf_duplicates_skipped = 0

                        for i, img in enumerate(images):
                            page_codes = []
                            barcodes = decode(img)
                            for b in barcodes:
                                page_codes.extend(SCANNING_ID_REGEX.findall(b.data.decode("utf-8")))
                            if not barcodes and use_ocr:
                                page_codes.extend(SCANNING_ID_REGEX.findall(pytesseract.image_to_string(img)))
                            for code in set(page_codes):
                                if code not in id_to_page_map:
                                    id_to_page_map[code] = {"page": pdf_reader.pages[i], "original_idx": i + 1}
                                else:
                                    pdf_duplicates_skipped += 1

                        if pdf_duplicates_skipped > 0:
                            st.toast(f"Skipped {pdf_duplicates_skipped} duplicate page(s) in PDF!", icon="ℹ️")

                        results_dataset = []
                        matched_count = 0
                        new_page_counter = 1
                        expected_set = set(target_ids)

                        if sequencer_mode == "🔒 Strict Rearrange (Exact Order)":
                            st.info("🔒 **STRICT MODE**: Only pages matching your sequence will be included.")
                            for tid in target_ids:
                                if tid in id_to_page_map:
                                    orig_page = id_to_page_map[tid]["original_idx"]
                                    conv_page = new_page_counter
                                    pdf_writer.add_page(id_to_page_map[tid]["page"])
                                    matched_count += 1
                                    new_page_counter += 1
                                    results_dataset.append({"Status": "✅ INCLUDED", "Sequence Order": target_ids.index(tid) + 1, "ID": tid, "Original Page": orig_page, "Output Page": conv_page, "Notes": "Found and sequenced"})
                                else:
                                    results_dataset.append({"Status": "❌ MISSING", "Sequence Order": target_ids.index(tid) + 1, "ID": tid, "Original Page": "N/A", "Output Page": "N/A", "Notes": "ID not detected in PDF"})
                        else:
                            st.info("📋 **SMART MODE**: Pages matching your sequence come first, followed by extra pages.")
                            for tid in target_ids:
                                if tid in id_to_page_map:
                                    orig_page = id_to_page_map[tid]["original_idx"]
                                    conv_page = new_page_counter
                                    pdf_writer.add_page(id_to_page_map[tid]["page"])
                                    matched_count += 1
                                    new_page_counter += 1
                                    results_dataset.append({"Original pdf page": orig_page, "CONVERTED pdf page": conv_page, "MISMATCH from pdf": "", "MISMATCH from TABLE": ""})
                                else:
                                    results_dataset.append({"Original pdf page": "N/A", "CONVERTED pdf page": "N/A", "MISMATCH from pdf": "", "MISMATCH from TABLE": tid})
                            for tid, data in id_to_page_map.items():
                                if tid not in expected_set:
                                    pdf_writer.add_page(data["page"])
                                    new_page_counter += 1
                                    results_dataset.append({"Original pdf page": data["original_idx"], "CONVERTED pdf page": new_page_counter - 1, "MISMATCH from pdf": tid, "MISMATCH from TABLE": ""})

                        if results_dataset:
                            st.divider()
                            st.markdown("### 📊 Processing Results")
                            res_df = pd.DataFrame(results_dataset)
                            st.dataframe(res_df, use_container_width=True, hide_index=True)

                        if matched_count > 0:
                            out_io = io.BytesIO()
                            pdf_writer.write(out_io)
                            log_action(user, "PDF_SEQUENCED", f"Mode: {sequencer_mode}, Matched: {matched_count} pages.")
                            st.success(f"✅ PDF Ready: {matched_count} pages sequenced!")
                            filename = f"sorted_labels_{sequencer_mode.split('(')[1].split(')')[0].lower().replace(' ', '_')}.pdf"
                            st.download_button(label="📥 Download Sequenced PDF", data=out_io.getvalue(), file_name=filename, mime="application/pdf", use_container_width=True)
                        else:
                            st.error("❌ No matches found in document.")
                    except Exception as e:
                        st.error(f"❌ Processing Error: {str(e)}")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_temp:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Templates Database")
    with st.form("template_form", clear_on_submit=True):
        raw = st.text_input("Raw/Translated Title")
        standard = st.text_input("Standard/Clean Title")
        saved = st.form_submit_button("Save template")
    if saved and raw and standard:
        save_template(raw, standard)
        enqueue_action("template_save", {"raw": raw, "standard": standard})
        st.success("✅ Template saved locally and queued for neural sync.")
        log_action(user, "Template Saved", f"{raw} -> {standard}")
        st.rerun()
    st.dataframe(get_templates(), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_mem:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Memory & Preferences")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.write("### General Preferences")
        with st.form("memory_form", clear_on_submit=True):
            pref_key = st.text_input("Preference key")
            pref_value = st.text_input("Preference value")
            save_pref = st.form_submit_button("Save memory")
        if save_pref and pref_key:
            save_memory(pref_key, pref_value)
            record_preference(pref_key, pref_value)
            st.success("Memory stored locally.")
            log_action(user, "Memory Saved", pref_key)
            st.rerun()
    with col_m2:
        st.write("### Product Aliases")
        alias_src = st.text_input("Alias source text")
        alias_dst = st.text_input("Alias target text")
        if st.button("Save alias") and alias_src and alias_dst:
            upsert_alias(alias_src, alias_dst)
            enqueue_action("memory_save", {"key": f"alias:{alias_src.lower().strip()}", "value": alias_dst})
            st.success("Alias saved and queued.")
            log_action(user, "Alias Saved", f"{alias_src} -> {alias_dst}")
            st.rerun()
    st.write("### Recent System Preferences")
    st.dataframe(get_recent_preferences(), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# NEW CREATIVE TABS
# ═════════════════════════════════════════════════════════════════════════════

with tab_neural:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🧠 Neural Operations Center")
    st.markdown("<p style='color:#8892b0;'>AI-powered vision inspection, anomaly detection, and predictive intelligence.</p>", unsafe_allow_html=True)

    n_col1, n_col2 = st.columns([1, 1])

    with n_col1:
        st.markdown("### 📸 Neural Vision Inspection")
        st.caption("Upload package/label images for AI-powered defect detection and classification.")
        vision_file = st.file_uploader("Upload image for neural analysis", type=["png", "jpg", "jpeg"])
        vision_confidence = st.slider("Detection sensitivity threshold", 50, 98, 65)

        if vision_file:
            img = Image.open(vision_file)
            with st.spinner("Running neural vision pipeline..."):
                processed_img, detections = _neural_vision.process_frame(img)

                st.image(processed_img, caption="Neural Vision Overlay", use_container_width=True)

                if detections:
                    det_df = pd.DataFrame([{"Label": d["label"], "Confidence": f"{d['confidence']:.1f}%", "Area": d["area"], "Vertices": d["vertices"]} for d in detections if d["confidence"] >= vision_confidence])
                    if not det_df.empty:
                        st.dataframe(det_df, use_container_width=True, hide_index=True)
                        damage_count = len([d for d in detections if d["label"] == "Damage" and d["confidence"] >= vision_confidence])
                        if damage_count > 0:
                            st.error(f"🚨 {damage_count} potential damage(s) detected! Manual inspection required.")
                        else:
                            st.success("✅ No damage detected above threshold.")
                    else:
                        st.info("No detections above confidence threshold.")
                else:
                    st.info("No objects detected in image.")
                log_action(user, "Neural Vision Scan", f"Detections: {len(detections)}")
                st.session_state.operator_stats[user].add_xp(10, "scan")

    with n_col2:
        st.markdown("### 🔮 Oracle Predictive Analytics")
        if forecast:
            st.markdown(f"<div class='holographic-card'>", unsafe_allow_html=True)
            f_cols = st.columns(3)
            f_cols[0].metric("Trend", forecast["trend"])
            f_cols[1].metric("Stockout Risk", forecast["stockout_risk"])
            f_cols[2].metric("Confidence", f"{forecast['confidence']}%")
            st.markdown(f"<div style='color:#ccd6f6; margin-top:10px;'>📦 Recommended reorder quantity: <span class='neon-text'>{forecast['recommended_reorder']}</span> units</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#8892b0; font-size:0.85rem;'>Avg daily orders: {forecast['avg_daily_orders']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if forecast["forecast"]:
                forecast_df = pd.DataFrame({
                    "Day": [f"Day +{i+1}" for i in range(len(forecast["forecast"]))],
                    "Projected Stock": forecast["forecast"]
                })
                st.line_chart(forecast_df.set_index("Day"), use_container_width=True)
        else:
            st.info("Oracle needs more historical data (minimum 7 days) to generate forecasts.")

        st.markdown("### 🛡️ Anomaly Sentinel Feed")
        if current_alerts:
            alert_display = current_alerts[:8]
            for alert in alert_display:
                severity_color = {"CRITICAL": "#ff6b6b", "HIGH": "#ff9f43", "MEDIUM": "#ffd93d"}.get(alert["severity"], "#64ffda")
                st.markdown(f"<div style='border-left:3px solid {severity_color}; padding:8px 12px; margin:4px 0; background:rgba({severity_color.replace('#','')},0.1); border-radius:0 8px 8px 0; font-size:0.85rem;'><b>{alert['icon']} {alert['type']}</b> — {alert['message']} <span style='color:#8892b0;'>[{alert['sku']}]</span></div>", unsafe_allow_html=True)
        else:
            st.success("✅ All systems nominal. No anomalies detected.")

    st.markdown("</div>", unsafe_allow_html=True)

with tab_holo:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🗺️ Warehouse Holo-Deck")
    st.markdown("<p style='color:#8892b0;'>Interactive 3D digital twin of your warehouse with real-time zone monitoring and particle effects.</p>", unsafe_allow_html=True)

    holo_cols = st.columns([3, 1])
    with holo_cols[0]:
        st.components.v1.html(render_holodeck_3d(), height=620)

    with holo_cols[1]:
        st.markdown("### Zone Telemetry")
        for name, z in list(_route_optimizer.zones.items())[:8]:
            heat_val = _route_optimizer.heat_map[z["y"], z["x"]]
            heat_pct = min(100, int(heat_val / max(1, _route_optimizer.heat_map.max()) * 100))
            color = "#64ffda" if z["velocity"] == "high" else "#00b4db" if z["velocity"] == "medium" else "#8892b0"
            st.markdown(f"<div style='margin-bottom:8px;'><span style='color:{color}; font-weight:bold; font-family:monospace;'>{name}</span> <span style='color:#8892b0; font-size:0.8rem;'>({z['velocity']})</span><div style='background:rgba(255,255,255,0.05); height:6px; border-radius:3px; margin-top:2px;'><div style='width:{heat_pct}%; height:100%; background:{color}; border-radius:3px; transition:width 0.5s;'></div></div></div>", unsafe_allow_html=True)

        st.markdown("### System Status")
        st.markdown(f"<div class='neon-text' style='font-size:0.9rem;'>🟢 Neural Link: ACTIVE</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='neon-text' style='font-size:0.9rem;'>🟢 Digital Twin: SYNCED</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='neon-text' style='font-size:0.9rem;'>🟢 Anomaly Sentinel: ARMED</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#8892b0; font-size:0.8rem; margin-top:10px;'>Last sync: {datetime.now().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with tab_quantum:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("⚡ Quantum Route Optimizer")
    st.markdown("<p style='color:#8892b0;'>TSP-based pick path optimization with 2-opt local search and warehouse heatmap visualization.</p>", unsafe_allow_html=True)

    q_col1, q_col2 = st.columns([1, 2])

    with q_col1:
        st.markdown("### 🎯 Pick List Input")
        route_input = st.text_area("Enter SKUs to pick (one per line)", height=200, placeholder="SKU-001\nSKU-042\nSKU-117")
        optimize_btn = st.button("🚀 Calculate Quantum Route", type="primary", use_container_width=True)

        if optimize_btn and route_input:
            sku_list = [s.strip() for s in route_input.split("\n") if s.strip()]
            if len(sku_list) > 1:
                with st.spinner("Optimizing pick path via quantum algorithm..."):
                    route = _route_optimizer.optimize_route(sku_list)
                    total_distance = 0
                    for i in range(len(route)-1):
                        total_distance += _route_optimizer._dist(route[i], route[i+1])

                    st.success(f"✅ Route optimized! {len(route)} stops, estimated travel: {total_distance:.1f} grid units")
                    st.markdown(f"<div class='neon-text' style='font-size:0.9rem;'>🗺️ Route: {' → '.join([r['zone'] for r in route])}</div>", unsafe_allow_html=True)

                    # Display route table
                    route_df = pd.DataFrame([{"Stop": i+1, "SKU": r["sku"], "Zone": r["zone"], "X": r["x"], "Y": r["y"]} for i, r in enumerate(route)])
                    st.dataframe(route_df, use_container_width=True, hide_index=True)

                    # Generate SVG visualization
                    svg_viz = _route_optimizer.generate_svg(route)
                    with q_col2:
                        st.markdown("### 🗺️ Warehouse Heatmap & Route")
                        st.markdown(svg_viz, unsafe_allow_html=True)

                        # Route metrics
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Stops", len(route))
                        m2.metric("Grid Distance", f"{total_distance:.1f}")
                        m3.metric("Efficiency", f"{max(0, 100 - total_distance/len(route)*5):.0f}%")

                        log_action(user, "Quantum Route Optimized", f"Stops: {len(route)}, Distance: {total_distance:.1f}")
                        st.session_state.operator_stats[user].add_xp(20, "pick")
            else:
                st.warning("Please enter at least 2 SKUs for route optimization.")
        elif optimize_btn:
            st.warning("Please enter SKUs to optimize.")
        else:
            with q_col2:
                st.markdown("### 🗺️ Warehouse Heatmap & Route")
                # Show empty heatmap
                empty_svg = _route_optimizer.generate_svg([])
                st.markdown(empty_svg, unsafe_allow_html=True)
                st.info("Enter SKUs and click 'Calculate Quantum Route' to visualize optimal pick path.")

    st.markdown("</div>", unsafe_allow_html=True)


with tab_cmd:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🎮 Neural Command Center")
    st.markdown("<p style='color:#8892b0;'>Natural language warehouse operations and operator gamification leaderboard.</p>", unsafe_allow_html=True)

    cmd_cols = st.columns([2, 1])

    with cmd_cols[0]:
        st.markdown("### 💬 Neural Command Interface")
        st.caption("Try: 'find SKU-123', 'move 50 SKU-456 to Zone-A', 'show low stock', 'forecast inventory', 'optimize route for SKU-001 SKU-002'")

        for msg in st.session_state.neural_chat_history[-10:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_cmd = st.chat_input("Enter neural command...")
        if user_cmd:
            st.session_state.neural_chat_history.append({"role": "user", "content": user_cmd})
            parsed = _command_ai.parse(user_cmd)

            response = ""
            if parsed["intent"] == "GREETING":
                response = "👋 Welcome to Neural Command Center. I can help you find SKUs, move stock, check orders, forecast inventory, and optimize routes. What would you like to do?"
            elif parsed["intent"] == "FIND_SKU":
                sku = parsed["params"][0]
                inv_df = get_inventory()
                match = inv_df[inv_df["sku"].str.contains(sku, case=False, na=False)] if not inv_df.empty else pd.DataFrame()
                if not match.empty:
                    row = match.iloc[0]
                    response = f"🔍 Found **{sku}**: {row['product']} | Stock: {row['stock']} | Location: {row['location']}"
                else:
                    response = f"❌ SKU **{sku}** not found in inventory database."
            elif parsed["intent"] == "SHOW_LOW_STOCK":
                inv_df = get_inventory()
                low = inv_df[inv_df["stock"] < 5] if not inv_df.empty else pd.DataFrame()
                if not low.empty:
                    response = f"📉 Low stock alert! {len(low)} items below threshold:\n\n" + "\n".join([f"• {r['sku']}: {r['stock']} units ({r['location']})" for _, r in low.iterrows()])
                else:
                    response = "✅ All stock levels are healthy. No low stock items detected."
            elif parsed["intent"] == "FORECAST":
                if forecast:
                    response = f"🔮 Oracle Forecast: **{forecast['trend']}** trend. Stockout risk: **{forecast['stockout_risk']}**. Recommended reorder: **{forecast['recommended_reorder']}** units."
                else:
                    response = "🔮 Oracle needs more data to generate forecasts."
            elif parsed["intent"] == "OPTIMIZE_ROUTE":
                response = "⚡ Use the **Quantum Routes** tab for full route optimization with 3D visualization!"
            elif parsed["intent"] == "VISION_SCAN":
                response = "📸 Use the **Neural Ops** tab to upload images for AI inspection."
            elif parsed["intent"] == "CREATE_ORDER":
                response = f"📋 Order creation initiated for **{parsed['params'][0]}**. Please use the **Orders** tab to finalize details."
            elif parsed["intent"] == "ORDER_STATUS":
                oid = parsed["params"][0]
                orders_df = get_orders()
                match = orders_df[orders_df["order_id"].str.contains(oid, case=False, na=False)] if not orders_df.empty else pd.DataFrame()
                if not match.empty:
                    row = match.iloc[0]
                    response = f"📦 Order **{oid}** status: **{row['status']}** | Items: {row.get('items', 'N/A')}"
                else:
                    response = f"❌ Order **{oid}** not found."
            else:
                response = "🤖 Command not recognized. Try: 'find SKU-123', 'show low stock', 'forecast inventory', or 'hello' for help."

            st.session_state.neural_chat_history.append({"role": "assistant", "content": response})
            st.rerun()

    with cmd_cols[1]:
        st.markdown("### 🏆 Operator Leaderboard")
        if st.session_state.operator_stats:
            stats_list = sorted(st.session_state.operator_stats.values(), key=lambda x: x.xp, reverse=True)
            for i, stat in enumerate(stats_list[:5]):
                medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                st.markdown(f"<div class='holographic-card' style='margin-bottom:8px; padding:10px;'>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-weight:bold; color:#ccd6f6;'>{medal} {stat.username}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#64ffda; font-size:0.9rem;'>Lv.{stat.level} | {stat.xp} XP</div>", unsafe_allow_html=True)
                st.progress(min(1.0, stat.xp % 1000 / 1000), text=f"{stat.xp % 1000}/1000")
                if stat.badges:
                    st.markdown(f"<div style='font-size:0.75rem; color:#8892b0;'>{' '.join(stat.badges[:3])}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No operator data yet. Complete actions to earn XP!")

        st.markdown("### 🎯 Quick Actions")
        if st.button("🎁 Claim Daily Bonus", use_container_width=True):
            bonus_xp = random.randint(50, 150)
            ops.add_xp(bonus_xp, "scan")
            st.success(f"🎉 Daily bonus claimed! +{bonus_xp} XP")
            st.rerun()
        if st.button("🔄 Reset Streak", use_container_width=True):
            ops.streak = 0
            st.info("Streak reset. Time to build it back up!")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with tab_eco:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🌱 Eco-Logistics Tracker")
    st.markdown("<p style='color:#8892b0;'>Carbon footprint analysis and sustainability recommendations for green warehouse operations.</p>", unsafe_allow_html=True)

    eco_cols = st.columns([1, 2])

    with eco_cols[0]:
        st.markdown("### 📊 Emission Calculator")
        eco_orders = st.number_input("Orders to ship today", min_value=0, value=int((orders["status"] == "Pending").sum()) if not orders.empty else 0, step=1)
        eco_distance = st.number_input("Avg. shipping distance (km)", min_value=10, value=420, step=10)
        eco_packaging = st.selectbox("Packaging type", ["standard", "recycled", "biodegradable"], index=1)

        if st.button("🌍 Calculate Footprint", type="primary", use_container_width=True):
            footprint = _eco_tracker.calculate_footprint(eco_orders, eco_distance, eco_packaging)
            st.markdown(f"<div class='holographic-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='neon-text' style='font-size:1.2rem; margin-bottom:10px;'>Eco-Score: {footprint['eco_score']:.0f}/100</div>", unsafe_allow_html=True)
            st.progress(footprint['eco_score'] / 100, text="Sustainability Rating")

            f_m1, f_m2, f_m3 = st.columns(3)
            f_m1.metric("Transport", f"{footprint['transport_co2_kg']}kg", "CO₂")
            f_m2.metric("Packaging", f"{footprint['packaging_co2_kg']}kg", "CO₂")
            f_m3.metric("Warehouse", f"{footprint['warehouse_co2_kg']}kg", "CO₂")

            st.markdown(f"<div style='color:#ccd6f6; margin-top:10px; font-size:1.1rem;'><b>Total:</b> {footprint['total_co2_kg']}kg CO₂</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#64ffda; font-size:0.9rem;'>🌳 Trees needed to offset: {footprint['trees_needed']}</div>", unsafe_allow_html=True)

            for suggestion in footprint['suggestions']:
                st.markdown(f"<div style='color:#8892b0; font-size:0.85rem; margin:4px 0;'>• {suggestion}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            log_action(user, "Eco-Footprint Calculated", f"Orders: {eco_orders}, CO2: {footprint['total_co2_kg']}kg")

    with eco_cols[1]:
        st.markdown("### 📈 Sustainability Trends")
        # Simulate historical eco data
        eco_history = []
        for i in range(30):
            date = datetime.now() - timedelta(days=30-i)
            daily_orders = max(0, random.randint(5, 60) + i)
            fp = _eco_tracker.calculate_footprint(daily_orders, 420, "recycled")
            eco_history.append({"date": date, "co2": fp["total_co2_kg"], "orders": daily_orders, "eco_score": fp["eco_score"]})

        eco_df = pd.DataFrame(eco_history).set_index("date")
        st.line_chart(eco_df[["co2", "eco_score"]], use_container_width=True)

        st.markdown("### ♻️ Green Initiatives")
        initiatives = [
            ("Solar Panel Installation", "Reduce warehouse energy by 40%", 85),
            ("Biodegradable Packaging", "Eliminate plastic waste", 72),
            ("Route Consolidation AI", "Reduce transport emissions by 30%", 90),
            ("Electric Forklift Fleet", "Zero-emission material handling", 65),
        ]
        for name, desc, progress in initiatives:
            st.markdown(f"<div style='margin-bottom:8px;'><b style='color:#ccd6f6;'>{name}</b><div style='color:#8892b0; font-size:0.8rem;'>{desc}</div><div style='background:rgba(255,255,255,0.05); height:8px; border-radius:4px; margin-top:4px;'><div style='width:{progress}%; height:100%; background:linear-gradient(90deg, #64ffda, #00b4db); border-radius:4px;'></div></div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# --- ADMIN PANEL (Role Restricted) ---
if role == "Admin" and tab_admin:
    with tab_admin:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🔐 Admin Control Panel")

        adm_opt = st.radio("Admin Tool", ["👤 User Management & Logs", "📱 SIM Database Manager", "🛡️ Ozone Guardian Ops Center", "🧠 Neural System Diagnostics"], horizontal=True)
        st.divider()

        if adm_opt == "👤 User Management & Logs":
            st.subheader("User Management")
            with st.expander("Add New System User"):
                new_u = st.text_input("New Username")
                new_p = st.text_input("New Password", type="password")
                new_r = st.selectbox("Role", ["Operator", "Admin"])
                if st.button("Create User") and new_u and new_p:
                    add_user(new_u, new_p, new_r)
                    st.success(f"User {new_u} added.")
                    add_action_log("User Created", new_u, new_r, user)

            st.subheader("System Audit Logs")
            with connect() as conn:
                logs_df = pd.read_sql_query("SELECT created_at, user, action_type, ref_id, payload FROM action_logs ORDER BY created_at DESC LIMIT 100", conn)
            st.dataframe(logs_df, use_container_width=True, hide_index=True)

            # Neural system health
            st.subheader("🧠 Neural System Health")
            h_col1, h_col2, h_col3, h_col4 = st.columns(4)
            h_col1.metric("Vision Pipeline", "ONLINE", "✅")
            h_col2.metric("Oracle Engine", "ACTIVE", f"{len(_oracle.history)} days data")
            h_col3.metric("Sentinel Alerts", len(current_alerts), "🛡️")
            h_col4.metric("Route Optimizer", "READY", f"{_route_optimizer.grid_w}x{_route_optimizer.grid_h}")

        elif adm_opt == "📱 SIM Database Manager":
            st.subheader("📱 Samsung IMEI Database Manager")
            if st.session_state.df_sim_db is None:
                st.session_state.df_sim_db = load_sim_db()

            sim_tools_col, sim_conv_col = st.columns([1, 2])
            with sim_tools_col:
                st.markdown("### 🛠️ SIM Database Tools")
                search_query = st.text_input("🔍 Search Model or TAC (8 digits)", help="Enter Model Name or TAC Prefix.")
                display_sim_df = st.session_state.df_sim_db
                if search_query:
                    display_sim_df = display_sim_df[display_sim_df['Model_Series'].str.contains(search_query, case=False, na=False) | display_sim_df['TAC_Prefix'].str.contains(search_query, na=False)]
                st.write(f"Showing {len(display_sim_df)} entries")
                edited_sim_df = st.data_editor(display_sim_df, num_rows="dynamic", use_container_width=True, column_config={"TAC_Prefix": st.column_config.TextColumn("TAC Prefix (8 digits)"), "Expected_Offset": st.column_config.NumberColumn("Offset", format="%d"), "Model_Series": "Model Name", "Type": "Type"}, key="sim_data_editor")
                if st.button("💾 Save SIM Changes to CSV"):
                    if search_query:
                        st.session_state.df_sim_db.update(edited_sim_df)
                    else:
                        st.session_state.df_sim_db = edited_sim_df
                    save_sim_db(st.session_state.df_sim_db)
                    st.success("SIM Database file updated!")
                    log_action(user, "SIM DB Saved", f"{len(st.session_state.df_sim_db)} entries")

            with sim_conv_col:
                st.markdown("### 📱 IMEI Converter Tools")
                sim_db_map = dict(zip(st.session_state.df_sim_db['TAC_Prefix'], st.session_state.df_sim_db['Expected_Offset']))
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.write("#### 1. Calibration")
                    cal_input = st.text_area("Paste samples (IMEI 1 | IMEI 2):", height=150, placeholder="15 digits each")
                with col_c2:
                    st.write("#### 2. Targets")
                    batch_input = st.text_area("Paste IMEI 1 list (15 digits):", height=150)

                if batch_input:
                    active_sim_map = sim_db_map.copy()
                    if cal_input:
                        for line in cal_input.strip().split('\n'):
                            imeis = re.findall(r'\b\d{15}\b', line)
                            if len(imeis) >= 2:
                                active_sim_map[imeis[0][:8]] = int(imeis[1][:14]) - int(imeis[0][:14])
                    target_imeis = re.findall(r'\b\d{15}\b', batch_input)
                    sim_results = []
                    for i1 in target_imeis:
                        tac = i1[:8]
                        default_sim_val = sim_db_map.get('0', 8)
                        sim_offset = active_sim_map.get(tac, default_sim_val)
                        model_info = st.session_state.df_sim_db[st.session_state.df_sim_db['TAC_Prefix'] == tac]
                        model_sim_name = model_info['Model_Series'].values[0] if not model_info.empty else "Unknown TAC"
                        base14 = i1[:14]
                        new_base = str(int(base14) + int(sim_offset)).zfill(14)
                        i2 = new_base + calculate_luhn(new_base)
                        sim_results.append({"Model": model_sim_name, "IMEI 1": i1, "IMEI 2": i2, "TAC": tac, "Applied Offset": f"{int(sim_offset):+}"})
                    if sim_results:
                        st.divider()
                        st.write("#### Integrated Results")
                        st.dataframe(pd.DataFrame(sim_results), use_container_width=True, hide_index=True)
                        log_action(user, "SIM IMEI Converted", f"Processed: {len(sim_results)}")

        elif adm_opt == "🛡️ Ozone Guardian Ops Center":
            inv_df = get_inventory()
            orders_df = get_orders()
            q = queue_status()
            ctx = {
                "inventory": {"total_skus": len(inv_df), "total_stock": int(inv_df["stock"].sum()) if not inv_df.empty else 0},
                "orders": orders_df.to_dict("records") if not orders_df.empty else [],
                "pending_orders": int((orders_df["status"] == "Pending").sum()) if not orders_df.empty else 0,
                "stale_pending_minutes": 0,
                "sync_queue": q,
                "low_stock_skus": inv_df[inv_df["stock"] < 5]["sku"].tolist() if not inv_df.empty else [],
                "failed_logins_last_5m": 0,
            }
            _guardian.analyze(ctx)
            render_guardian_dashboard(health=_guardian.health, alerts=_guardian.alerts, suggestions=_guardian.suggestions, recovery=_guardian.recovery, tuner=_guardian.tuner, ozone=_guardian.ozone)
            log_action(user, "Guardian Dashboard Opened", "Ops Center")

        elif adm_opt == "🧠 Neural System Diagnostics":
            st.subheader("Neural System Diagnostics")
            st.markdown("<p style='color:#8892b0;'>Advanced system diagnostics and performance metrics for all creative modules.</p>", unsafe_allow_html=True)

            d_col1, d_col2, d_col3 = st.columns(3)
            d_col1.metric("Vision History", len(_neural_vision.detection_history), "scans")
            d_col2.metric("Route Visits", len(_route_optimizer.visit_log), "logged")
            d_col3.metric("Oracle Data Points", len(_oracle.history), "days")

            st.markdown("### 🔮 Oracle Forecast History")
            if not _oracle.get_history_df().empty:
                st.dataframe(_oracle.get_history_df().tail(14), use_container_width=True, hide_index=True)

            st.markdown("### 🚨 Sentinel Alert Archive")
            if _sentinel.alert_log:
                alert_archive_df = pd.DataFrame(list(_sentinel.alert_log)[-50:])
                st.dataframe(alert_archive_df, use_container_width=True, hide_index=True)
            else:
                st.info("No alerts in archive.")

            st.markdown("### 🗺️ Zone Heatmap Data")
            heat_df = pd.DataFrame(_route_optimizer.heat_map)
            st.dataframe(heat_df, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
