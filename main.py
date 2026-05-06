"""
Valorant Agent Pick Rate Dashboard
Phân tích tỷ lệ chọn tướng theo map — Flask + Pandas + K-Means Clustering
"""

import os
import logging
import math
from datetime import datetime
from functools import lru_cache

import pandas as pd
from flask import Flask, render_template, request

# Cấu hình logging
LOG_DIR = "logs"

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt); ch.setLevel(logging.DEBUG)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(fh); logger.addHandler(ch)

setup_logging()

# Khởi tạo Flask
app = Flask(__name__)
DATA_FILE = "agents_pick_rates.csv"

# Bảng phân loại role của agent
AGENT_ROLES = {
    "Brimstone": "Controller", "Viper": "Controller", "Omen": "Controller",
    "Astra": "Controller", "Harbor": "Controller", "Clove": "Controller", "Miks": "Controller", 
    "Jett": "Duelist", "Phoenix": "Duelist", "Reyna": "Duelist",
    "Raze": "Duelist", "Yoru": "Duelist", "Neon": "Duelist",
    "Iso": "Duelist", "Waylay": "Duelist",
    "Sova": "Initiator", "Breach": "Initiator", "Skye": "Initiator",
    "Kayo": "Initiator", "Fade": "Initiator", "Gekko": "Initiator", "Tejo": "Initiator",
    "Sage": "Sentinel", "Cypher": "Sentinel", "Killjoy": "Sentinel",
    "Chamber": "Sentinel", "Deadlock": "Sentinel", "Vyse": "Sentinel", "Veto": "Sentinel",
}

TIER_COLORS = {"S": "#ff4655", "A": "#ff6b77", "B": "#ff9099", "C": "#ffb5bb"}
AVAILABLE_ROLES = ["Duelist", "Controller", "Initiator", "Sentinel"]


# Đọc & cache dữ liệu CSV
@lru_cache(maxsize=1)
def load_grouped_data():
    """Đọc CSV, chuyển đổi kiểu dữ liệu, nhóm theo (Map, Agent)."""
    if not os.path.exists(DATA_FILE):
        logging.error("Không tìm thấy file: %s", DATA_FILE)
        return pd.DataFrame(columns=["Map", "Agent", "Pick Rate"])

    df = pd.read_csv(DATA_FILE).copy()
    logging.info("Đọc %d dòng từ %s", len(df), DATA_FILE)

    # Chuyển "15%" → 15.0
    df["Pick Rate"] = pd.to_numeric(
        df["Pick Rate"].astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    ).fillna(0.0)
    df["Agent"] = df["Agent"].astype(str).str.title()

    # Xác thực: loại NaN, giới hạn 0-100
    df = df.dropna(subset=["Map", "Agent"])
    df["Pick Rate"] = df["Pick Rate"].clip(0.0, 100.0)

    # Nhóm & sắp xếp
    grouped = (
        df.groupby(["Map", "Agent"], as_index=False)["Pick Rate"]
        .mean().round(1)
        .sort_values(["Map", "Pick Rate", "Agent"], ascending=[True, False, True])
    )
    logging.info("Đã nhóm: %d dòng, %d map, %d agent", len(grouped), grouped["Map"].nunique(), grouped["Agent"].nunique())
    return grouped


# Thống kê nâng cao
def compute_statistics(agents_list):
    """Tính mean, median, std_dev, min, max, range."""
    if not agents_list:
        return {"mean": 0, "median": 0, "std_dev": 0, "min": 0, "max": 0, "range": 0}
    rates = [a["pick_rate"] for a in agents_list]
    n = len(rates)
    mean_val = sum(rates) / n
    s = sorted(rates)
    median_val = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    std_dev = math.sqrt(sum((r - mean_val) ** 2 for r in rates) / n)
    return {
        "mean": round(mean_val, 1), "median": round(median_val, 1),
        "std_dev": round(std_dev, 1), "min": round(min(rates), 1),
        "max": round(max(rates), 1), "range": round(max(rates) - min(rates), 1),
    }


def get_role_for_agent(name):
    return AGENT_ROLES.get(name, "Unknown")

def get_tier_color(tier):
    return TIER_COLORS.get(tier, "#cccccc")

def get_maps():
    """Lấy danh sách map (bỏ All Maps)."""
    return [m for m in load_grouped_data()["Map"].unique().tolist() if m != "All Maps"]


# K-Means Clustering — phân tier S/A/B/C
def kmeans_cluster_tiers(pick_rates, k=4, max_iter=100):
    """Phân cụm pick_rates thành k nhóm bằng K-Means thủ công."""
    labels = ["S", "A", "B", "C"]

    if len(pick_rates) <= k:
        return [labels[i] if i < k else labels[-1] for i in range(len(pick_rates))]

    lo, hi = min(pick_rates), max(pick_rates)
    if hi == lo:
        return [labels[0]] * len(pick_rates)

    # Khởi tạo centroids phân đều
    centroids = [hi - i * (hi - lo) / (k - 1) for i in range(k)]
    assignments = [0] * len(pick_rates)

    for _ in range(max_iter):
        # Gán mỗi điểm vào cụm gần nhất
        new_assign = [min(range(k), key=lambda j: abs(r - centroids[j])) for r in pick_rates]
        if new_assign == assignments:
            break
        assignments = new_assign
        # Cập nhật centroids
        for j in range(k):
            members = [pick_rates[i] for i in range(len(pick_rates)) if assignments[i] == j]
            if members:
                centroids[j] = sum(members) / len(members)

    # Sắp xếp cụm: centroid cao nhất = S
    order = sorted(range(k), key=lambda j: centroids[j], reverse=True)
    c2t = {idx: labels[r] if r < len(labels) else labels[-1] for r, idx in enumerate(order)}

    tiers = [c2t[a] for a in assignments]
    logging.debug("K-Means centroids: %s | S=%d A=%d B=%d C=%d",
                  [round(c, 1) for c in centroids],
                  tiers.count("S"), tiers.count("A"), tiers.count("B"), tiers.count("C"))
    return tiers


# Xây dựng chi tiết 1 map
def build_map_detail(map_name):
    """Tạo dữ liệu chi tiết: agents, tier (K-Means), stats, role."""
    filtered = load_grouped_data().loc[lambda d: d["Map"] == map_name].reset_index(drop=True)
    if filtered.empty:
        return None

    pick_rates = [float(r["Pick Rate"]) for _, r in filtered.iterrows()]
    tiers = kmeans_cluster_tiers(pick_rates, k=4)

    agents = []
    for idx, row in filtered.iterrows():
        t = tiers[idx]
        agents.append({
            "rank": idx + 1, "agent": row["Agent"],
            "pick_rate": float(row["Pick Rate"]), "tier": t,
            "tier_color": get_tier_color(t), "role": get_role_for_agent(row["Agent"]),
        })

    top = agents[0]
    role_counts = {}
    for a in agents:
        role_counts[a["role"]] = role_counts.get(a["role"], 0) + 1

    logging.info("Map %s: %d agents, best=%s (%.1f%%)", map_name, len(agents), top["agent"], top["pick_rate"])
    return {
        "map": map_name, "best_agent": top["agent"], "best_rate": top["pick_rate"],
        "agent_count": len(agents),
        "average_pick_rate": round(sum(a["pick_rate"] for a in agents) / len(agents), 1),
        "top_three": [a["agent"] for a in agents[:3]],
        "agents": agents, "statistics": compute_statistics(agents),
        "role_distribution": role_counts,
    }


# So sánh chéo giữa các map
def build_cross_map_analysis():
    """Đếm số map mỗi agent xuất hiện trong top 3."""
    maps = get_maps()
    counter = {}
    for m in maps:
        d = build_map_detail(m)
        if d:
            for name in d["top_three"]:
                counter[name] = counter.get(name, 0) + 1
    return [{"agent": a, "top3_count": c, "total_maps": len(maps), "role": get_role_for_agent(a)}
            for a, c in sorted(counter.items(), key=lambda x: x[1], reverse=True)]


# Tổng quan tất cả map
def build_overview():
    """Tạo tổng quan nhanh cho tất cả map."""
    return [
        {"map": d["map"], "best_agent": d["best_agent"],
         "best_rate": d["best_rate"], "agent_count": d["agent_count"]}
        for d in (build_map_detail(m) for m in get_maps()) if d
    ]


# Lọc theo vai trò (role)
def filter_detail_by_role(detail, role):
    """Lọc agents theo role, tính lại rank và stats."""
    if not detail or not role:
        return detail
    fa = [a for a in detail["agents"] if a["role"] == role]
    if not fa:
        return detail
    for i, a in enumerate(fa):
        a["rank"] = i + 1
    top = fa[0]
    return {
        "map": detail["map"], "best_agent": top["agent"], "best_rate": top["pick_rate"],
        "agent_count": len(fa),
        "average_pick_rate": round(sum(a["pick_rate"] for a in fa) / len(fa), 1),
        "top_three": [a["agent"] for a in fa[:3]],
        "agents": fa, "statistics": compute_statistics(fa),
        "role_distribution": {role: len(fa)},
    }


# Routes
@app.route("/")
def index():
    """Trang chủ — hiển thị dashboard."""
    maps = get_maps()
    selected_map = request.args.get("map") or (maps[0] if maps else None)
    selected_role = request.args.get("role", "")
    detail = build_map_detail(selected_map) if selected_map else None

    if detail and selected_role in AVAILABLE_ROLES:
        detail = filter_detail_by_role(detail, selected_role)

    return render_template("index.html", maps=maps, selected_map=selected_map,
                           selected_role=selected_role, available_roles=AVAILABLE_ROLES,
                           initial_detail=detail, overview=build_overview())

@app.errorhandler(404)
def handle_not_found(error):
    return render_template("index.html", maps=get_maps(), selected_map=None,
                           selected_role="", available_roles=AVAILABLE_ROLES,
                           initial_detail=None, overview=build_overview()), 404


if __name__ == "__main__":
    logging.info("=" * 50)
    logging.info("KHỞI ĐỘNG VALORANT PICK RATE DASHBOARD")
    logging.info("Thời gian: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("Port: 5001 | Debug: ON")
    logging.info("=" * 50)
    app.run(debug=True, port=5000)