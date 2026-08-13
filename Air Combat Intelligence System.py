"""
╔══════════════════════════════════════════════════════════════════╗
║          AIR COMBAT INTELLIGENCE SYSTEM  v1.0                  ║
║          Aerial Threat Detection & Tracking Platform            ║
║          Developer: KHIZER AHMED                                 ║
╚══════════════════════════════════════════════════════════════════╝

"""

import subprocess, sys, time, datetime, math, logging, csv
from collections import defaultdict, deque
from pathlib import Path
from dataclasses import dataclass, field

def install_dependencies():
    pkgs = ["ultralytics", "opencv-python", "numpy", "scipy",
            "matplotlib", "tqdm", "colorama", "pandas", "Pillow"]
    print("\033[96m[INIT] Checking dependencies...\033[0m")
    for p in pkgs:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", p, "-q"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    print("\033[92m[INIT] Ready.\033[0m")

install_dependencies()

import cv2, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("COMBAT")

BANNER = f"""
{Fore.RED}╔══════════════════════════════════════════════════════════════════╗
║       AIR COMBAT INTELLIGENCE SYSTEM   v1.0                   ║
║       Aerial Threat Detection & Tracking  |  tubakhxn         ║
╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}"""

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Colour palette (BGR) ─────────────────────────────────────────────────────
COL_FIGHTER  = (40,  40, 255)   # neon red
COL_MILAIR   = (255, 255, 40)   # neon cyan
COL_HELI     = (60,  255, 60)   # neon green
COL_DRONE    = (40,  255, 255)  # neon yellow
COL_MISSILE  = (0,   140, 255)  # neon orange
COL_VEHICLE  = (220, 60,  220)  # purple
COL_PERSON   = (255, 120, 40)   # blue-ish

PANEL_BG   = (14, 10, 8)
ACCENT     = (60, 220, 255)
TEXT_DIM   = (140, 140, 160)
TEXT_W     = (255, 255, 255)
THREAT_LOW = (80, 220, 80)
THREAT_MED = (0, 165, 255)
THREAT_HI  = (40, 40, 255)

CLASS_COLORS = {
    "fighter jet":        COL_FIGHTER,
    "military aircraft":  COL_MILAIR,
    "helicopter":         COL_HELI,
    "drone":              COL_DRONE,
    "vehicle":            COL_VEHICLE,
    "person":             COL_PERSON,
}

# COCO class ids relevant to airspace footage
AIRCRAFT_CLS = {4: "aircraft"}                 # airplane
GROUND_CLS   = {2: "vehicle", 5: "vehicle", 7: "vehicle", 0: "person"}
ALL_CLS_IDS  = list(AIRCRAFT_CLS.keys()) + list(GROUND_CLS.keys())

# ── IOU + centroid tracker (persistent IDs) ─────────────────────────────────
def _iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    area_a = max(1,(ax2-ax1)*(ay2-ay1)); area_b = max(1,(bx2-bx1)*(by2-by1))
    return inter/float(area_a+area_b-inter)

@dataclass
class Track:
    tid: int
    bbox: tuple
    base_cls: str               # "aircraft" / "vehicle" / "person"
    conf: float
    history: deque = field(default_factory=lambda: deque(maxlen=24))
    age: int = 0
    frames_seen: int = 0

    def centroid(self):
        x1,y1,x2,y2 = self.bbox
        return ((x1+x2)/2, (y1+y2)/2)

class Tracker:
    def __init__(self, max_age=15, iou_thresh=0.2):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.iou_thresh = iou_thresh

    def update(self, detections):
        """detections: list of (bbox, base_cls, conf)"""
        unmatched = list(range(len(detections)))
        for tid, tr in list(self.tracks.items()):
            best_iou, best_j = 0.0, -1
            for j in unmatched:
                iou = _iou(tr.bbox, detections[j][0])
                if iou > best_iou: best_iou, best_j = iou, j
            if best_iou >= self.iou_thresh:
                bbox, cls, conf = detections[best_j]
                tr.bbox = bbox; tr.conf = conf; tr.age = 0
                tr.frames_seen += 1
                tr.history.append(tr.centroid())
                unmatched.remove(best_j)
            else:
                tr.age += 1
        for j in unmatched:
            bbox, cls, conf = detections[j]
            t = Track(tid=self.next_id, bbox=bbox, base_cls=cls, conf=conf)
            t.history.append(t.centroid())
            self.tracks[self.next_id] = t
            self.next_id += 1
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].age > self.max_age:
                del self.tracks[tid]
        return list(self.tracks.values())

# ── Motion / classification helpers ─────────────────────────────────────────
def velocity_px(track: Track):
    """Pixel velocity vector per frame from last few history points."""
    if len(track.history) < 2:
        return (0.0, 0.0)
    (x0,y0) = track.history[-min(6,len(track.history))]
    (x1,y1) = track.history[-1]
    n = min(6,len(track.history)) - 1
    if n <= 0: return (0.0,0.0)
    return ((x1-x0)/n, (y1-y0)/n)

def classify_aircraft(track: Track, frame_w, frame_h):
    """Heuristic aircraft-type assignment -- see module docstring."""
    x1,y1,x2,y2 = track.bbox
    w, h = x2-x1, y2-y1
    area_frac = (w*h) / max(1, frame_w*frame_h)
    aspect = w / max(1, h)
    vx, vy = velocity_px(track)
    speed = math.hypot(vx, vy)

    if area_frac < 0.0025 and speed > 3.0:
        return "drone"
    if aspect > 2.2 and area_frac > 0.004:
        return "fighter jet"
    if speed < 0.6 and area_frac > 0.002:
        return "helicopter"
    return "military aircraft"

def threat_score(track: Track, all_tracks, frame_w, frame_h):
    """Composite 0-100 score from speed, screen-centre proximity and
    convergence with other tracks' projected paths (real measured motion,
    simple physics -- not an actual radar/IFF threat model)."""
    cx, cy = track.centroid()
    vx, vy = velocity_px(track)
    speed = math.hypot(vx, vy)

    center_dx = abs(cx - frame_w/2) / (frame_w/2)
    center_dy = abs(cy - frame_h/2) / (frame_h/2)
    proximity = 1.0 - min(1.0, math.hypot(center_dx, center_dy))

    convergence = 0.0
    for other in all_tracks:
        if other.tid == track.tid: continue
        ox, oy = other.centroid()
        ovx, ovy = velocity_px(other)
        # project both 15 frames ahead, see how close they get
        fx1, fy1 = cx + vx*15, cy + vy*15
        fx2, fy2 = ox + ovx*15, oy + ovy*15
        d_now = math.hypot(cx-ox, cy-oy)
        d_future = math.hypot(fx1-fx2, fy1-fy2)
        if d_now > 1 and d_future < d_now:
            convergence = max(convergence, 1.0 - d_future/max(d_now,1))

    score = (min(speed/12.0, 1.0)*40) + (proximity*30) + (convergence*30)
    return max(0.0, min(100.0, score))

def threat_bucket(score):
    if score >= 66: return "HIGH", THREAT_HI
    if score >= 33: return "MEDIUM", THREAT_MED
    return "LOW", THREAT_LOW

# ── Drawing helpers ──────────────────────────────────────────────────────────
def alpha_rect(frame, x1,y1,x2,y2, color, alpha=0.5):
    x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(frame.shape[1]-1,x2), min(frame.shape[0]-1,y2)
    if x2<=x1 or y2<=y1: return
    ov = frame.copy()
    cv2.rectangle(ov,(x1,y1),(x2,y2),color,-1)
    cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)

def glow_box(frame, x1,y1,x2,y2, color, fnum):
    alpha_rect(frame, x1,y1,x2,y2, color, 0.18)
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    pulse = 6 + int(3*abs(math.sin(fnum*0.15)))
    for px,py,sx,sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame,(px,py),(px+sx*pulse,py),color,2)
        cv2.line(frame,(px,py),(px,py+sy*pulse),color,2)

def radar_ring(frame, cx, cy, fnum, color):
    r = 18 + (fnum*3) % 22
    alpha = max(0.0, 1.0 - r/40.0)
    if alpha <= 0: return
    ov = frame.copy()
    cv2.circle(ov, (cx,cy), r, color, 1)
    cv2.addWeighted(ov, alpha*0.6, frame, 1-alpha*0.6, 0, frame)

def motion_trail(frame, history, color):
    pts = list(history)
    n = len(pts)
    for i in range(1, n):
        a = i/n
        p1 = tuple(map(int, pts[i-1])); p2 = tuple(map(int, pts[i]))
        ov = frame.copy()
        cv2.line(ov, p1, p2, color, 2, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.25+0.4*a, frame, 1-(0.25+0.4*a), 0, frame)

def direction_vector(frame, cx, cy, vx, vy, color):
    mag = math.hypot(vx,vy)
    if mag < 0.3: return
    scale = 8
    ex, ey = int(cx+vx*scale), int(cy+vy*scale)
    cv2.arrowedLine(frame, (int(cx),int(cy)), (ex,ey), color, 2, tipLength=0.35)

def future_path(frame, cx, cy, vx, vy, color, steps=5, gap=10):
    for i in range(1, steps+1):
        fx, fy = int(cx+vx*gap*i), int(cy+vy*gap*i)
        if i % 2 == 0:
            cv2.circle(frame, (fx,fy), 3, color, -1)
        if i == steps:
            cv2.circle(frame, (fx,fy), 9, color, 1)
            cv2.putText(frame,"PRED",(fx+10,fy),FONT,0.32,color,1,cv2.LINE_AA)

def pill_label(frame, text, x, y, fg, bg=(8,8,14), fs=0.36, th=1):
    (tw,th_),_ = cv2.getTextSize(text,FONT,fs,th)
    p=3
    cv2.rectangle(frame,(x-p,y-th_-p),(x+tw+p,y+p),bg,-1)
    cv2.rectangle(frame,(x-p,y-th_-p),(x+tw+p,y+p),fg,1)
    cv2.putText(frame,text,(x,y),FONT,fs,fg,th,cv2.LINE_AA)

def draw_target(frame, track: Track, label: str, color, fnum, frame_w, frame_h, score=None):
    x1,y1,x2,y2 = map(int, track.bbox)
    cx,cy = track.centroid()
    glow_box(frame, x1,y1,x2,y2, color, fnum)
    motion_trail(frame, track.history, color)
    radar_ring(frame, int(cx), int(cy), fnum, color)
    vx,vy = velocity_px(track)
    direction_vector(frame, cx, cy, vx, vy, color)
    future_path(frame, cx, cy, vx, vy, color)

    speed_kmh = math.hypot(vx,vy) * 18   # arbitrary px->kmh scale for display
    heading = (math.degrees(math.atan2(vy, vx)) + 360) % 360
    lines = [f"TARGET #{track.tid}", label.upper(), f"{speed_kmh:0.0f} KM/H  HDG {heading:0.0f}°"]
    if score is not None:
        bucket, bcol = threat_bucket(score)
        lines.append(f"THREAT: {bucket}")
    ty = y1 - 8 - 14*(len(lines)-1)
    for i, ln in enumerate(lines):
        col = bcol if (score is not None and i == len(lines)-1) else color
        pill_label(frame, ln, x1, ty + i*14, col, fs=0.32)

def draw_minimap(frame, tracks, fw, fh, heatmap_pts):
    mw, mh = 220, 150
    x, y = 10, frame.shape[0]-mh-10 if False else 10
    x, y = 10, frame.shape[0]-mh-12
    alpha_rect(frame, x, y, x+mw, y+mh, PANEL_BG, 0.85)
    cv2.rectangle(frame,(x,y),(x+mw,y+mh),ACCENT,1)
    cv2.putText(frame,"AIRSPACE DIGITAL TWIN",(x+5,y+14),FONT,0.32,ACCENT,1,cv2.LINE_AA)
    for (px,py) in list(heatmap_pts)[-300:]:
        mx = x+8+int(px/fw*(mw-16)); my = y+20+int(py/fh*(mh-28))
        cv2.circle(frame,(mx,my),1,(70,70,90),-1)
    for tr in tracks:
        cx,cy = tr.centroid()
        mx = x+8+int(cx/fw*(mw-16)); my = y+20+int(cy/fh*(mh-28))
        col = CLASS_COLORS.get(getattr(tr,'_disp_label',''), ACCENT)
        cv2.circle(frame,(mx,my),3,col,-1)

def draw_dashboard(frame, stats, fps, fnum):
    fh,fw = frame.shape[:2]
    pw, ph = 300, 210
    x, y = fw-pw-10, 10
    alpha_rect(frame, x,y,x+pw,y+ph, PANEL_BG, 0.90)
    cv2.rectangle(frame,(x,y),(x+pw,y+ph),ACCENT,1)
    cv2.line(frame,(x+1,y+22),(x+pw-1,y+22),ACCENT,1)
    cv2.putText(frame,"TACTICAL DASHBOARD",(x+5,y+15),FONT,0.36,ACCENT,1,cv2.LINE_AA)

    bucket, bcol = threat_bucket(stats["max_threat"])
    rows = [
        ("FPS",            f"{fps:>5.1f}",                 TEXT_DIM),
        ("FRAME",          f"{fnum:>6d}",                  TEXT_DIM),
        ("ACTIVE AIRCRAFT",f"{stats['aircraft']:>4d}",     ACCENT),
        ("ACTIVE DRONES",  f"{stats['drones']:>4d}",       COL_DRONE),
        ("AVG SPEED",      f"{stats['avg_speed']:>5.0f} KM/H", TEXT_DIM),
        ("HIGHEST THREAT", f"#{stats['top_target']}",      bcol),
        ("THREAT LEVEL",   bucket,                          bcol),
        ("AIRSPACE DENSITY", f"{stats['density']:>5.1f}%", TEXT_DIM),
        ("COLLISION RISKS",f"{stats['collisions']:>4d}",   THREAT_HI if stats['collisions'] else TEXT_DIM),
        ("SYSTEM STATUS",  "OPERATIONAL",                   THREAT_LOW),
    ]
    for i,(k,v,col) in enumerate(rows):
        ky = y+34+i*17
        cv2.putText(frame,f" {k:<17}",(x+4,ky),FONT,0.30,TEXT_DIM,1,cv2.LINE_AA)
        cv2.putText(frame,str(v),(x+195,ky),FONT,0.30,col,1,cv2.LINE_AA)

def draw_corners(frame):
    fh,fw = frame.shape[:2]; s=34; t=2
    for cx,cy,sx,sy in [(0,0,1,1),(fw-1,0,-1,1),(0,fh-1,1,-1),(fw-1,fh-1,-1,-1)]:
        cv2.line(frame,(cx,cy),(cx+sx*s,cy),ACCENT,t)
        cv2.line(frame,(cx,cy),(cx,cy+sy*s),ACCENT,t)

def draw_scanline(frame, fnum):
    fh,fw = frame.shape[:2]
    yy = (fnum*4) % fh
    ov = frame.copy(); cv2.line(ov,(0,yy),(fw,yy),ACCENT,1)
    cv2.addWeighted(ov,0.05,frame,0.95,0,frame)

# ── Main system ──────────────────────────────────────────────────────────────
class AirCombatSystem:
    def __init__(self, source, model_name="yolov8s.pt", conf=0.25, detect_every=1):
        self.source = source
        self.model_name = model_name
        self.conf = conf
        self.detect_every = detect_every
        self.out_dir = Path("output_combat"); self.out_dir.mkdir(exist_ok=True)
        self.tracker = Tracker(max_age=15, iou_thresh=0.2)
        self.heatmap_pts = deque(maxlen=4000)
        self.csv_rows = []
        self.frame_num = 0
        self.peak_threat = 0.0
        self._last_dets = []
        self._load_model()

    def _load_model(self):
        from ultralytics import YOLO
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"{Fore.CYAN}Device: {self.device.upper()}{Style.RESET_ALL}")
        log.info(f"{Fore.CYAN}Loading {self.model_name}...{Style.RESET_ALL}")
        self.model = YOLO(self.model_name)
        dummy = np.zeros((640,640,3), np.uint8)
        self.model(dummy, verbose=False, device=self.device, conf=self.conf)
        log.info(f"{Fore.GREEN}Model ready.{Style.RESET_ALL}")

    def _detect(self, frame):
        res = self.model(frame, verbose=False, device=self.device,
                         classes=ALL_CLS_IDS, conf=self.conf)[0]
        dets = []
        for box in res.boxes:
            cls_id = int(box.cls[0]); c = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            if cls_id in AIRCRAFT_CLS:
                base = "aircraft"
            else:
                base = GROUND_CLS.get(cls_id, "vehicle")
            dets.append(((x1,y1,x2,y2), base, c))
        return dets

    def run(self):
        print(BANNER)
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {self.source}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        log.info(f"{Fore.CYAN}Source: {self.source} [{w}x{h} @ {fps_in:.0f}fps]{Style.RESET_ALL}")

        writer = cv2.VideoWriter(str(self.out_dir/"output.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w,h))
        pbar = tqdm(total=total_f or None, desc=f"{Fore.RED}COMBAT INTEL{Style.RESET_ALL}",
                   unit="f", dynamic_ncols=True, colour="red")
        log_ev = max(1, int(fps_in*2))
        t_prev = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret: break
            self.frame_num += 1
            t_now = time.perf_counter()
            fps_disp = 1.0/max(t_now-t_prev, 1e-6); t_prev = t_now

            if self.frame_num % self.detect_every == 0 or self.frame_num == 1:
                self._last_dets = self._detect(frame)
            tracks = self.tracker.update(self._last_dets)

            aircraft_tracks = [t for t in tracks if t.base_cls == "aircraft"]
            ground_tracks    = [t for t in tracks if t.base_cls != "aircraft"]

            speeds = []
            scores = []
            drones = 0
            for tr in aircraft_tracks:
                label = classify_aircraft(tr, w, h)
                tr._disp_label = label
                score = threat_score(tr, aircraft_tracks, w, h)
                scores.append(score)
                vx,vy = velocity_px(tr)
                speeds.append(math.hypot(vx,vy)*18)
                if label == "drone": drones += 1
                color = CLASS_COLORS.get(label, COL_MILAIR)
                draw_target(frame, tr, label, color, self.frame_num, w, h, score=score)
                cx,cy = tr.centroid(); self.heatmap_pts.append((cx,cy))

            for tr in ground_tracks:
                label = tr.base_cls
                tr._disp_label = label
                color = CLASS_COLORS.get(label, COL_VEHICLE)
                draw_target(frame, tr, label, color, self.frame_num, w, h, score=None)

            max_threat = max(scores) if scores else 0.0
            self.peak_threat = max(self.peak_threat, max_threat)
            top_target = aircraft_tracks[scores.index(max_threat)].tid if scores else -1
            collisions = sum(1 for s in scores if s >= 66)
            density = min(100.0, len(tracks) / 12.0 * 100)

            stats = {
                "aircraft": len(aircraft_tracks),
                "drones": drones,
                "avg_speed": sum(speeds)/len(speeds) if speeds else 0,
                "top_target": top_target,
                "max_threat": max_threat,
                "density": density,
                "collisions": collisions,
            }

            draw_scanline(frame, self.frame_num)
            draw_minimap(frame, tracks, w, h, self.heatmap_pts)
            draw_dashboard(frame, stats, fps_disp, self.frame_num)
            draw_corners(frame)

            if self.frame_num % log_ev == 0:
                self.csv_rows.append({
                    "frame": self.frame_num,
                    "ts": datetime.datetime.now().isoformat(),
                    "active_aircraft": stats["aircraft"],
                    "active_drones": stats["drones"],
                    "avg_speed_kmh": round(stats["avg_speed"],1),
                    "max_threat": round(stats["max_threat"],1),
                    "top_target_id": stats["top_target"],
                    "airspace_density_pct": round(stats["density"],1),
                    "collision_risks": stats["collisions"],
                })

            writer.write(frame)
            pbar.update(1)
            pbar.set_postfix({"aircraft":stats["aircraft"], "threat":f"{max_threat:.0f}",
                             "fps":f"{fps_disp:.1f}"}, refresh=True)

        pbar.close(); cap.release(); writer.release()
        self._save_outputs(w, h)
        log.info(f"{Fore.GREEN}Done. Frames={self.frame_num}  Peak threat={self.peak_threat:.0f}{Style.RESET_ALL}")

    def _save_outputs(self, w, h):
        if self.csv_rows:
            pd.DataFrame(self.csv_rows).to_csv(self.out_dir/"combat_report.csv", index=False)
            log.info(f"{Fore.GREEN}CSV -> {self.out_dir/'combat_report.csv'}{Style.RESET_ALL}")

        if self.heatmap_pts:
            hm = np.zeros((h, w), np.float32)
            for (x,y) in self.heatmap_pts:
                xi, yi = int(x), int(y)
                if 0 <= yi < h and 0 <= xi < w:
                    hm[yi, xi] += 1.0
            blurred = gaussian_filter(hm, sigma=20)
            fig, ax = plt.subplots(figsize=(12,7), facecolor="#0a0a14")
            ax.set_facecolor("#0a0a14")
            im = ax.imshow(blurred, cmap="inferno", interpolation="bilinear")
            plt.colorbar(im, ax=ax, label="Airspace Activity")
            ax.set_title("Airspace Activity Heatmap", color="white", fontsize=13, fontweight="bold")
            ax.tick_params(colors="white")
            for sp in ax.spines.values(): sp.set_edgecolor("#333")
            plt.tight_layout()
            plt.savefig(self.out_dir/"airspace_heatmap.png", dpi=130, facecolor="#0a0a14")
            plt.close()
            log.info(f"{Fore.GREEN}Heatmap -> {self.out_dir/'airspace_heatmap.png'}{Style.RESET_ALL}")

        if self.csv_rows:
            df = pd.DataFrame(self.csv_rows)
            fig, axes = plt.subplots(2, 2, figsize=(12,8), facecolor="#0a0a14")
            for ax in axes.flat:
                ax.set_facecolor("#0a0a14")
                ax.tick_params(colors="white")
                for sp in ax.spines.values(): sp.set_edgecolor("#333")
            axes[0,0].plot(df["frame"], df["active_aircraft"], color="#00d4ff")
            axes[0,0].set_title("Active Aircraft", color="white")
            axes[0,1].plot(df["frame"], df["max_threat"], color="#ff3030")
            axes[0,1].set_title("Threat Score", color="white")
            axes[1,0].plot(df["frame"], df["avg_speed_kmh"], color="#30ff90")
            axes[1,0].set_title("Avg Speed (km/h)", color="white")
            axes[1,1].plot(df["frame"], df["airspace_density_pct"], color="#ffd400")
            axes[1,1].set_title("Airspace Density (%)", color="white")
            plt.tight_layout()
            plt.savefig(self.out_dir/"combat_dashboard.png", dpi=120, facecolor="#0a0a14")
            plt.close()
            log.info(f"{Fore.GREEN}Dashboard -> {self.out_dir/'combat_dashboard.png'}{Style.RESET_ALL}")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Air Combat Intelligence System -- tubakhxn")
    ap.add_argument("source", help="Input video file")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--every", type=int, default=1, help="Run YOLO every N frames")
    args = ap.parse_args()
    AirCombatSystem(args.source, model_name=args.model, conf=args.conf,
                    detect_every=args.every).run()

if __name__ == "__main__":
    main()
