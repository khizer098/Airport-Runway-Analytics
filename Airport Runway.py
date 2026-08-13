"""
╔══════════════════════════════════════════════════════════════════╗
║     AIRPORT DIGITAL TWIN PERCEPTION SYSTEM  v1.0                ║
║     Cinematic AI Surveillance / ATC-style Demo Platform         ║
║     Developer: KHIZER AHMED                                     ║
╚══════════════════════════════════════════════════════════════════╝

"""

import subprocess, sys, time, math, datetime, threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

def install_dependencies():
    pkgs = ["ultralytics", "opencv-python", "numpy", "scipy",
            "matplotlib", "tqdm", "colorama", "Pillow"]
    print("\033[96m[INIT] Checking dependencies...\033[0m")
    for p in pkgs:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", p, "-q"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    print("\033[92m[INIT] Ready.\033[0m")

install_dependencies()

import cv2
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style, init as colorama_init
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

colorama_init(autoreset=True)

BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════╗
║   AIRPORT DIGITAL TWIN PERCEPTION SYSTEM   v1.0                ║
║   Cinematic AI Surveillance Demo  |  KHIZER AHMED              ║
╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
{Fore.YELLOW}[NOTE] Visual demo system. Risk/zone metrics are illustrative
       heuristics for showcase purposes — not an ATC safety tool.{Style.RESET_ALL}
"""

# ── Object categories & colors (BGR) ────────────────────────────────────────────
class Category(Enum):
    COMMERCIAL_AIRCRAFT = "commercial_aircraft"
    CARGO_AIRCRAFT      = "cargo_aircraft"
    PRIVATE_JET         = "private_jet"
    HELICOPTER          = "helicopter"
    FUEL_TRUCK          = "fuel_truck"
    SERVICE_VEHICLE     = "service_vehicle"
    BAGGAGE_VEHICLE     = "baggage_vehicle"
    MAINTENANCE_VEHICLE = "maintenance_vehicle"
    GROUND_CREW         = "ground_crew"
    AIRPORT_STAFF       = "airport_staff"
    UNKNOWN             = "unknown"

# Neon BGR colors
NEON = {
    Category.COMMERCIAL_AIRCRAFT: (255, 255,   0),   # neon cyan
    Category.CARGO_AIRCRAFT:      (  0, 140, 255),   # neon orange
    Category.PRIVATE_JET:         (255,   0, 200),   # neon purple/magenta
    Category.HELICOPTER:          (220,   0, 255),   # neon pink
    Category.FUEL_TRUCK:          (  0, 255, 255),   # neon yellow
    Category.SERVICE_VEHICLE:     ( 60, 255,  60),   # neon green
    Category.BAGGAGE_VEHICLE:     ( 80, 255, 170),   # neon lime
    Category.MAINTENANCE_VEHICLE: (255, 255, 255),   # neon white
    Category.GROUND_CREW:         (255, 120,  20),   # bright blue
    Category.AIRPORT_STAFF:       (255, 200, 100),   # sky blue
    Category.UNKNOWN:             (180, 180, 180),
}

ZONE_RUNWAY_ACTIVE = (40,  40, 220)   # red overlay
ZONE_TAXIWAY       = (220, 200,  40)  # cyan overlay
ZONE_SAFE          = (60, 200,  60)   # green overlay
ZONE_RESTRICTED    = (0,  140, 255)   # orange overlay

ACCENT   = (255, 220,   0)
PANEL_BG = (16,  12,   8)
TEXT_W   = (255, 255, 255)
TEXT_DIM = (150, 150, 170)
FONT     = cv2.FONT_HERSHEY_SIMPLEX


COCO_TO_CATEGORY = {
    4:  Category.COMMERCIAL_AIRCRAFT,   
    0:  Category.GROUND_CREW,           
    7:  Category.SERVICE_VEHICLE,       
    2:  Category.MAINTENANCE_VEHICLE,   
    5:  Category.CARGO_AIRCRAFT,       
    3:  Category.BAGGAGE_VEHICLE,     
}

print(BANNER)

@dataclass
class Track:
    track_id:   int
    category:   Category
    bbox:       tuple
    conf:       float
    history:    deque = field(default_factory=lambda: deque(maxlen=40))   # centroids
    last_seen:  int = 0
    speed_kmh:  float = 0.0
    heading_deg: float = 0.0

    def centroid(self):
        x1,y1,x2,y2 = self.bbox
        return ((x1+x2)//2, (y1+y2)//2)

class Tracker:
    def __init__(self, max_disappeared=25, px_per_meter=8.0, fps=30):
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.max_disappeared = max_disappeared
        self.px_per_meter = px_per_meter   
        self.fps = fps

    @staticmethod
    def _iou(b1,b2):
        x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
        x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
        inter=max(0,x2-x1)*max(0,y2-y1)
        a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
        union=a1+a2-inter
        return inter/union if union>0 else 0.0

    def update(self, detections, frame_num):
        matched=set()
        for bbox,cat,conf in detections:
            best_id,best_iou=None,0.25
            for tid,tr in self.tracks.items():
                if tid in matched: continue
                iou=self._iou(bbox,tr.bbox)
                if iou>best_iou:
                    best_iou=iou; best_id=tid
            if best_id is not None:
                tr=self.tracks[best_id]
                old_c=tr.centroid()
                tr.bbox=bbox; tr.conf=conf; tr.last_seen=frame_num
                tr.history.append(tr.centroid())
                new_c=tr.centroid()
                dx=new_c[0]-old_c[0]; dy=new_c[1]-old_c[1]
                dist_m=math.hypot(dx,dy)/max(self.px_per_meter,1e-3)
                tr.speed_kmh=dist_m*self.fps*3.6
                if dx!=0 or dy!=0:
                    tr.heading_deg=math.degrees(math.atan2(dy,dx))
                matched.add(best_id)
            else:
                tid=self.next_id; self.next_id+=1
                tr=Track(track_id=tid,category=cat,bbox=bbox,conf=conf,last_seen=frame_num)
                tr.history.append(tr.centroid())
                self.tracks[tid]=tr
                matched.add(tid)
        stale=[tid for tid,tr in self.tracks.items() if frame_num-tr.last_seen>self.max_disappeared]
        for tid in stale: del self.tracks[tid]
        return [tr for tid,tr in self.tracks.items() if tid in matched]

    def predict_path(self, track: Track, steps=15):
        """Linear extrapolation from last N centroids → ghost future points."""
        if len(track.history) < 3:
            return []
        pts = list(track.history)[-8:]
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        t  = np.arange(len(pts))
        try:
            vx = np.polyfit(t, xs, 1)[0]
            vy = np.polyfit(t, ys, 1)[0]
        except Exception:
            return []
        last = pts[-1]
        future=[]
        for s in range(1, steps+1):
            future.append((int(last[0]+vx*s), int(last[1]+vy*s)))
        return future

def alpha_blend(frame, mask_color_layer, alpha=0.35):
    cv2.addWeighted(mask_color_layer, alpha, frame, 1-alpha, 0, frame)

def draw_seg_mask(layer, bbox, color, shape="ellipse"):
    x1,y1,x2,y2 = bbox
    if shape=="ellipse":
        cx,cy=(x1+x2)//2,(y1+y2)//2
        ax,ay=max(2,(x2-x1)//2),max(2,(y2-y1)//2)
        cv2.ellipse(layer,(cx,cy),(ax,ay),0,0,360,color,-1)
    else:
        cv2.rectangle(layer,(x1,y1),(x2,y2),color,-1)

def glow_edge(frame, bbox, color, passes=3):
    x1,y1,x2,y2 = bbox
    for i in range(passes,0,-1):
        t = passes - i + 1
        glow_col = tuple(min(255,int(c*0.55)) for c in color)
        cv2.rectangle(frame,(x1-i,y1-i),(x2+i,y2+i),glow_col,1,cv2.LINE_AA)
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2,cv2.LINE_AA)

def corner_ticks(frame, bbox, color, sz=14, th=2):
    x1,y1,x2,y2=bbox
    for px,py,sx,sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame,(px,py),(px+sx*sz,py),color,th)
        cv2.line(frame,(px,py),(px,py+sy*sz),color,th)

def target_lock_anim(frame, bbox, color, frame_num):
    """Rotating bracket / pulsing target-lock effect."""
    x1,y1,x2,y2=bbox
    cx,cy=(x1+x2)//2,(y1+y2)//2
    r = max(x2-x1,y2-y1)//2 + 14
    pulse = 0.5+0.5*math.sin(frame_num*0.25)
    rad = int(r*(0.9+0.1*pulse))
    n=4
    ang0 = (frame_num*4) % 360
    for k in range(n):
        a0 = math.radians(ang0 + k*(360/n))
        a1 = math.radians(ang0 + k*(360/n) + 28)
        pts=[]
        for a in np.linspace(a0,a1,6):
            pts.append((int(cx+rad*math.cos(a)), int(cy+rad*math.sin(a))))
        cv2.polylines(frame,[np.array(pts,np.int32)],False,color,2,cv2.LINE_AA)

def label_panel(frame, lines, x, y, color, width=210, line_h=15, title=None):
    h = (len(lines)+ (1 if title else 0))*line_h + 10
    ov = frame.copy()
    cv2.rectangle(ov,(x,y),(x+width,y+h),(10,10,18),-1)
    cv2.addWeighted(ov,0.78,frame,0.22,0,frame)
    cv2.rectangle(frame,(x,y),(x+width,y+h),color,1)
    yy=y+14
    if title:
        cv2.putText(frame,title,(x+6,yy),FONT,0.34,color,1,cv2.LINE_AA)
        yy+=line_h
        cv2.line(frame,(x+2,yy-9),(x+width-2,yy-9),color,1)
    for ln in lines:
        cv2.putText(frame,ln,(x+6,yy),FONT,0.32,TEXT_W,1,cv2.LINE_AA)
        yy+=line_h

def draw_trajectory(frame, history, color, fade=True):
    pts=list(history)
    n=len(pts)
    if n<2: return
    for i in range(1,n):
        a = i/n
        col = tuple(int(c*(0.25+0.75*a)) for c in color) if fade else color
        cv2.line(frame, pts[i-1], pts[i], col, max(1,int(2*a)), cv2.LINE_AA)

def draw_predicted_path(frame, future_pts, color):
    if not future_pts: return
    for i,p in enumerate(future_pts):
        alpha = 1 - i/len(future_pts)
        r = max(2,int(5*alpha))
        ghost = tuple(int(c*0.6) for c in color)
        cv2.circle(frame, p, r, ghost, 1, cv2.LINE_AA)
    pts = np.array(future_pts, np.int32)
    cv2.polylines(frame,[pts],False, tuple(int(c*0.7) for c in color),1,cv2.LINE_AA)

def estimate_status(track: Track) -> str:
    if track.speed_kmh < 1.5: return "Stationary"
    if track.speed_kmh < 25:  return "Taxiing"
    if track.speed_kmh < 60:  return "Moving"
    return "High-Speed"

def estimate_risk(track: Track, restricted_zones: list) -> tuple:
    """Toy heuristic combining speed + proximity to a restricted zone line.
    Returns (label, color). Purely illustrative — demo purposes only."""
    score = 0
    if track.speed_kmh > 50: score += 2
    elif track.speed_kmh > 25: score += 1
    cx, cy = track.centroid()
    for (zx1,zy1,zx2,zy2) in restricted_zones:
        if zx1<=cx<=zx2 and zy1<=cy<=zy2:
            score += 2
            break
    if score >= 3: return "HIGH RISK", (40,40,235)
    if score >= 1: return "MED RISK",  (0,165,255)
    return "LOW RISK", (60,200,60)

class DigitalTwin:
    def __init__(self, fw, fh, panel_w=320, panel_h=220):
        self.fw=fw; self.fh=fh
        self.pw=panel_w; self.ph=panel_h

    def render(self, tracks: list, zones: dict):
        """Returns a small BGR image representing the schematic bird's-eye view."""
        panel = np.full((self.ph, self.pw, 3), (14,14,22), np.uint8)
        cv2.rectangle(panel,(0,0),(self.pw-1,self.ph-1),ACCENT,1)
        cv2.putText(panel,"DIGITAL TWIN - BIRDS EYE",(6,14),FONT,0.35,ACCENT,1,cv2.LINE_AA)

        sx = self.pw/self.fw; sy=self.ph/self.fh

        for zname,(zx1,zy1,zx2,zy2,zcol) in zones.items():
            p1=(int(zx1*sx),int(zy1*sy)); p2=(int(zx2*sx),int(zy2*sy))
            ov = panel.copy()
            cv2.rectangle(ov,p1,p2,zcol,-1)
            cv2.addWeighted(ov,0.25,panel,0.75,0,panel)
            cv2.rectangle(panel,p1,p2,zcol,1)

        for tr in tracks:
            cx,cy = tr.centroid()
            px,py = int(cx*sx), int(cy*sy)
            col = NEON.get(tr.category, (180,180,180))
            cv2.circle(panel,(px,py),3,col,-1)
            cv2.circle(panel,(px,py),5,col,1)
            pts=list(tr.history)[-10:]
            if len(pts)>1:
                scaled=[(int(p[0]*sx),int(p[1]*sy)) for p in pts]
                cv2.polylines(panel,[np.array(scaled,np.int32)],False,col,1,cv2.LINE_AA)

        cv2.putText(panel,f"Tracked: {len(tracks)}",(6,self.ph-8),
                    FONT,0.32,TEXT_DIM,1,cv2.LINE_AA)
        return panel

class HeatmapBank:
    def __init__(self, fw, fh):
        self.fw=fw; self.fh=fh
        self.layers = {
            "aircraft": np.zeros((fh,fw),np.float32),
            "crew":     np.zeros((fh,fw),np.float32),
            "vehicle":  np.zeros((fh,fw),np.float32),
        }

    def update(self, tracks: list):
        for tr in tracks:
            cx,cy = tr.centroid()
            if not (0<=cy<self.fh and 0<=cx<self.fw): continue
            if tr.category in (Category.COMMERCIAL_AIRCRAFT,Category.CARGO_AIRCRAFT,
                               Category.PRIVATE_JET,Category.HELICOPTER):
                self.layers["aircraft"][cy,cx]+=1
            elif tr.category in (Category.GROUND_CREW,Category.AIRPORT_STAFF):
                self.layers["crew"][cy,cx]+=1
            else:
                self.layers["vehicle"][cy,cx]+=1

    def save_all(self, out_dir: Path):
        for name, layer in self.layers.items():
            if layer.max() <= 0: continue
            blurred = gaussian_filter(layer, sigma=16)
            fig,ax=plt.subplots(figsize=(10,6),facecolor='#0a0a14')
            ax.set_facecolor('#0a0a14')
            im=ax.imshow(blurred,cmap='plasma',interpolation='bilinear')
            plt.colorbar(im,ax=ax,label='Activity Density')
            ax.set_title(f'{name.title()} Activity Heatmap',color='white',fontweight='bold')
            ax.tick_params(colors='white')
            for sp in ax.spines.values(): sp.set_edgecolor('#333')
            plt.tight_layout()
            p = out_dir/f"heatmap_{name}.png"
            plt.savefig(p,dpi=130,facecolor='#0a0a14'); plt.close()
            print(f"{Fore.GREEN}[HEATMAP] {name} → {p}{Style.RESET_ALL}")

def draw_corner_frame(frame, color=ACCENT):
    fh,fw=frame.shape[:2]; s=36; t=2
    for cx,cy,sx,sy in [(0,0,1,1),(fw-1,0,-1,1),(0,fh-1,1,-1),(fw-1,fh-1,-1,-1)]:
        cv2.line(frame,(cx,cy),(cx+sx*s,cy),color,t)
        cv2.line(frame,(cx,cy),(cx,cy+sy*s),color,t)

def draw_scanline(frame, frame_num, color=ACCENT):
    fh,fw=frame.shape[:2]
    y=(frame_num*4)%fh
    ov=frame.copy(); cv2.line(ov,(0,y),(fw,y),color,1)
    cv2.addWeighted(ov,0.06,frame,0.94,0,frame)

def draw_global_dashboard(frame, stats, fps, frame_num):
    fh,fw=frame.shape[:2]
    pw,ph=290,200
    x,y = fw-pw-12, 12
    ov=frame.copy()
    cv2.rectangle(ov,(x,y),(x+pw,y+ph),PANEL_BG,-1)
    cv2.addWeighted(ov,0.85,frame,0.15,0,frame)
    cv2.rectangle(frame,(x,y),(x+pw,y+ph),ACCENT,1)
    cv2.line(frame,(x+1,y+22),(x+pw-1,y+22),ACCENT,1)
    cv2.putText(frame,"AIRPORT OPS - LIVE FEED",(x+5,y+15),FONT,0.37,ACCENT,1,cv2.LINE_AA)
    now=datetime.datetime.now().strftime("%H:%M:%S  %d/%m/%Y")
    rows=[
        ("FPS",        f"{fps:>6.1f}"),
        ("TIME",       now),
        ("FRAME",      f"{frame_num:>6d}"),
        ("AIRCRAFT",   f"{stats.get('aircraft',0):>4d}"),
        ("VEHICLES",   f"{stats.get('vehicles',0):>4d}"),
        ("CREW",       f"{stats.get('crew',0):>4d}"),
        ("HIGH RISK",  f"{stats.get('high_risk',0):>4d}"),
    ]
    for i,(k,v) in enumerate(rows):
        ky=y+36+i*20
        cv2.putText(frame,f" {k:<12}",(x+4,ky),FONT,0.32,TEXT_DIM,1,cv2.LINE_AA)
        cv2.putText(frame,v,(x+150,ky),FONT,0.34,TEXT_W,1,cv2.LINE_AA)
    pulse = int((time.time()*3)%2)
    cv2.putText(frame, "● LIVE" if pulse else "○ LIVE", (x+pw-58,y+15),
                FONT,0.36,(60,220,60) if pulse else (0,120,0),1,cv2.LINE_AA)

def overlay_digital_twin(frame, twin_img, x=12, y=None):
    fh,fw = frame.shape[:2]
    th,tw = twin_img.shape[:2]
    if y is None: y = fh - th - 12
    frame[y:y+th, x:x+tw] = cv2.addWeighted(
        frame[y:y+th, x:x+tw], 0.05, twin_img, 0.95, 0)
    cv2.rectangle(frame,(x,y),(x+tw,y+th),ACCENT,1)

class AirportPerceptionSystem:
    def __init__(self, source: str, model_name="yolov8s.pt",
                 conf=0.25, detect_every=2):
        self.source = source
        self.model_name = model_name
        self.conf = conf
        self.detect_every = detect_every
        self.out_dir = Path("output_airport"); self.out_dir.mkdir(exist_ok=True)
        self.frame_num = 0
        self._last_dets = []
        self._load_model()

    def _load_model(self):
        from ultralytics import YOLO
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"{Fore.CYAN}[INIT] Device: {self.device.upper()}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[INIT] Loading {self.model_name}...{Style.RESET_ALL}")
        self.model = YOLO(self.model_name)
        dummy = np.zeros((640,640,3),np.uint8)
        self.model(dummy, verbose=False, device=self.device, conf=self.conf)
        print(f"{Fore.GREEN}[INIT] Model ready.{Style.RESET_ALL}")

    def _open(self):
        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source}")
        return cap

    def _detect(self, frame):
        res = self.model(frame, verbose=False, device=self.device,
                         classes=list(COCO_TO_CATEGORY.keys()), conf=self.conf)[0]
        dets=[]
        for box in res.boxes:
            cls_id=int(box.cls[0]); conf=float(box.conf[0])
            cat = COCO_TO_CATEGORY.get(cls_id, Category.UNKNOWN)
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            dets.append(((x1,y1,x2,y2), cat, conf))
        return dets

    def run(self):
        cap = self._open()
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"{Fore.CYAN}[VIDEO] {self.source}  [{w}x{h} @ {fps_in:.1f}fps, "
              f"{total_f} frames]{Style.RESET_ALL}")

        tracker = Tracker(fps=fps_in)
        twin    = DigitalTwin(w, h, panel_w=min(360,w//3), panel_h=min(240,h//3))
        heatmap = HeatmapBank(w, h)

        zones_minimap = {
            "Active Runway": (int(w*0.10), int(h*0.05), int(w*0.90), int(h*0.22), ZONE_RUNWAY_ACTIVE),
            "Taxiway":        (int(w*0.10), int(h*0.25), int(w*0.90), int(h*0.40), ZONE_TAXIWAY),
            "Safe Apron":     (int(w*0.10), int(h*0.45), int(w*0.90), int(h*0.75), ZONE_SAFE),
            "Restricted":     (int(w*0.70), int(h*0.78), int(w*0.95), int(h*0.95), ZONE_RESTRICTED),
        }
        restricted_boxes = [v[:4] for k,v in zones_minimap.items() if k=="Restricted"]

        out_path = str(self.out_dir/"output.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w,h))

        pbar = tqdm(total=total_f or None, desc=f"{Fore.CYAN}AIRPORT{Style.RESET_ALL}",
                    unit="f", dynamic_ncols=True, colour="cyan")

        t_prev = time.perf_counter(); fps_disp = fps_in

        while True:
            ret, frame = cap.read()
            if not ret: break
            self.frame_num += 1

            t_now = time.perf_counter()
            fps_disp = 1.0/max(t_now-t_prev,1e-6); t_prev=t_now

            if self.frame_num % self.detect_every == 0 or self.frame_num == 1:
                self._last_dets = self._detect(frame)
            dets = self._last_dets

            tracks = tracker.update(dets, self.frame_num)
            heatmap.update(tracks)

            mask_layer = np.zeros_like(frame)
            for tr in tracks:
                col = NEON.get(tr.category, (180,180,180))
                draw_seg_mask(mask_layer, tr.bbox, col, shape="ellipse")
            alpha_blend(frame, mask_layer, alpha=0.30)

            stats = defaultdict(int)
            for tr in tracks:
                col = NEON.get(tr.category, (180,180,180))
                glow_edge(frame, tr.bbox, col, passes=3)
                corner_ticks(frame, tr.bbox, col)
                draw_trajectory(frame, tr.history, col)

                is_aircraft = tr.category in (Category.COMMERCIAL_AIRCRAFT,
                                              Category.CARGO_AIRCRAFT,
                                              Category.PRIVATE_JET,
                                              Category.HELICOPTER)
                if is_aircraft:
                    future = tracker.predict_path(tr, steps=12)
                    draw_predicted_path(frame, future, col)
                    target_lock_anim(frame, tr.bbox, col, self.frame_num)
                    status = estimate_status(tr)
                    risk_label, risk_col = estimate_risk(tr, restricted_boxes)
                    lines = [
                        f"Status: {status}",
                        f"Speed: {tr.speed_kmh:5.1f} km/h",
                        f"Heading: {tr.heading_deg:6.1f} deg",
                        f"Risk: {risk_label}",
                    ]
                    label_panel(frame, lines, tr.bbox[0], max(0,tr.bbox[1]-78),
                               col, width=190, title=f"Aircraft #{tr.track_id}")
                    stats["aircraft"] += 1
                    if risk_label=="HIGH RISK": stats["high_risk"]+=1
                elif tr.category in (Category.GROUND_CREW, Category.AIRPORT_STAFF):
                    lines=[f"Movement: {estimate_status(tr)}",
                           f"Speed: {tr.speed_kmh:4.1f} km/h"]
                    label_panel(frame, lines, tr.bbox[0], max(0,tr.bbox[1]-46),
                               col, width=140, title=f"Crew #{tr.track_id}")
                    stats["crew"] += 1
                else:
                    lines=[f"Type: {tr.category.value.replace('_',' ').title()}",
                           f"Speed: {tr.speed_kmh:4.1f} km/h"]
                    label_panel(frame, lines, tr.bbox[0], max(0,tr.bbox[1]-46),
                               col, width=160, title=f"Vehicle #{tr.track_id}")
                    stats["vehicles"] += 1

            zone_layer = frame.copy()
            for zname,(zx1,zy1,zx2,zy2,zcol) in zones_minimap.items():
                cv2.rectangle(zone_layer,(zx1,zy1),(zx2,zy2),zcol,-1)
            cv2.addWeighted(zone_layer, 0.07, frame, 0.93, 0, frame)
            for zname,(zx1,zy1,zx2,zy2,zcol) in zones_minimap.items():
                cv2.rectangle(frame,(zx1,zy1),(zx2,zy2),zcol,1)
                cv2.putText(frame,zname,(zx1+4,zy1+14),FONT,0.34,zcol,1,cv2.LINE_AA)

            draw_scanline(frame, self.frame_num)
            draw_corner_frame(frame)
            draw_global_dashboard(frame, stats, fps_disp, self.frame_num)

            twin_img = twin.render(tracks, zones_minimap)
            overlay_digital_twin(frame, twin_img)

            writer.write(frame)
            pbar.update(1)
            pbar.set_postfix({
                "aircraft": stats["aircraft"], "vehicles": stats["vehicles"],
                "crew": stats["crew"], "fps": f"{fps_disp:.1f}"
            }, refresh=True)

        pbar.close(); cap.release(); writer.release()
        heatmap.save_all(self.out_dir)
        print(f"{Fore.GREEN}[DONE] Output video → {out_path}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[DONE] Total frames processed: {self.frame_num}{Style.RESET_ALL}")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Airport Digital Twin Perception System — tubakhxn (visual demo)"
    )
    ap.add_argument("source", nargs="?", default="0",
                    help="Video file path or camera index (default: 0)")
    ap.add_argument("--model", default="yolov8s.pt",
                    help="YOLO model (default: yolov8s.pt)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Detection confidence threshold (default: 0.25)")
    ap.add_argument("--every", type=int, default=2,
                    help="Run detector every N frames (default: 2)")
    args = ap.parse_args()

    system = AirportPerceptionSystem(
        source=args.source, model_name=args.model,
        conf=args.conf, detect_every=args.every
    )
    system.run()


if __name__ == "__main__":
    main()
