"""
╔══════════════════════════════════════════════════════════════════╗
║        LICENSE PLATE INTELLIGENCE SYSTEM v4.0                  ║
║        Fast Automatic Number Plate Recognition                 ║
║        Developer: KHIZER AHMED                                 ║
╚══════════════════════════════════════════════════════════════════╝

"""

import subprocess, sys, time, datetime, csv, re, threading, os, urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

def install_dependencies():
    pkgs = ["ultralytics", "opencv-python", "easyocr", "numpy",
            "tqdm", "colorama", "pandas", "Pillow"]
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
import tkinter as tk
from PIL import Image, ImageTk
from tqdm import tqdm
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

BANNER = f"""
{Fore.GREEN}╔══════════════════════════════════════════════════════════════════╗
║     LICENSE PLATE INTELLIGENCE SYSTEM   v4.0                  ║
║     Real Plate Detector  |  Fast ANPR  |  tubakhxn             ║
╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}"""

PLATE_MODEL_FILENAME = "license_plate_detector.pt"
PLATE_MODEL_URL = ("https://github.com/Muhammad-Zeerak-Khan/"
                    "Automatic-License-Plate-Recognition-using-YOLOv8/"
                    "raw/main/license_plate_detector.pt")

def ensure_plate_model() -> str:
    """Download the license-plate YOLOv8 weights if not already present."""
    local = Path(__file__).resolve().parent / PLATE_MODEL_FILENAME
    if local.exists() and local.stat().st_size > 1_000_000:
        return str(local)
    print(f"{Fore.CYAN}[INIT] Downloading license-plate detector model...{Style.RESET_ALL}")
    try:
        urllib.request.urlretrieve(PLATE_MODEL_URL, str(local))
        if local.exists() and local.stat().st_size > 1_000_000:
            print(f"{Fore.GREEN}[INIT] Plate model downloaded.{Style.RESET_ALL}")
            return str(local)
    except Exception as e:
        print(f"{Fore.RED}[INIT] Plate model download failed: {e}{Style.RESET_ALL}")
    raise RuntimeError(
        "Could not obtain license_plate_detector.pt. Download it manually from:\n"
        f"  {PLATE_MODEL_URL}\n"
        f"and place it next to this script as '{PLATE_MODEL_FILENAME}'."
    )

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
NEON_GREEN  = (57, 255,  20)
NEON_CYAN   = (255, 255,   0)
NEON_PINK   = (255,   0, 255)
NEON_YELLOW = (0,  220, 255)
TEXT_W      = (255, 255, 255)
FONT        = cv2.FONT_HERSHEY_SIMPLEX

# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class PlateRecord:
    plate_text: str
    confidence: float
    vehicle_id: int
    timestamp: str
    frame_number: int

@dataclass
class VehicleTrack:
    vehicle_id: int
    bbox: tuple
    class_name: str
    confidence: float
    plates: list = field(default_factory=list)
    last_seen: int = 0
    last_ocr_frame: int = -999
    confirmed_plate: str = ""
    confirmed_conf: float = 0.0

# ── HUD drawing ────────────────────────────────────────────────────────────────
def draw_neon_box(frame, x1, y1, x2, y2, color, thickness=2, glow=True):
    if glow:
        glow_col = tuple(min(255, int(c*0.4)) for c in color)
        cv2.rectangle(frame, (x1,y1), (x2,y2), glow_col, thickness+3)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
    corner = 12
    for cx,cy in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
        dx = 1 if cx==x1 else -1
        dy = 1 if cy==y1 else -1
        cv2.line(frame,(cx,cy),(cx+dx*corner,cy),color,thickness+1)
        cv2.line(frame,(cx,cy),(cx,cy+dy*corner),color,thickness+1)

def draw_label(frame, text, x, y, color, bg=(0,0,0), fs=0.5):
    (tw,th),base = cv2.getTextSize(text, FONT, fs, 2)
    pad=4
    y = max(y, th+pad+2)
    cv2.rectangle(frame,(x-pad,y-th-pad),(x+tw+pad,y+base+pad),bg,-1)
    cv2.rectangle(frame,(x-pad,y-th-pad),(x+tw+pad,y+base+pad),color,1)
    cv2.putText(frame,text,(x,y),FONT,fs,color,2,cv2.LINE_AA)

def draw_plate_panel(frame, plate_text, conf, x, y, vehicle_id, confirmed=False):
    pw,ph = 270, 52
    x = max(0, min(x, frame.shape[1]-pw))
    y = max(0, min(y, frame.shape[0]-ph))
    ov = frame.copy()
    cv2.rectangle(ov,(x,y),(x+pw,y+ph),(0,0,0),-1)
    cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
    col = NEON_GREEN if confirmed else NEON_YELLOW
    cv2.rectangle(frame,(x,y),(x+pw,y+ph),col,1)
    cv2.line(frame,(x,y+18),(x+pw,y+18),col,1)
    status = "LOCKED" if confirmed else "READING"
    cv2.putText(frame,f"VEH#{vehicle_id:03d}  {status}",(x+5,y+13),
                FONT,0.36,NEON_CYAN,1,cv2.LINE_AA)
    cv2.putText(frame, plate_text if plate_text else "...",
                (x+5,y+40),FONT,0.85,col,2,cv2.LINE_AA)
    if conf > 0:
        cv2.putText(frame,f"{conf*100:.0f}%",(x+pw-50,y+40),
                    FONT,0.45,NEON_YELLOW,1,cv2.LINE_AA)

def draw_dashboard(frame, stats, fps, frame_num):
    pw,ph = 290, 190
    x,y = 10,10
    ov = frame.copy()
    cv2.rectangle(ov,(x,y),(x+pw,y+ph),(5,5,15),-1)
    cv2.addWeighted(ov,0.82,frame,0.18,0,frame)
    cv2.rectangle(frame,(x,y),(x+pw,y+ph),NEON_GREEN,1)
    cv2.line(frame,(x,y+22),(x+pw,y+22),NEON_GREEN,1)
    cv2.putText(frame,"  ANPR INTELLIGENCE SYSTEM",(x+4,y+16),
                FONT,0.42,NEON_CYAN,1,cv2.LINE_AA)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        ("FPS",       f"{fps:>6.1f}",                          NEON_GREEN),
        ("FRAME",     f"{frame_num:>6d}",                      NEON_GREEN),
        ("TIME",      now,                                     NEON_CYAN),
        ("VEHICLES",  f"{stats.get('vehicles',0):>6d}",        NEON_GREEN),
        ("PLATES",    f"{stats.get('plates',0):>6d}",          NEON_GREEN),
        ("UNIQUE",    f"{stats.get('unique',0):>6d}",          NEON_YELLOW),
        ("TOP PLATE", f"{stats.get('top_plate','N/A')}",       NEON_PINK),
    ]
    for i,(k,v,col) in enumerate(rows):
        cv2.putText(frame,f"  {k:<10}: {v}",(x+4,y+38+i*22),
                    FONT,0.40,col,1,cv2.LINE_AA)
    pulse = int((time.time()*3)%2)
    cv2.putText(frame,"● LIVE" if pulse else "○ LIVE",(x+pw-58,y+16),
                FONT,0.4, NEON_GREEN if pulse else (0,140,0),1,cv2.LINE_AA)

# ── Tracker ────────────────────────────────────────────────────────────────────
class VehicleTracker:
    def __init__(self, max_disappeared=30):
        self.tracks: dict[int, VehicleTrack] = {}
        self.next_id = 1
        self.max_disappeared = max_disappeared

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
        for bbox,cls,conf in detections:
            best_id,best_iou=None,0.3
            for vid,tr in self.tracks.items():
                if vid in matched: continue
                iou=self._iou(bbox,tr.bbox)
                if iou>best_iou: best_iou=iou; best_id=vid
            if best_id is not None:
                tr=self.tracks[best_id]
                tr.bbox=bbox; tr.last_seen=frame_num
                matched.add(best_id)
            else:
                tid=self.next_id; self.next_id+=1
                self.tracks[tid]=VehicleTrack(
                    vehicle_id=tid,bbox=bbox,class_name=cls,
                    confidence=conf,last_seen=frame_num)
                matched.add(tid)
        stale=[vid for vid,t in self.tracks.items()
               if frame_num-t.last_seen>self.max_disappeared]
        for vid in stale: del self.tracks[vid]
        return [t for vid,t in self.tracks.items() if vid in matched]

# ── OCR Engine ─────────────────────────────────────────────────────────────────
class OCREngine:
    def __init__(self):
        import easyocr
        print(f"{Fore.CYAN}[INIT] Loading EasyOCR...{Style.RESET_ALL}")
        self.reader = easyocr.Reader(['en'], gpu=self._has_gpu(), verbose=False)
        print(f"{Fore.GREEN}[INIT] EasyOCR ready.{Style.RESET_ALL}")

    @staticmethod
    def _has_gpu():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def read_plate(self, roi: np.ndarray) -> tuple:
        if roi is None or roi.size == 0:
            return "", 0.0
        h, w = roi.shape[:2]
        if h < 5 or w < 5:
            return "", 0.0
        target_h = 80
        scale = target_h / max(h, 1)
        roi_r = cv2.resize(roi, (max(1,int(w*scale)), target_h),
                           interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 8
        )
        results = self.reader.readtext(
            gray,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1
        )
        if not results:
            return "", 0.0
        texts, confs = [], []
        for (_, text, conf) in results:
            clean = re.sub(r'[^A-Z0-9]', '', text.upper())
            if clean and conf > 0.25:
                texts.append(clean); confs.append(conf)
        if texts:
            return "".join(texts), float(np.mean(confs))
        return "", 0.0

# ── Tkinter GUI ────────────────────────────────────────────────────────────────
class ANPRGui:
    def __init__(self, dw=1280, dh=720):
        self.dw=dw; self.dh=dh
        self.root=tk.Tk()
        self.root.title("ANPR Intelligence System v4.0 — tubakhxn")
        self.root.configure(bg="#0a0f08")
        self._stopped=False; self._paused=False
        self._q: deque=deque(maxlen=2); self._lock=threading.Lock()
        self._latest=None; self._tk_img=None
        self._out_dir=Path("output_anpr"); self._out_dir.mkdir(exist_ok=True)

        hdr=tk.Frame(self.root,bg="#0a0f08"); hdr.pack(fill=tk.X)
        tk.Label(hdr,text="◈  ANPR INTELLIGENCE SYSTEM  v4.0  ◈  Real Plate Detector",
                 bg="#0a0f08",fg="#39ff14",
                 font=("Consolas",12,"bold")).pack(side=tk.LEFT,padx=12,pady=6)
        tk.Label(hdr,text="tubakhxn",bg="#0a0f08",fg="#ff00ff",
                 font=("Consolas",10)).pack(side=tk.RIGHT,padx=12)

        self.canvas=tk.Canvas(self.root,width=dw,height=dh,
                              bg="#020602",highlightthickness=2,
                              highlightbackground="#39ff14")
        self.canvas.pack(padx=8,pady=(0,4))

        ftr=tk.Frame(self.root,bg="#0a0f08"); ftr.pack(fill=tk.X)
        tk.Label(ftr,text="  [Q] Quit   [S] Snapshot   [Space] Pause/Resume",
                 bg="#0a0f08",fg="#374151",
                 font=("Consolas",9)).pack(side=tk.LEFT)
        self.status_var=tk.StringVar(value="▶  Initialising...")
        tk.Label(ftr,textvariable=self.status_var,bg="#0a0f08",fg="#39ff14",
                 font=("Consolas",9)).pack(side=tk.RIGHT,padx=12)

        for k in ("<q>","<Q>"): self.root.bind(k,lambda e:self.stop())
        for k in ("<s>","<S>"): self.root.bind(k,self._snap)
        self.root.bind("<space>",self._pause)
        self.root.protocol("WM_DELETE_WINDOW",self.stop)

    def push_frame(self,bgr):
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        pil=Image.fromarray(rgb).resize((self.dw,self.dh),Image.LANCZOS)
        with self._lock:
            self._q.append(pil); self._latest=bgr.copy()

    def _refresh(self):
        with self._lock:
            if self._q:
                pil=self._q.pop()
                self._tk_img=ImageTk.PhotoImage(pil)
                self.canvas.create_image(0,0,anchor=tk.NW,image=self._tk_img)
        if not self._stopped: self.root.after(16,self._refresh)

    def _snap(self,e=None):
        if self._latest is not None:
            p=self._out_dir/f"snap_{int(time.time())}.jpg"
            cv2.imwrite(str(p),self._latest)
            self.status_var.set(f"✔ Saved {p.name}")

    def _pause(self,e=None):
        self._paused=not self._paused
        self.status_var.set("⏸ Paused" if self._paused else "▶  Processing...")

    def set_status(self,t): self.status_var.set(t)
    def stop(self):
        self._stopped=True
        try: self.root.quit()
        except: pass
    def start(self):
        self.root.after(16,self._refresh); self.root.mainloop()
    @property
    def stopped(self): return self._stopped
    @property
    def paused(self): return self._paused

# ── Main System ────────────────────────────────────────────────────────────────
class LicensePlateSystem:
    VEHICLE_CLASSES = {2:"car",3:"motorcycle",5:"bus",7:"truck"}

    def __init__(self, source, detect_every=2, half_res=False,
                 ocr_recheck_interval=45, max_ocr_per_frame=3,
                 plate_conf=0.25, dw=1280, dh=720):
        self.source = source
        self.detect_every = detect_every
        self.half_res = half_res                      # OFF by default now
        self.ocr_recheck_interval = ocr_recheck_interval
        self.max_ocr_per_frame = max_ocr_per_frame
        self.plate_conf = plate_conf
        self.dw=dw; self.dh=dh
        self.output_dir = Path("output_anpr")
        self.screenshots_dir = self.output_dir/"screenshots"
        self.output_dir.mkdir(exist_ok=True)
        self.screenshots_dir.mkdir(exist_ok=True)
        self.plate_records: list[PlateRecord] = []
        self.plate_counts: dict = defaultdict(int)
        self.total_vehicles = 0
        self.frame_num = 0
        self._last_dets = []
        self._load_models()
        self.tracker = VehicleTracker()
        self.ocr = OCREngine()

    def _load_models(self):
        from ultralytics import YOLO
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"{Fore.CYAN}[INIT] Device: {self.device.upper()}{Style.RESET_ALL}")

        self.model = YOLO("yolov8n.pt")
        dummy = np.zeros((320,320,3),np.uint8)
        self.model(dummy,verbose=False,device=self.device)
        print(f"{Fore.GREEN}[INIT] YOLOv8n (vehicles) ready.{Style.RESET_ALL}")

        plate_weights = ensure_plate_model()
        self.plate_model = YOLO(plate_weights)
        self.plate_model(dummy, verbose=False, device=self.device)
        print(f"{Fore.GREEN}[INIT] License-plate detector ready.{Style.RESET_ALL}")

    def _open(self):
        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {self.source}")
        return cap

    def _detect_vehicles(self, frame):
        h,w = frame.shape[:2]
        if self.half_res:
            small = cv2.resize(frame,(w//2,h//2))
            scale = 2.0
        else:
            small = frame; scale = 1.0
        res = self.model(small, verbose=False, device=self.device,
                         classes=list(self.VEHICLE_CLASSES.keys()), conf=0.35)[0]
        dets=[]
        for box in res.boxes:
            cls_id=int(box.cls[0]); conf=float(box.conf[0])
            x1,y1,x2,y2=map(int,box.xyxy[0])
            x1=int(x1*scale); y1=int(y1*scale); x2=int(x2*scale); y2=int(y2*scale)
            dets.append(((x1,y1,x2,y2), self.VEHICLE_CLASSES[cls_id], conf))
        return dets

    def _detect_plates(self, frame):
        """Run the REAL plate detector on the full frame. Returns list of
        (x1,y1,x2,y2,conf) in frame coordinates."""
        res = self.plate_model(frame, verbose=False, device=self.device,
                               conf=self.plate_conf)[0]
        plates=[]
        for box in res.boxes:
            conf=float(box.conf[0])
            x1,y1,x2,y2=map(int,box.xyxy[0])
            plates.append((x1,y1,x2,y2,conf))
        return plates

    @staticmethod
    def _plate_in_vehicle(plate_box, vehicle_box):
        px1,py1,px2,py2,_ = plate_box
        vx1,vy1,vx2,vy2 = vehicle_box
        cx, cy = (px1+px2)/2, (py1+py2)/2
        return vx1 <= cx <= vx2 and vy1 <= cy <= vy2

    def _stats(self):
        top = max(self.plate_counts, key=self.plate_counts.get) if self.plate_counts else "N/A"
        return {"vehicles": self.total_vehicles, "plates": len(self.plate_records),
                "unique": len(self.plate_counts), "top_plate": top}

    def _save_csv(self):
        p = self.output_dir/"plates.csv"
        with open(p,"w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["plate_text","confidence","vehicle_id","timestamp","frame_number"])
            for r in self.plate_records:
                w.writerow([r.plate_text,f"{r.confidence:.3f}",r.vehicle_id,
                           r.timestamp,r.frame_number])
        print(f"{Fore.GREEN}[SAVE] CSV → {p}{Style.RESET_ALL}")

    def _screenshot(self, frame):
        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(str(self.screenshots_dir/f"frame_{ts}.jpg"), frame)

    def _process(self, gui: ANPRGui):
        cap = self._open()
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"{Fore.CYAN}[VIDEO] {self.source}  [{w}x{h} @ {fps_in:.1f}fps]  "
              f"detect_every={self.detect_every}  half_res={self.half_res}{Style.RESET_ALL}")

        out_path = str(self.output_dir/"output.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w,h))
        screenshot_interval = max(1, int(fps_in*5))

        pbar = tqdm(total=total_f or None, desc=f"{Fore.GREEN}ANPR v4{Style.RESET_ALL}",
                    unit="f", dynamic_ncols=True, colour="green")

        t_prev = time.perf_counter(); fps_disp = fps_in

        while not gui.stopped:
            if gui.paused: time.sleep(0.04); continue
            ret, frame = cap.read()
            if not ret: break
            self.frame_num += 1

            t_now = time.perf_counter()
            fps_disp = 1.0/max(t_now-t_prev,1e-6); t_prev = t_now

            # ── Vehicle detection (every N frames) ───────────────────
            if self.frame_num % self.detect_every == 0 or self.frame_num == 1:
                self._last_dets = self._detect_vehicles(frame)
            dets = self._last_dets

            active_tracks = self.tracker.update(dets, self.frame_num)
            self.total_vehicles = max(self.total_vehicles, self.tracker.next_id - 1)

            # ── Plate detection: REAL model, run on full frame ───────
            ocr_budget = self.max_ocr_per_frame
            need_plate_pass = any(
                ocr_budget > 0 and
                (not t.confirmed_plate or
                 self.frame_num - t.last_ocr_frame >= self.ocr_recheck_interval)
                and self.frame_num - t.last_ocr_frame >= 8
                for t in active_tracks
            )
            frame_plates = self._detect_plates(frame) if need_plate_pass else []

            for track in active_tracks:
                x1,y1,x2,y2 = track.bbox
                draw_neon_box(frame, x1, y1, x2, y2, NEON_GREEN, thickness=2)
                draw_label(frame, f"[{track.class_name.upper()}#{track.vehicle_id:03d}] "
                                  f"{track.confidence:.0%}", x1, y1-6, NEON_GREEN)

                needs_ocr = (
                    ocr_budget > 0 and
                    (not track.confirmed_plate or
                     self.frame_num - track.last_ocr_frame >= self.ocr_recheck_interval)
                    and self.frame_num - track.last_ocr_frame >= 8
                )

                if needs_ocr and frame_plates:
                    match = next((p for p in frame_plates
                                  if self._plate_in_vehicle(p, track.bbox)), None)
                    if match:
                        px1,py1,px2,py2,pconf = match
                        pad_x = int((px2-px1)*0.08) + 3
                        pad_y = int((py2-py1)*0.15) + 3
                        cx1 = max(0, px1-pad_x); cy1 = max(0, py1-pad_y)
                        cx2 = min(frame.shape[1], px2+pad_x)
                        cy2 = min(frame.shape[0], py2+pad_y)
                        plate_roi = frame[cy1:cy2, cx1:cx2]

                        text, conf = self.ocr.read_plate(plate_roi)
                        track.last_ocr_frame = self.frame_num
                        ocr_budget -= 1

                        if text and len(text) >= 4:
                            if conf > track.confirmed_conf:
                                track.confirmed_plate = text
                                track.confirmed_conf  = conf
                            self.plate_records.append(PlateRecord(
                                plate_text=text, confidence=conf,
                                vehicle_id=track.vehicle_id,
                                timestamp=datetime.datetime.now().isoformat(),
                                frame_number=self.frame_num
                            ))
                            self.plate_counts[text] += 1
                            track.plates.append(text)
                        draw_neon_box(frame, px1, py1, px2, py2, NEON_YELLOW,
                                     thickness=2, glow=False)
                        draw_label(frame, f"PLATE {pconf:.0%}", px1, py1-4,
                                  NEON_YELLOW, fs=0.4)

                if track.confirmed_plate:
                    draw_plate_panel(frame, track.confirmed_plate, track.confirmed_conf,
                                     x1, y2+5, track.vehicle_id, confirmed=True)
                elif track.plates:
                    draw_plate_panel(frame, track.plates[-1], 0.0,
                                     x1, y2+5, track.vehicle_id, confirmed=False)

            stats = self._stats()
            draw_dashboard(frame, stats, fps_disp, self.frame_num)

            if self.frame_num % screenshot_interval == 0:
                self._screenshot(frame)

            writer.write(frame)
            gui.push_frame(frame)
            gui.set_status(
                f"▶  F:{self.frame_num}  Vehicles:{self.total_vehicles}  "
                f"Plates:{len(self.plate_records)}  FPS:{fps_disp:.1f}"
            )
            pbar.update(1)
            pbar.set_postfix({"fps":f"{fps_disp:.1f}","veh":self.total_vehicles,
                              "plates":len(self.plate_records)}, refresh=True)

        pbar.close(); cap.release(); writer.release()
        self._save_csv()
        print(f"{Fore.GREEN}[DONE] Output → {out_path}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[DONE] Vehicles: {self.total_vehicles}  "
              f"Plates read: {len(self.plate_records)}  "
              f"Unique: {len(self.plate_counts)}{Style.RESET_ALL}")
        gui.set_status(f"✔ Done — Vehicles:{self.total_vehicles}  "
                       f"Plates:{len(self.plate_records)}  [Q] exit")
        gui.stop()

    def run(self):
        print(BANNER)
        gui = ANPRGui(dw=self.dw, dh=self.dh)
        t = threading.Thread(target=self._process, args=(gui,), daemon=True)
        t.start(); gui.start(); t.join(timeout=5)

# ── Entry Point ────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="License Plate Intelligence v4.0 — tubakhxn")
    ap.add_argument("source", nargs="?", default="0",
                    help="Video file or camera index (default: 0)")
    ap.add_argument("--every", type=int, default=2,
                    help="Run vehicle detection every N frames (default: 2)")
    ap.add_argument("--half-res", action="store_true",
                    help="Use half resolution for vehicle detection (faster, "
                         "lower confidence -- not recommended for small/far vehicles)")
    ap.add_argument("--ocr-recheck", type=int, default=45,
                    help="Re-attempt OCR on a confirmed plate every N frames (default: 45)")
    ap.add_argument("--max-ocr", type=int, default=3,
                    help="Max OCR calls per frame, caps slowdown on busy scenes (default: 3)")
    ap.add_argument("--plate-conf", type=float, default=0.25,
                    help="Confidence threshold for plate detector (default: 0.25)")
    ap.add_argument("--width",  type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    LicensePlateSystem(
        source=args.source,
        detect_every=args.every,
        half_res=args.half_res,
        ocr_recheck_interval=args.ocr_recheck,
        max_ocr_per_frame=args.max_ocr,
        plate_conf=args.plate_conf,
        dw=args.width, dh=args.height,
    ).run()

if __name__ == "__main__":
    main()
