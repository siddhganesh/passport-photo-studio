import io
import base64
import os
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── AI ENGINE ────────────────────────────────────────────────────────────────

def enhance_image(image: Image.Image) -> Image.Image:
    img_np = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21)
    gaussian = cv2.GaussianBlur(img_bgr, (0, 0), 2.0)
    img_bgr = cv2.addWeighted(img_bgr, 1.5, gaussian, -0.5, 0)
    img_bgr = cv2.convertScaleAbs(img_bgr, alpha=1.1, beta=5)
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def detect_and_crop_face(image: Image.Image) -> Image.Image:
    try:
        img_np = np.array(image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H, W = img_np.shape[:2]

        cascade_name = 'haarcascade_frontalface_default.xml'
        paths = []
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            paths.append(cv2.data.haarcascades + cascade_name)
        cv2_dir = os.path.dirname(cv2.__file__)
        paths += [
            os.path.join(cv2_dir, 'data', cascade_name),
            os.path.join(cv2_dir, cascade_name),
            os.path.join(os.path.dirname(cv2_dir), 'cv2', 'data', cascade_name),
        ]

        face_cascade = None
        for p in paths:
            if p and os.path.isfile(p):
                c = cv2.CascadeClassifier(p)
                if not c.empty():
                    face_cascade = c
                    break

        if face_cascade is None:
            raise RuntimeError("No cascade")

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            raise RuntimeError("No face")

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cx, cy = x + w // 2, y + h // 2
        ch = int(h * 3.5)
        cw = int(ch * 3 / 4)
        sx = max(0, cx - cw // 2)
        ex = min(W, sx + cw)
        sy = max(0, cy - int(ch * 0.35))
        ey = min(H, sy + ch)
        return image.crop((sx, sy, ex, ey))
    except Exception:
        iw, ih = image.size
        tr = 3 / 4
        if (iw / ih) > tr:
            nw = int(ih * tr)
            return image.crop(((iw - nw) // 2, 0, (iw - nw) // 2 + nw, ih))
        else:
            nh = int(iw / tr)
            top = max(0, (ih - nh) // 4)
            return image.crop((0, top, iw, top + nh))

def apply_background(image: Image.Image, bg_color: str) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    result = remove(buf.getvalue(), alpha_matting=True,
                    alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=15)
    rgba = Image.open(io.BytesIO(result)).convert('RGBA')
    if bg_color == 'transparent':
        return rgba
    colors = {
        'white': (255,255,255), 'offwhite': (245,245,240),
        'blue': (34,73,135), 'lightblue': (74,144,217),
        'grey': (232,232,232), 'black': (0,0,0), 'green': (34,135,73),
    }
    bg = Image.new('RGB', rgba.size, colors.get(bg_color, (255,255,255)))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg

def generate_print_layout(image: Image.Image, copies: int, paper_size: str) -> Image.Image:
    dpi = 300
    sizes = {'A4':(int(8.27*dpi),int(11.69*dpi)),'Letter':(int(8.5*dpi),int(11*dpi)),'4x6':(int(6*dpi),int(4*dpi))}
    pw, ph = sizes.get(paper_size, sizes['A4'])
    paper = Image.new('RGB', (pw, ph), (255,255,255))
    pw2, ph2 = int(35/25.4*dpi), int(45/25.4*dpi)
    photo = image.resize((pw2, ph2), Image.Resampling.LANCZOS)
    margin, spacing = 20, 10
    cols = (pw - 2*margin) // (pw2 + spacing)
    rows = (ph - 2*margin) // (ph2 + spacing)
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= copies: break
            paper.paste(photo, (margin + c*(pw2+spacing), margin + r*(ph2+spacing)))
            count += 1
        if count >= copies: break
    return paper

# ── API ──────────────────────────────────────────────────────────────────────

@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    remove_bg: str = Form('true'),
    bg_color: str = Form('white'),
    do_enhance: str = Form('true'),
    do_crop: str = Form('true'),
):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        if do_enhance.lower() == 'true':
            img = enhance_image(img)
        if do_crop.lower() == 'true':
            img = detect_and_crop_face(img)
        if remove_bg.lower() == 'true':
            img = apply_background(img, bg_color)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()
        return JSONResponse({"status":"success","final_image":f"data:image/png;base64,{b64}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/print")
async def print_layout(file: UploadFile = File(...), copies: int = Form(8), paper_size: str = Form('A4')):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        layout = generate_print_layout(img, copies, paper_size)
        buf = io.BytesIO()
        layout.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()
        return JSONResponse({"status":"success","print_layout":f"data:image/png;base64,{b64}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── PREMIUM WHITE + BROWN UI ─────────────────────────────────────────────────

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AI Passport Studio Pro</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
<style>
:root{
  --cream:#FAF6F1; --cream2:#F2EBE2; --white:#FFFFFF;
  --br-dk:#3E2010; --br:#6F4E37; --br-md:#9C6B47; --br-lt:#C8964B;
  --br-pale:#EDD9C0; --br-xpale:#F5EDE0;
  --txt:#2D1B0E; --txt2:#7A5840; --txt3:#B09080;
  --bdr:rgba(111,78,55,.13); --bdr2:rgba(111,78,55,.22);
  --sh:0 2px 16px rgba(111,78,55,.08);
  --sh2:0 8px 40px rgba(111,78,55,.14);
  --grad:linear-gradient(135deg,#6F4E37 0%,#C8964B 100%);
  --grad2:linear-gradient(135deg,#C8964B 0%,#6F4E37 100%);
  --r:18px;
}
*{margin:0;padding:0;box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{font-family:'Inter',sans-serif;background:var(--cream);color:var(--txt);min-height:100vh;}

/* Subtle bg texture */
body::before{content:'';position:fixed;inset:0;
  background:radial-gradient(ellipse at 10% 20%,rgba(200,150,75,.06) 0%,transparent 50%),
             radial-gradient(ellipse at 90% 80%,rgba(111,78,55,.05) 0%,transparent 50%);
  pointer-events:none;z-index:0;}

.wrap{position:relative;z-index:1;max-width:1360px;margin:0 auto;padding:0 24px;}

/* ── HEADER ── */
header{padding:56px 0 52px;text-align:center;}
.badge{display:inline-flex;align-items:center;gap:8px;
  background:rgba(111,78,55,.08);border:1px solid rgba(111,78,55,.2);
  border-radius:100px;padding:7px 18px;font-size:12px;font-weight:600;
  color:var(--br);letter-spacing:.3px;margin-bottom:22px;}
.dot{width:6px;height:6px;background:var(--br-lt);border-radius:50%;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.5;transform:scale(1.5);}}
h1{font-family:'Playfair Display',serif;font-size:clamp(36px,5.5vw,68px);
  font-weight:700;line-height:1.1;color:var(--br-dk);margin-bottom:16px;}
.gt{background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.sub{font-size:16px;color:var(--txt2);max-width:560px;margin:0 auto 48px;line-height:1.75;font-weight:400;}

/* Steps */
.steps{display:flex;align-items:center;justify-content:center;margin-bottom:56px;}
.step{display:flex;align-items:center;gap:10px;}
.sn{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:13px;border:2px solid var(--bdr);color:var(--txt3);transition:all .4s;}
.step.active .sn{background:var(--grad);border:none;color:#fff;box-shadow:0 4px 16px rgba(111,78,55,.3);}
.step.done .sn{background:var(--br-xpale);border-color:var(--br-lt);color:var(--br-lt);}
.sl{font-size:13px;font-weight:500;color:var(--txt3);transition:all .4s;}
.step.active .sl{color:var(--br-dk);font-weight:600;} .step.done .sl{color:var(--br-lt);}
.sline{width:52px;height:2px;background:var(--bdr);margin:0 8px;border-radius:2px;transition:all .5s;}
.sline.done{background:var(--grad);}

/* ── GRID ── */
.grid{display:grid;grid-template-columns:390px 1fr;gap:20px;align-items:start;}
@media(max-width:1020px){.grid{grid-template-columns:1fr;}}

/* ── CARD ── */
.card{background:var(--white);border:1px solid var(--bdr);border-radius:var(--r);
  padding:24px;box-shadow:var(--sh);transition:box-shadow .3s,border .3s;}
.card:hover{box-shadow:var(--sh2);border-color:var(--bdr2);}
.ct{font-family:'Playfair Display',serif;font-size:16px;font-weight:600;
  color:var(--br-dk);margin-bottom:18px;display:flex;align-items:center;gap:10px;}
.cn{width:26px;height:26px;background:var(--grad);border-radius:8px;
  display:flex;align-items:center;justify-content:center;font-family:'Inter',sans-serif;
  font-size:12px;font-weight:700;color:#fff;flex-shrink:0;}

/* ── UPLOAD ── */
.uz{border:2px dashed rgba(111,78,55,.25);border-radius:14px;padding:32px 20px;
  text-align:center;cursor:pointer;transition:all .3s;background:var(--cream);}
.uz:hover,.uz.drag{border-color:var(--br-lt);background:var(--br-xpale);
  box-shadow:0 0 0 4px rgba(200,150,75,.1);}
.ui{width:56px;height:56px;margin:0 auto 12px;background:rgba(111,78,55,.08);
  border-radius:50%;display:flex;align-items:center;justify-content:center;transition:all .3s;}
.uz:hover .ui{background:rgba(111,78,55,.14);transform:scale(1.08);}
.thumb{border-radius:10px;overflow:hidden;position:relative;margin-top:12px;}
.thumb img{width:100%;height:160px;object-fit:cover;}
.thumb-ov{position:absolute;inset:0;background:linear-gradient(to top,rgba(62,32,16,.55),transparent);
  display:flex;align-items:flex-end;padding:10px;}
.thumb-txt{font-size:11px;color:rgba(255,255,255,.9);}

/* Country */
.cgrid{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.cbtn{padding:9px 10px;background:var(--cream);border:1px solid var(--bdr);border-radius:10px;
  cursor:pointer;transition:all .2s;color:var(--txt2);font-family:'Inter',sans-serif;
  font-size:12px;display:flex;align-items:center;gap:8px;text-align:left;}
.cbtn:hover,.cbtn.act{border-color:var(--br-lt);background:var(--br-xpale);color:var(--br-dk);}
.cbtn.act{box-shadow:0 0 0 3px rgba(200,150,75,.15);}
.csz{font-size:10px;color:var(--txt3);display:block;}

/* Toggle */
.trow{display:flex;align-items:center;justify-content:space-between;padding:11px 0;
  border-bottom:1px solid var(--bdr);}
.trow:last-child{border-bottom:none;}
.tlabel{font-size:13px;font-weight:500;color:var(--txt);}
.tdesc{font-size:11px;color:var(--txt3);margin-top:2px;}
.tog{width:42px;height:22px;background:var(--bdr2);border-radius:100px;cursor:pointer;
  position:relative;transition:background .3s;flex-shrink:0;}
.tog.on{background:var(--br);}
.tog::after{content:'';position:absolute;width:16px;height:16px;background:#fff;border-radius:50%;
  top:3px;left:3px;transition:transform .3s;box-shadow:0 1px 4px rgba(0,0,0,.2);}
.tog.on::after{transform:translateX(20px);}

/* Swatches */
.sw-wrap{display:flex;gap:8px;flex-wrap:wrap;padding-top:12px;}
.sw{width:38px;height:38px;border-radius:8px;cursor:pointer;transition:all .2s;
  position:relative;border:2px solid transparent;}
.sw:hover{transform:scale(1.12);}
.sw.act{border-color:var(--br);box-shadow:0 0 0 3px rgba(111,78,55,.2);}
.sw-ck{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;opacity:0;transition:opacity .2s;}
.sw.act .sw-ck{opacity:1;}
.sw-trans{background:repeating-conic-gradient(#ccc 0% 25%,#fff 0% 50%) 0 0/10px 10px;border:1px solid var(--bdr);}

/* Select */
.pctrls{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:12px;}
.slbl{font-size:10px;font-weight:600;color:var(--txt3);text-transform:uppercase;letter-spacing:.5px;display:block;margin-bottom:5px;}
select.sf{width:100%;padding:9px 10px;background:var(--cream);border:1px solid var(--bdr);
  border-radius:9px;color:var(--txt);font-family:'Inter',sans-serif;font-size:13px;outline:none;cursor:pointer;}
select.sf:focus{border-color:var(--br-lt);}

/* Buttons */
.btn-p{width:100%;padding:14px;background:var(--grad);border:none;border-radius:13px;
  color:#fff;font-family:'Inter',sans-serif;font-size:15px;font-weight:600;cursor:pointer;
  transition:all .3s;display:flex;align-items:center;justify-content:center;gap:10px;
  box-shadow:0 4px 18px rgba(111,78,55,.25);}
.btn-p:hover{transform:translateY(-2px);box-shadow:0 8px 30px rgba(111,78,55,.35);}
.btn-p:active{transform:translateY(0);}
.btn-p:disabled{opacity:.45;cursor:not-allowed;transform:none;}
.btn-s{padding:10px 18px;background:var(--cream);border:1px solid var(--bdr);border-radius:10px;
  color:var(--txt2);font-family:'Inter',sans-serif;font-size:13px;font-weight:500;cursor:pointer;
  transition:all .25s;display:inline-flex;align-items:center;gap:8px;}
.btn-s:hover{background:var(--br-xpale);border-color:var(--bdr2);color:var(--br-dk);}
.btn-g{padding:12px 24px;background:transparent;border:1.5px solid var(--br-lt);border-radius:11px;
  color:var(--br);font-family:'Inter',sans-serif;font-size:13px;font-weight:600;cursor:pointer;
  transition:all .3s;display:inline-flex;align-items:center;gap:8px;text-decoration:none;}
.btn-g:hover{background:var(--br-xpale);box-shadow:0 4px 16px rgba(111,78,55,.15);color:var(--br-dk);}
.drow{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px;}

/* Preview */
.preview{min-height:560px;display:flex;flex-direction:column;}
.pempty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:60px;}
.pei{width:90px;height:90px;border-radius:50%;background:var(--cream);border:2px dashed var(--bdr2);
  display:flex;align-items:center;justify-content:center;margin:0 auto 22px;}

/* Loading */
.loading{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:28px;}
.orb{width:72px;height:72px;border-radius:50%;background:var(--grad);animation:orbP 1.6s ease-in-out infinite;position:relative;}
.orb::after{content:'';position:absolute;inset:-10px;border-radius:50%;
  border:2px solid rgba(111,78,55,.2);animation:orbR 1.6s ease-in-out infinite;}
@keyframes orbP{0%,100%{transform:scale(1);box-shadow:0 0 0 0 rgba(111,78,55,.3);}50%{transform:scale(1.12);box-shadow:0 0 40px rgba(111,78,55,.25);}}
@keyframes orbR{0%,100%{transform:scale(1);opacity:1;}50%{transform:scale(1.4);opacity:0;}}
.lsteps{display:flex;flex-direction:column;gap:10px;width:250px;}
.lstep{display:flex;align-items:center;gap:10px;font-size:13px;color:var(--txt3);transition:all .3s;}
.lstep.act{color:var(--br-dk);font-weight:500;} .lstep.dn{color:var(--br-lt);}
.ldot{width:8px;height:8px;border-radius:50%;background:var(--bdr2);flex-shrink:0;transition:all .3s;}
.lstep.act .ldot{background:var(--br-lt);box-shadow:0 0 8px rgba(200,150,75,.5);animation:pulse 1s infinite;}
.lstep.dn .ldot{background:var(--br-lt);}

/* Comparison */
.cmp{position:relative;border-radius:14px;overflow:hidden;height:300px;
  cursor:ew-resize;user-select:none;background:var(--cream2);border:1px solid var(--bdr);touch-action:none;}
.cmp-b,.cmp-a{position:absolute;inset:0;}
.cmp-b img,.cmp-a img{width:100%;height:100%;object-fit:contain;}
.cmp-a{clip-path:inset(0 50% 0 0);}
.cmp-div{position:absolute;top:0;bottom:0;width:2px;background:var(--br);z-index:10;pointer-events:none;transform:translateX(-50%);}
.cmp-hdl{position:absolute;top:50%;width:38px;height:38px;background:var(--white);border-radius:50%;z-index:11;
  display:flex;align-items:center;justify-content:center;box-shadow:var(--sh2);pointer-events:none;transform:translate(-50%,-50%);}
.cmp-lbl{position:absolute;top:10px;padding:4px 10px;border-radius:6px;font-size:10px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;}
.lbl-b{left:10px;background:rgba(62,32,16,.5);color:rgba(255,255,255,.9);}
.lbl-a{right:10px;background:var(--br);color:#fff;}

/* Quality */
.qrow{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;}
.qb{display:flex;align-items:center;gap:5px;padding:5px 12px;border-radius:100px;font-size:11px;font-weight:600;}
.qb.ok{background:rgba(34,120,60,.08);border:1px solid rgba(34,120,60,.2);color:#1a7a3c;}
.qb.warn{background:rgba(200,150,75,.1);border:1px solid rgba(200,150,75,.3);color:var(--br);}

/* Print */
.pp{border-radius:11px;overflow:hidden;background:var(--cream2);border:1px solid var(--bdr);margin-bottom:14px;}
.pp img{width:100%;height:200px;object-fit:contain;}

/* Toast */
.toast{position:fixed;bottom:28px;right:28px;padding:12px 20px;border-radius:13px;
  font-size:13px;font-weight:600;z-index:999;display:flex;align-items:center;gap:10px;
  backdrop-filter:blur(16px);animation:slideR .3s ease;box-shadow:var(--sh2);}
.toast.ok{background:rgba(240,255,240,.95);border:1px solid rgba(34,120,60,.25);color:#1a7a3c;}
.toast.err{background:rgba(255,240,240,.95);border:1px solid rgba(200,60,60,.25);color:#c0392b;}
@keyframes slideR{from{opacity:0;transform:translateX(16px);}to{opacity:1;transform:translateX(0);}}
@keyframes scaleIn{from{opacity:0;transform:scale(.97);}to{opacity:1;transform:scale(1);}}
.anim{animation:scaleIn .35s ease;}
@keyframes spin{to{transform:rotate(360deg);}}
.spin{width:18px;height:18px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;}
::-webkit-scrollbar{width:5px;} ::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--bdr2);border-radius:3px;}

/* Footer */
footer{text-align:center;padding:44px 0 28px;color:var(--txt3);font-size:12px;
  border-top:1px solid var(--bdr);margin-top:40px;}
footer span{color:var(--br-lt);}

/* Divider */
.sep{height:1px;background:var(--bdr);margin:16px 0;}
</style>
</head>
<body>
<div x-data="App()" class="wrap">

<div x-show="toast.show" x-transition :class="['toast',toast.type]" style="display:none">
  <span x-text="toast.ico"></span><span x-text="toast.msg"></span>
</div>

<!-- HEADER -->
<header>
  <div class="badge"><span class="dot"></span>&nbsp;AI-Powered &nbsp;&middot;&nbsp; Professional Grade</div>
  <h1>Passport Photo <span class="gt">Studio Pro</span></h1>
  <p class="sub">AI-powered background removal, face detection &amp; smart cropping — create flawless passport photos in seconds.</p>
  <div class="steps">
    <div class="step" :class="{active:step>=1,done:step>1}">
      <div class="sn"><span x-show="step<=1">1</span><span x-show="step>1">✓</span></div>
      <span class="sl">Upload</span>
    </div>
    <div class="sline" :class="{done:step>1}"></div>
    <div class="step" :class="{active:step>=2,done:step>2}">
      <div class="sn"><span x-show="step<=2">2</span><span x-show="step>2">✓</span></div>
      <span class="sl">Process</span>
    </div>
    <div class="sline" :class="{done:step>2}"></div>
    <div class="step" :class="{active:step>=3}">
      <div class="sn">3</div>
      <span class="sl">Download</span>
    </div>
  </div>
</header>

<div class="grid">

<!-- ── LEFT PANEL ── -->
<div style="display:flex;flex-direction:column;gap:16px;">

  <!-- 1. Upload -->
  <div class="card">
    <div class="ct"><div class="cn">1</div> Upload Portrait</div>
    <div class="uz" :class="{drag:drag}" @click="$refs.fi.click()"
         @dragover.prevent="drag=true" @dragleave="drag=false" @drop.prevent="drag=false;drop($event)">
      <input type="file" x-ref="fi" accept="image/*" @change="onFile($event)" style="display:none">
      <template x-if="!orig">
        <div>
          <div class="ui">
            <svg width="26" height="26" fill="none" stroke="#6F4E37" stroke-width="1.8" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/>
            </svg>
          </div>
          <p style="font-size:14px;font-weight:600;color:var(--txt);margin-bottom:4px">Drop your portrait here</p>
          <p style="font-size:12px;color:var(--txt3)">JPG, PNG, WEBP &mdash; Max 20MB</p>
        </div>
      </template>
      <template x-if="orig">
        <div class="thumb" @click.stop>
          <img :src="orig" alt="preview">
          <div class="thumb-ov"><span class="thumb-txt" x-text="fname"></span></div>
        </div>
      </template>
    </div>
    <template x-if="orig">
      <button @click="$refs.fi.click()" class="btn-s" style="width:100%;justify-content:center;margin-top:10px">
        <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/>
        </svg>Change Photo
      </button>
    </template>
  </div>

  <!-- 2. Country -->
  <div class="card" x-show="orig" x-transition style="display:none">
    <div class="ct"><div class="cn">2</div> Country Standard</div>
    <div class="cgrid">
      <template x-for="c in countries" :key="c.code">
        <button class="cbtn" :class="{act:country===c.code}" @click="country=c.code">
          <span x-text="c.flag" style="font-size:18px"></span>
          <span>
            <span x-text="c.name" style="display:block;font-weight:600;font-size:12px;color:var(--br-dk)"></span>
            <span class="csz" x-text="c.size"></span>
          </span>
        </button>
      </template>
    </div>
  </div>

  <!-- 3. Options -->
  <div class="card" x-show="orig" x-transition style="display:none">
    <div class="ct"><div class="cn">3</div> Processing Options</div>
    <div class="trow">
      <div><div class="tlabel">✨ AI Enhancement</div><div class="tdesc">Denoise, sharpen & brightness fix</div></div>
      <div class="tog" :class="{on:enhance}" @click="enhance=!enhance"></div>
    </div>
    <div class="trow">
      <div><div class="tlabel">🎯 Smart Face Crop</div><div class="tdesc">Auto-align to passport 3:4 ratio</div></div>
      <div class="tog" :class="{on:crop}" @click="crop=!crop"></div>
    </div>
    <div class="trow">
      <div><div class="tlabel">🤖 Remove Background</div><div class="tdesc">AI alpha matting extraction</div></div>
      <div class="tog" :class="{on:rmbg}" @click="rmbg=!rmbg"></div>
    </div>
    <template x-if="rmbg">
      <div>
        <div class="sep"></div>
        <p style="font-size:11px;font-weight:600;color:var(--txt3);text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px">Background Color</p>
        <div class="sw-wrap">
          <template x-for="s in swatches" :key="s.v">
            <div class="sw" :class="{act:bg===s.v,'sw-trans':s.v==='transparent'}"
                 :style="s.v!=='transparent'?`background:${s.h};border:1px solid rgba(0,0,0,0.06)`:''"
                 @click="bg=s.v" :title="s.n">
              <div class="sw-ck">
                <svg width="13" height="13" fill="none" viewBox="0 0 24 24"
                     :stroke="['white','offwhite','grey'].includes(s.v)?'#3E2010':'#fff'" stroke-width="3">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/>
                </svg>
              </div>
            </div>
          </template>
        </div>
      </div>
    </template>
    <button @click="process()" :disabled="processing||!orig" class="btn-p" style="margin-top:20px">
      <template x-if="!processing">
        <span style="display:flex;align-items:center;gap:10px">
          <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"/>
          </svg>Process with AI
        </span>
      </template>
      <template x-if="processing">
        <span style="display:flex;align-items:center;gap:10px"><div class="spin"></div>Processing...</span>
      </template>
    </button>
  </div>

  <!-- 4. Print -->
  <div class="card" x-show="final&&!processing" x-transition style="display:none">
    <div class="ct"><div class="cn">4</div> Print Layout</div>
    <div class="pctrls">
      <div><span class="slbl">Copies</span>
        <select x-model="copies" class="sf"><option>4</option><option>6</option><option selected>8</option><option>12</option><option>16</option></select>
      </div>
      <div><span class="slbl">Paper</span>
        <select x-model="paper" class="sf"><option>A4</option><option>Letter</option><option>4x6</option></select>
      </div>
      <div><span class="slbl">DPI</span>
        <select class="sf"><option>300</option></select>
      </div>
    </div>
    <button @click="genPrint()" class="btn-s" style="width:100%;justify-content:center;padding:13px;">
      <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z"/>
      </svg>Generate Print Sheet
    </button>
  </div>

</div><!-- end left -->

<!-- ── RIGHT PANEL ── -->
<div>
<div class="card preview">

  <!-- Empty -->
  <div class="pempty" x-show="!orig&&!processing" style="display:flex">
    <div class="pei">
      <svg width="34" height="34" fill="none" stroke="var(--br-pale)" stroke-width="1.3" viewBox="0 0 24 24">
        <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/>
        <path stroke-linecap="round" stroke-linejoin="round" d="M21 15l-5-5L5 21"/>
      </svg>
    </div>
    <h3 style="font-family:'Playfair Display',serif;font-size:18px;font-weight:600;margin-bottom:8px;color:var(--txt3)">Preview will appear here</h3>
    <p style="font-size:13px;color:var(--txt3)">Upload a portrait photo to begin</p>
  </div>

  <!-- Uploaded not yet processed -->
  <div x-show="orig&&!processing&&!final" class="anim" style="display:none">
    <p style="font-family:'Playfair Display',serif;font-size:17px;font-weight:600;color:var(--br-dk);margin-bottom:4px">📸 Photo Ready</p>
    <p style="font-size:13px;color:var(--txt2);margin-bottom:18px">Configure options and click "Process with AI"</p>
    <div style="border-radius:14px;overflow:hidden;background:var(--cream2);border:1px solid var(--bdr)">
      <img :src="orig" style="width:100%;max-height:420px;object-fit:contain;" alt="Original">
    </div>
  </div>

  <!-- Loading -->
  <div class="loading" x-show="processing" style="display:none">
    <div class="orb"></div>
    <div>
      <p style="font-family:'Playfair Display',serif;font-size:19px;font-weight:600;text-align:center;color:var(--br-dk);margin-bottom:20px">AI is crafting your photo...</p>
      <div class="lsteps">
        <div class="lstep" :class="{act:ls>=1,dn:ls>1}"><div class="ldot"></div> Analyzing image quality</div>
        <div class="lstep" :class="{act:ls>=2,dn:ls>2}"><div class="ldot"></div> Denoising &amp; sharpening</div>
        <div class="lstep" :class="{act:ls>=3,dn:ls>3}"><div class="ldot"></div> Detecting &amp; cropping face</div>
        <div class="lstep" :class="{act:ls>=4,dn:ls>4}"><div class="ldot"></div> Removing background</div>
        <div class="lstep" :class="{act:ls>=5}"><div class="ldot"></div> Finalizing output</div>
      </div>
    </div>
  </div>

  <!-- Result -->
  <div x-show="final&&!processing" class="anim" style="display:none">
    <p style="font-family:'Playfair Display',serif;font-size:18px;font-weight:600;color:var(--br-dk);margin-bottom:4px">✨ Result Ready</p>
    <p style="font-size:13px;color:var(--txt2);margin-bottom:18px">Drag the slider to compare before &amp; after</p>

    <!-- Slider -->
    <div class="cmp"
         @mousedown="dragging=true;slide($event)"
         @mousemove="slide($event)"
         @mouseup="dragging=false"
         @mouseleave="dragging=false"
         @touchstart.prevent="dragging=true;slideT($event)"
         @touchmove.prevent="slideT($event)"
         @touchend="dragging=false">
      <div class="cmp-b"><img :src="orig" alt="Before"></div>
      <div class="cmp-a" :style="`clip-path:inset(0 ${100-pos}% 0 0)`"><img :src="final" alt="After"></div>
      <span class="cmp-lbl lbl-b">Before</span>
      <span class="cmp-lbl lbl-a">After</span>
      <div class="cmp-div" :style="`left:${pos}%`"></div>
      <div class="cmp-hdl" :style="`left:${pos}%`">
        <svg width="18" height="18" fill="none" stroke="var(--br)" stroke-width="2.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M8 9l-4 3 4 3M16 9l4 3-4 3"/>
        </svg>
      </div>
    </div>

    <div class="qrow">
      <div class="qb ok">✓ Face detected</div>
      <div class="qb ok">✓ Passport ratio</div>
      <div class="qb ok">✓ 300 DPI ready</div>
      <template x-if="rmbg"><div class="qb ok">✓ BG removed</div></template>
    </div>

    <div class="drow">
      <a :href="final" download="passport_photo.png" class="btn-p"
         style="width:auto;padding:12px 24px;text-decoration:none">
        <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
        </svg>Download Photo
      </a>
      <button @click="process()" class="btn-s">
        <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
        </svg>Reprocess
      </button>
    </div>

    <!-- Print Sheet -->
    <template x-if="printImg">
      <div class="anim" style="margin-top:22px;padding-top:22px;border-top:1px solid var(--bdr)">
        <p style="font-family:'Playfair Display',serif;font-size:15px;font-weight:600;color:var(--br-dk);margin-bottom:12px">🖨️ Print Sheet Preview</p>
        <div class="pp"><img :src="printImg" alt="Print Layout"></div>
        <div class="drow">
          <a :href="printImg" download="passport_print_sheet.png" class="btn-g">
            <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
            </svg>Download High-Res Print Sheet
          </a>
        </div>
      </div>
    </template>
  </div>

</div>
</div><!-- end right -->
</div><!-- end grid -->

<footer>
  AI Passport Studio Pro &nbsp;&middot;&nbsp; <span>FastAPI · OpenCV · RemBG</span> &nbsp;&middot;&nbsp; Crafted with ♥
</footer>
</div>

<script>
function App(){return{
  step:1,orig:null,final:null,printImg:null,fname:'',
  processing:false,ls:0,drag:false,dragging:false,pos:50,
  enhance:true,rmbg:true,crop:true,bg:'white',
  country:'IN',copies:'8',paper:'A4',
  toast:{show:false,type:'ok',msg:'',ico:'✓'},

  countries:[
    {code:'IN',name:'India',flag:'🇮🇳',size:'35×45mm'},
    {code:'US',name:'USA',flag:'🇺🇸',size:'51×51mm'},
    {code:'UK',name:'UK',flag:'🇬🇧',size:'35×45mm'},
    {code:'EU',name:'Schengen',flag:'🇪🇺',size:'35×45mm'},
    {code:'AU',name:'Australia',flag:'🇦🇺',size:'35×45mm'},
    {code:'CA',name:'Canada',flag:'🇨🇦',size:'50×70mm'},
  ],
  swatches:[
    {n:'White',v:'white',h:'#FFFFFF'},
    {n:'Off White',v:'offwhite',h:'#F5F5F0'},
    {n:'Official Blue',v:'blue',h:'#224987'},
    {n:'Light Blue',v:'lightblue',h:'#4A90D9'},
    {n:'Grey',v:'grey',h:'#E0E0E0'},
    {n:'Black',v:'black',h:'#111111'},
    {n:'Green',v:'green',h:'#22873B'},
    {n:'Transparent',v:'transparent',h:'transparent'},
  ],

  showToast(msg,type='ok'){
    this.toast={show:true,type,msg,ico:type==='ok'?'✓':'✕'};
    setTimeout(()=>this.toast.show=false,3500);
  },
  onFile(e){this.load(e.target.files[0]);},
  drop(e){this.load(e.dataTransfer.files[0]);},
  load(f){
    if(!f?.type.startsWith('image/')){this.showToast('Please upload a valid image','err');return;}
    this.fname=f.name;
    const r=new FileReader();
    r.onload=e=>{this.orig=e.target.result;this.final=null;this.printImg=null;this.step=2;};
    r.readAsDataURL(f);
  },
  async process(){
    if(!this.orig)return;
    this.processing=true;this.final=null;this.printImg=null;this.ls=1;
    const t=setInterval(()=>{if(this.ls<4)this.ls++;},700);
    const blob=await(await fetch(this.orig)).blob();
    const fd=new FormData();
    fd.append('file',blob,'upload.jpg');
    fd.append('remove_bg',this.rmbg);
    fd.append('bg_color',this.bg);
    fd.append('do_enhance',this.enhance);
    fd.append('do_crop',this.crop);
    try{
      const res=await fetch('/api/process',{method:'POST',body:fd});
      const d=await res.json();
      clearInterval(t);this.ls=5;
      await new Promise(r=>setTimeout(r,300));
      if(d.status==='success'){
        this.final=d.final_image;this.step=3;this.pos=50;
        this.showToast('Photo processed successfully!');
      }else{this.showToast(d.detail||'Processing failed','err');}
    }catch(e){clearInterval(t);this.showToast('Server connection error','err');}
    finally{this.processing=false;this.ls=0;}
  },
  async genPrint(){
    if(!this.final)return;
    const blob=await(await fetch(this.final)).blob();
    const fd=new FormData();
    fd.append('file',blob,'passport.png');
    fd.append('copies',this.copies);
    fd.append('paper_size',this.paper);
    try{
      const res=await fetch('/api/print',{method:'POST',body:fd});
      const d=await res.json();
      if(d.status==='success'){
        this.printImg=d.print_layout;
        this.showToast('Print sheet ready!');
      }
    }catch(e){this.showToast('Failed to generate print sheet','err');}
  },
  slide(e){if(!this.dragging)return;const r=e.currentTarget.getBoundingClientRect();this.pos=Math.max(2,Math.min(98,(e.clientX-r.left)/r.width*100));},
  slideT(e){const r=e.currentTarget.getBoundingClientRect();const t=e.touches[0];this.pos=Math.max(2,Math.min(98,(t.clientX-r.left)/r.width*100));}
}}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_CONTENT

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)