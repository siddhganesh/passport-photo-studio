import io
import base64
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Passport Photo Studio Pro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── BACKEND IMAGE PROCESSING ENGINE ──────────────────────────────────────────

def enhance_image(image: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    img = enhancer.enhance(1.15)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)
    return img

def detect_and_crop_face(image: Image.Image) -> Image.Image:
    try:
        img_np = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        H, W = img_np.shape[:2]
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                cx, cy = x + w // 2, y + h // 2
                ch = int(h * 3.4)
                cw = int(ch * 3 / 4)
                sx = max(0, cx - cw // 2)
                ex = min(W, sx + cw)
                sy = max(0, cy - int(ch * 0.35))
                ey = min(H, sy + ch)
                return image.crop((sx, sy, ex, ey))
    except Exception:
        pass

    iw, ih = image.size
    tr = 3 / 4
    if (iw / ih) > tr:
        nw = int(ih * tr)
        return image.crop(((iw - nw) // 2, 0, (iw - nw) // 2 + nw, ih))
    else:
        nh = int(iw / tr)
        top = max(0, (ih - nh) // 4)
        return image.crop((0, top, iw, top + nh))

def apply_background_fast(image: Image.Image, bg_color: str) -> Image.Image:
    img_np = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    H, W = img_np.shape[:2]

    mask = np.zeros((H, W), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (int(W * 0.05), int(H * 0.05), int(W * 0.9), int(H * 0.9))
    
    try:
        cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        mask_blur = cv2.GaussianBlur(mask2 * 255, (5, 5), 0)
        alpha = mask_blur.astype(float) / 255.0
    except Exception:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        alpha = cv2.GaussianBlur(thresh, (5, 5), 0).astype(float) / 255.0

    if bg_color == 'transparent':
        rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))

    colors = {
        'white': (255, 255, 255), 'offwhite': (245, 245, 240),
        'blue': (34, 73, 135), 'lightblue': (74, 144, 217),
        'grey': (232, 232, 232), 'black': (0, 0, 0), 'green': (34, 135, 73),
    }
    bg_rgb = colors.get(bg_color, (255, 255, 255))
    
    result = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(3):
        result[:, :, i] = alpha * img_np[:, :, i] + (1 - alpha) * bg_rgb[i]
        
    return Image.fromarray(result)

def generate_print_layout(image: Image.Image, copies: int, paper_size: str) -> Image.Image:
    dpi = 300
    sizes = {'A4': (int(8.27 * dpi), int(11.69 * dpi)), 'Letter': (int(8.5 * dpi), int(11 * dpi)), '4x6': (int(6 * dpi), int(4 * dpi))}
    pw, ph = sizes.get(paper_size, sizes['A4'])
    paper = Image.new('RGB', (pw, ph), (255, 255, 255))
    pw2, ph2 = int(35 / 25.4 * dpi), int(45 / 25.4 * dpi)
    photo = image.resize((pw2, ph2), Image.Resampling.LANCZOS)
    margin, spacing = 20, 10
    cols = (pw - 2 * margin) // (pw2 + spacing)
    rows = (ph - 2 * margin) // (ph2 + spacing)
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= copies: break
            paper.paste(photo, (margin + c * (pw2 + spacing), margin + r * (ph2 + spacing)))
            count += 1
        if count >= copies: break
    return paper

# ── API ENDPOINTS ─────────────────────────────────────────────────────────────

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
            img = apply_background_fast(img, bg_color)
        
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return JSONResponse({"status": "success", "final_image": f"data:image/jpeg;base64,{b64}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/print")
async def print_layout(file: UploadFile = File(...), copies: int = Form(8), paper_size: str = Form('A4')):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        layout = generate_print_layout(img, copies, paper_size)
        buf = io.BytesIO()
        layout.save(buf, format='JPEG', quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return JSONResponse({"status": "success", "print_layout": f"data:image/jpeg;base64,{b64}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── FRONTEND HTML UI ──────────────────────────────────────────────────────────

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
  --r:18px;
}
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Inter',sans-serif;background:var(--cream);color:var(--txt);min-height:100vh;}
.wrap{position:relative;z-index:1;max-width:1360px;margin:0 auto;padding:0 24px;}
header{padding:40px 0 32px;text-align:center;}
.badge{display:inline-flex;align-items:center;gap:8px;background:rgba(111,78,55,.08);border:1px solid rgba(111,78,55,.2);border-radius:100px;padding:7px 18px;font-size:12px;font-weight:600;color:var(--br);}
.dot{width:6px;height:6px;background:var(--br-lt);border-radius:50%;}
h1{font-family:'Playfair Display',serif;font-size:clamp(32px,4.5vw,56px);font-weight:700;color:var(--br-dk);margin-bottom:12px;}
.gt{background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sub{font-size:15px;color:var(--txt2);max-width:560px;margin:0 auto 32px;line-height:1.6;}
.grid{display:grid;grid-template-columns:390px 1fr;gap:20px;align-items:start;}
@media(max-width:1020px){.grid{grid-template-columns:1fr;}}
.card{background:var(--white);border:1px solid var(--bdr);border-radius:var(--r);padding:24px;box-shadow:var(--sh);}
.ct{font-family:'Playfair Display',serif;font-size:16px;font-weight:600;color:var(--br-dk);margin-bottom:18px;display:flex;align-items:center;gap:10px;}
.cn{width:26px;height:26px;background:var(--grad);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#fff;}
.uz{border:2px dashed rgba(111,78,55,.25);border-radius:14px;padding:28px 20px;text-align:center;cursor:pointer;background:var(--cream);}
.thumb img{width:100%;height:160px;object-fit:cover;border-radius:10px;}
.trow{display:flex;align-items:center;justify-content:space-between;padding:11px 0;border-bottom:1px solid var(--bdr);}
.trow:last-child{border-bottom:none;}
.tlabel{font-size:13px;font-weight:500;}
.tog{width:42px;height:22px;background:var(--bdr2);border-radius:100px;cursor:pointer;position:relative;transition:background .3s;}
.tog.on{background:var(--br);}
.tog::after{content:'';position:absolute;width:16px;height:16px;background:#fff;border-radius:50%;top:3px;left:3px;transition:transform .3s;}
.tog.on::after{transform:translateX(20px);}
.sw-wrap{display:flex;gap:8px;flex-wrap:wrap;padding-top:10px;}
.sw{width:34px;height:34px;border-radius:8px;cursor:pointer;border:2px solid transparent;}
.sw.act{border-color:var(--br);box-shadow:0 0 0 3px rgba(111,78,55,.2);}
.sw-trans{background:repeating-conic-gradient(#ccc 0% 25%,#fff 0% 50%) 0 0/10px 10px;border:1px solid var(--bdr);}
.btn-p{width:100%;padding:14px;background:var(--grad);border:none;border-radius:13px;color:#fff;font-size:15px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:10px;margin-top:16px;}
.btn-s{padding:10px 18px;background:var(--cream);border:1px solid var(--bdr);border-radius:10px;color:var(--txt2);font-size:13px;font-weight:500;cursor:pointer;display:inline-flex;align-items:center;gap:8px;}
.preview{min-height:520px;display:flex;flex-direction:column;}
.pempty{flex:1;display:flex;align-items:center;justify-content:center;}
.cmp{position:relative;border-radius:14px;overflow:hidden;height:320px;cursor:ew-resize;user-select:none;background:var(--cream2);border:1px solid var(--bdr);touch-action:none;}
.cmp-b,.cmp-a{position:absolute;inset:0;}
.cmp-b img,.cmp-a img{width:100%;height:100%;object-fit:contain;}
.cmp-a{clip-path:inset(0 50% 0 0);}
.cmp-div{position:absolute;top:0;bottom:0;width:2px;background:var(--br);z-index:10;transform:translateX(-50%);}
.cmp-hdl{position:absolute;top:50%;width:36px;height:36px;background:var(--white);border-radius:50%;z-index:11;display:flex;align-items:center;justify-content:center;box-shadow:var(--sh2);transform:translate(-50%,-50%);}
.cmp-lbl{position:absolute;top:10px;padding:4px 10px;border-radius:6px;font-size:10px;font-weight:700;text-transform:uppercase;}
.lbl-b{left:10px;background:rgba(62,32,16,.5);color:#fff;}
.lbl-a{right:10px;background:var(--br);color:#fff;}
.toast{position:fixed;bottom:24px;right:24px;padding:12px 20px;border-radius:13px;font-size:13px;font-weight:600;z-index:999;display:flex;align-items:center;gap:10px;}
.toast.ok{background:#E8F5E9;border:1px solid #A5D6A7;color:#2E7D32;}
.toast.err{background:#FFEBEE;border:1px solid #FFCDD2;color:#C62828;}
.spin{width:18px;height:18px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
footer{text-align:center;padding:36px 0 24px;color:var(--txt3);font-size:12px;margin-top:30px;}
</style>
</head>
<body>
<div x-data="App()" class="wrap">

<div x-show="toast.show" x-transition :class="['toast',toast.type]" style="display:none">
  <span x-text="toast.msg"></span>
</div>

<header>
  <div class="badge"><span class="dot"></span>&nbsp;AI Passport Photo Generator</div>
  <h1>Passport Photo <span class="gt">Studio Pro</span></h1>
  <p class="sub">AI Background Remover, Face Detector &amp; Print Sheet Generator</p>
</header>

<div class="grid">
<!-- LEFT -->
<div style="display:flex;flex-direction:column;gap:16px;">
  <div class="card">
    <div class="ct"><div class="cn">1</div> Upload Portrait</div>
    <div class="uz" @click="$refs.fi.click()">
      <input type="file" x-ref="fi" accept="image/*" @change="onFile($event)" style="display:none">
      <template x-if="!orig">
        <div>
          <p style="font-size:14px;font-weight:600;color:var(--txt)">Select Photo</p>
          <p style="font-size:11px;color:var(--txt3);margin-top:4px">JPG, PNG, WEBP</p>
        </div>
      </template>
      <template x-if="orig">
        <div class="thumb">
          <img :src="orig" alt="preview">
        </div>
      </template>
    </div>
  </div>

  <div class="card" x-show="orig">
    <div class="ct"><div class="cn">2</div> Options</div>
    <div class="trow">
      <div class="tlabel">✨ Auto Enhance</div>
      <div class="tog" :class="{on:enhance}" @click="enhance=!enhance"></div>
    </div>
    <div class="trow">
      <div class="tlabel">🎯 Smart Face Crop (3:4)</div>
      <div class="tog" :class="{on:crop}" @click="crop=!crop"></div>
    </div>
    <div class="trow">
      <div class="tlabel">🤖 Remove Background</div>
      <div class="tog" :class="{on:rmbg}" @click="rmbg=!rmbg"></div>
    </div>
    <template x-if="rmbg">
      <div style="margin-top:10px">
        <p style="font-size:11px;font-weight:600;color:var(--txt3)">BACKGROUND COLOR</p>
        <div class="sw-wrap">
          <template x-for="s in swatches" :key="s.v">
            <div class="sw" :class="{act:bg===s.v,'sw-trans':s.v==='transparent'}"
                 :style="s.v!=='transparent'?`background:${s.h}`:''"
                 @click="bg=s.v" :title="s.n"></div>
          </template>
        </div>
      </div>
    </template>
    <button @click="process()" :disabled="processing||!orig" class="btn-p">
      <template x-if="!processing"><span>🚀 Process with AI</span></template>
      <template x-if="processing"><span style="display:flex;gap:8px;"><div class="spin"></div> Processing...</span></template>
    </button>
  </div>

  <div class="card" x-show="final&&!processing">
    <div class="ct"><div class="cn">3</div> Print Sheet</div>
    <div style="display:flex;gap:10px;margin-bottom:10px">
      <select x-model="copies" style="flex:1;padding:8px;border-radius:8px"><option>4</option><option selected>8</option><option>12</option></select>
      <select x-model="paper" style="flex:1;padding:8px;border-radius:8px"><option>A4</option><option>4x6</option></select>
    </div>
    <button @click="genPrint()" class="btn-s" style="width:100%;justify-content:center">🖨️ Generate Print Sheet</button>
  </div>
</div>

<!-- RIGHT -->
<div>
<div class="card preview">
  <div class="pempty" x-show="!orig&&!processing">
    <h3 style="font-family:'Playfair Display',serif;font-size:18px;color:var(--txt3)">Upload Photo to Start</h3>
  </div>

  <div x-show="processing" style="flex:1;display:flex;align-items:center;justify-content:center;">
    <p style="font-weight:600;color:var(--br-dk)">Processing Photo with AI...</p>
  </div>

  <div x-show="final&&!processing">
    <p style="font-weight:600;margin-bottom:12px;color:var(--br-dk)">✨ Processed Result:</p>
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
      <div class="cmp-hdl" :style="`left:${pos}%`">↔</div>
    </div>

    <div style="margin-top:16px;">
      <a :href="final" download="passport_photo.jpg" class="btn-p" style="text-decoration:none">Download Passport Photo</a>
    </div>

    <template x-if="printImg">
      <div style="margin-top:20px;padding-top:16px;border-top:1px solid var(--bdr)">
        <p style="font-weight:600;margin-bottom:10px">🖨️ Print Sheet Ready:</p>
        <img :src="printImg" style="width:100%;max-height:220px;object-fit:contain;border-radius:8px">
        <a :href="printImg" download="print_sheet.jpg" class="btn-s" style="margin-top:10px;text-decoration:none;display:inline-block">Download Print Sheet</a>
      </div>
    </template>
  </div>
</div>
</div>
</div>

<footer>Passport Photo Studio Pro</footer>
</div>

<script>
function App(){return{
  orig:null,final:null,printImg:null,
  processing:false,dragging:false,pos:50,
  enhance:true,rmbg:true,crop:true,bg:'white',
  copies:'8',paper:'A4',
  toast:{show:false,type:'ok',msg:''},

  swatches:[
    {n:'White',v:'white',h:'#FFFFFF'},
    {n:'Off White',v:'offwhite',h:'#F5F5F0'},
    {n:'Official Blue',v:'blue',h:'#224987'},
    {n:'Light Blue',v:'lightblue',h:'#4A90D9'},
    {n:'Grey',v:'grey',h:'#E0E0E0'},
    {n:'Transparent',v:'transparent',h:'transparent'},
  ],

  showToast(msg,type='ok'){
    this.toast={show:true,type,msg};
    setTimeout(()=>this.toast.show=false,3000);
  },
  onFile(e){
    const f=e.target.files[0];
    if(!f)return;
    const r=new FileReader();
    r.onload=evt=>{
      this.orig=evt.target.result;
      this.final=null;
      this.printImg=null;
    };
    r.readAsDataURL(f);
  },
  async process(){
    if(!this.orig)return;
    this.processing=true;
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
      if(d.status==='success'){
        this.final=d.final_image;
        this.showToast('Photo Processed Successfully!');
      }else{this.showToast('Processing Failed','err');}
    }catch(e){this.showToast('Server connection error','err');}
    finally{this.processing=false;}
  },
  async genPrint(){
    if(!this.final)return;
    const blob=await(await fetch(this.final)).blob();
    const fd=new FormData();
    fd.append('file',blob,'passport.jpg');
    fd.append('copies',this.copies);
    fd.append('paper_size',this.paper);
    try{
      const res=await fetch('/api/print',{method:'POST',body:fd});
      const d=await res.json();
      if(d.status==='success'){
        this.printImg=d.print_layout;
        this.showToast('Print Sheet Generated!');
      }
    }catch(e){this.showToast('Error generating print sheet','err');}
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