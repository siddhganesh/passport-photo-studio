import io
import base64
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove
import uvicorn

app = FastAPI(title="AI Passport Photo Studio PRO - Direct Print")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── BACKEND PROCESSING ENGINE ─────────────────────────────────────────────────

def enhance_image(image: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    img = enhancer.enhance(1.12)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.25)
    return img

def apply_background_rembg(image: Image.Image, bg_color: str) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    
    result_bytes = remove(
        buf.getvalue(),
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10
    )
    rgba = Image.open(io.BytesIO(result_bytes)).convert('RGBA')

    if bg_color == 'transparent':
        return rgba

    colors = {
        'white': (255, 255, 255), 'offwhite': (245, 245, 240),
        'blue': (34, 73, 135), 'lightblue': (74, 144, 217),
        'grey': (232, 232, 232), 'black': (0, 0, 0), 'green': (34, 135, 73),
    }
    bg_rgb = colors.get(bg_color, (255, 255, 255))
    
    bg = Image.new('RGB', rgba.size, bg_rgb)
    bg.paste(rgba, mask=rgba.split()[3])
    return bg

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
):
    try:
        img = Image.open(io.BytesIO(await file.read()))
            
        if do_enhance.lower() == 'true':
            img = enhance_image(img)
            
        if remove_bg.lower() == 'true':
            img = apply_background_rembg(img, bg_color)
        
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

# ── FRONTEND WITH DIRECT PRINT BUTTON & PRINT DIALOG ──────────────────────────

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AI Passport Studio Pro - Direct Print</title>
<script src="https://cdn.tailwindcss.com"></script>
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<!-- Cropper.js for Interactive Zoom & Crop Box -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.js"></script>
<style>
  body {
    font-family: 'Outfit', sans-serif;
    background: #090d16;
    color: #f1f5f9;
    background-image: 
      radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
      radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.15) 0px, transparent 50%);
    background-attachment: fixed;
  }
  .glass-card {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  }
  .glow-button {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    box-shadow: 0 0 20px rgba(168, 85, 247, 0.35);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .glow-button:hover {
    box-shadow: 0 0 30px rgba(168, 85, 247, 0.6);
    transform: translateY(-2px);
  }
  .cmp {
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    height: 420px;
    cursor: ew-resize;
    user-select: none;
    background: #020617;
    border: 1px solid rgba(255,255,255,0.1);
  }
  .cmp-b, .cmp-a { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; }
  .cmp-b img, .cmp-a img { height: 100%; width: auto; max-width: 100%; object-fit: contain; }
  .cmp-a { clip-path: inset(0 50% 0 0); }
  .cmp-div { position: absolute; top: 0; bottom: 0; width: 2px; background: #a855f7; z-index: 10; transform: translateX(-50%); }
  .cmp-hdl {
    position: absolute; top: 50%; width: 36px; height: 36px;
    background: #ffffff; color: #0f172a; border-radius: 50%; z-index: 11;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 15px rgba(168, 85, 247, 0.8); transform: translate(-50%, -50%);
    font-weight: bold; font-size: 14px;
  }
  /* PRINT CSS */
  @media print {
    body * { visibility: hidden; }
    #printArea, #printArea * { visibility: visible; }
    #printArea { position: absolute; left: 0; top: 0; width: 100%; height: 100%; margin: 0; padding: 0; }
  }
</style>
</head>
<body class="min-h-screen p-4 md:p-8">
<div x-data="App()" class="max-w-7xl mx-auto">

  <!-- HIDDEN PRINT CONTAINER -->
  <div id="printArea" class="hidden">
    <img :src="printImg" style="width: 100%; height: auto; display: block;">
  </div>

  <!-- TOAST NOTIFICATION -->
  <div x-show="toast.show" x-transition class="fixed bottom-6 right-6 z-50 px-5 py-3 rounded-xl glass-card border border-purple-500/30 text-slate-100 flex items-center gap-3 shadow-2xl" style="display:none">
    <span class="w-2.5 h-2.5 rounded-full bg-purple-400 animate-ping"></span>
    <span x-text="toast.msg"></span>
  </div>

  <!-- HEADER -->
  <header class="text-center mb-10">
    <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-300 text-xs font-semibold tracking-wider uppercase mb-3">
      🖨️ One-Click Direct Printer Support
    </div>
    <h1 class="text-4xl md:text-6xl font-extrabold tracking-tight mb-3">
      AI Passport Studio <span class="bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">PRO</span>
    </h1>
    <p class="text-slate-400 text-base max-w-xl mx-auto">
      Interactive Zoom Crop, AI RemBG &amp; One-Click Direct Printing
    </p>
  </header>

  <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
    
    <!-- LEFT CONTROL PANEL (5 Cols) -->
    <div class="lg:col-span-5 space-y-6">
      
      <!-- 1. UPLOAD -->
      <div class="glass-card p-6 rounded-2xl">
        <div class="flex items-center gap-3 mb-4">
          <span class="w-7 h-7 rounded-lg bg-indigo-500/20 border border-indigo-500/40 text-indigo-300 flex items-center justify-center font-bold text-xs">1</span>
          <h2 class="text-lg font-bold text-slate-100">Upload Photo</h2>
        </div>
        <input type="file" x-ref="fi" accept="image/*" @change="onFile($event)" class="hidden">
        <div @click="$refs.fi.click()" class="border-2 border-dashed border-slate-700 hover:border-purple-500/50 rounded-xl p-6 text-center cursor-pointer transition bg-slate-900/40 group">
          <template x-if="!rawOrig">
            <div>
              <div class="w-12 h-12 rounded-full bg-purple-500/10 text-purple-400 flex items-center justify-center mx-auto mb-3 group-hover:scale-110 transition">
                📸
              </div>
              <p class="font-semibold text-slate-200">Click to Upload Photo</p>
              <p class="text-xs text-slate-500 mt-1">Full-Body or Close-up</p>
            </div>
          </template>
          <template x-if="rawOrig">
            <div class="relative">
              <img :src="rawOrig" class="w-full h-44 object-cover rounded-lg border border-slate-700">
              <div class="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center text-xs font-semibold text-white transition rounded-lg">
                Change Photo
              </div>
            </div>
          </template>
        </div>
      </div>

      <!-- INTERACTIVE CROPPER MODAL/BOX -->
      <div x-show="rawOrig && !croppedConfirmed" x-transition class="glass-card p-6 rounded-2xl space-y-4">
        <div class="flex items-center justify-between">
          <h3 class="text-md font-bold text-purple-300">✂️ Adjust Crop &amp; Zoom Frame</h3>
          <span class="text-xs text-slate-400">3:4 Passport Ratio</span>
        </div>
        <div class="max-h-80 bg-slate-950 rounded-xl overflow-hidden border border-slate-800">
          <img id="cropTarget" :src="rawOrig" class="max-w-full">
        </div>
        <button @click="confirmCrop()" class="w-full py-3 glow-button text-white font-bold rounded-xl text-sm">
          ✓ Confirm Crop Frame &amp; Zoom
        </button>
      </div>

      <!-- 2. AI CONTROLS AFTER CROP -->
      <div x-show="croppedConfirmed" x-transition class="glass-card p-6 rounded-2xl space-y-5">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <span class="w-7 h-7 rounded-lg bg-purple-500/20 border border-purple-500/40 text-purple-300 flex items-center justify-center font-bold text-xs">2</span>
            <h2 class="text-lg font-bold text-slate-100">AI Controls</h2>
          </div>
          <button @click="reCrop()" class="text-xs text-purple-400 hover:underline">Re-Adjust Crop/Zoom</button>
        </div>

        <div class="space-y-3">
          <label class="flex items-center justify-between cursor-pointer p-2.5 rounded-lg bg-slate-900/40 hover:bg-slate-900/70 border border-slate-800">
            <span class="text-sm font-medium text-slate-200">🤖 AI Background Erase (RemBG)</span>
            <input type="checkbox" x-model="rmbg" class="w-4 h-4 accent-purple-500">
          </label>
          
          <label class="flex items-center justify-between cursor-pointer p-2.5 rounded-lg bg-slate-900/40 hover:bg-slate-900/70 border border-slate-800">
            <span class="text-sm font-medium text-slate-200">✨ Auto Clarity &amp; Sharpen</span>
            <input type="checkbox" x-model="enhance" class="w-4 h-4 accent-purple-500">
          </label>
        </div>

        <template x-if="rmbg">
          <div class="pt-2">
            <span class="text-xs font-semibold text-slate-400 tracking-wider uppercase block mb-2.5">Studio Background Color</span>
            <div class="flex gap-2.5 flex-wrap">
              <button @click="bg='white'" :class="bg==='white'?'ring-2 ring-purple-400 scale-105':''" class="w-8 h-8 rounded-lg bg-white border border-slate-300 transition" title="White"></button>
              <button @click="bg='offwhite'" :class="bg==='offwhite'?'ring-2 ring-purple-400 scale-105':''" class="w-8 h-8 rounded-lg bg-slate-100 border border-slate-300 transition" title="Off-White"></button>
              <button @click="bg='blue'" :class="bg==='blue'?'ring-2 ring-purple-400 scale-105':''" class="w-8 h-8 rounded-lg bg-blue-800 transition" title="Official Blue"></button>
              <button @click="bg='lightblue'" :class="bg==='lightblue'?'ring-2 ring-purple-400 scale-105':''" class="w-8 h-8 rounded-lg bg-sky-400 transition" title="Light Blue"></button>
              <button @click="bg='grey'" :class="bg==='grey'?'ring-2 ring-purple-400 scale-105':''" class="w-8 h-8 rounded-lg bg-slate-400 transition" title="Grey"></button>
              <button @click="bg='transparent'" :class="bg==='transparent'?'ring-2 ring-purple-400 scale-105':''" class="w-8 h-8 rounded-lg bg-slate-950 border border-slate-700 text-[10px] text-slate-400 font-bold">PNG</button>
            </div>
          </div>
        </template>

        <button @click="process()" :disabled="processing" class="w-full py-3.5 glow-button text-white font-bold rounded-xl text-base flex items-center justify-center gap-2">
          <template x-if="!processing">
            <span class="flex items-center gap-2">🚀 Process with AI</span>
          </template>
          <template x-if="processing">
            <span class="flex items-center gap-2"><div class="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div> RemBG AI Processing...</span>
          </template>
        </button>
      </div>

      <!-- 3. PRINT LAYOUT -->
      <div x-show="final" x-transition class="glass-card p-6 rounded-2xl space-y-4">
        <div class="flex items-center gap-3">
          <span class="w-7 h-7 rounded-lg bg-pink-500/20 border border-pink-500/40 text-pink-300 flex items-center justify-center font-bold text-xs">3</span>
          <h2 class="text-lg font-bold text-slate-100">Print Sheet Setup</h2>
        </div>
        <div class="grid grid-cols-2 gap-3">
          <div>
            <span class="text-xs text-slate-400 font-medium block mb-1">Copies</span>
            <select x-model="copies" class="w-full bg-slate-900/80 border border-slate-800 rounded-lg p-2 text-sm text-slate-200">
              <option>4</option><option selected>8</option><option>12</option><option>16</option>
            </select>
          </div>
          <div>
            <span class="text-xs text-slate-400 font-medium block mb-1">Paper Size</span>
            <select x-model="paper" class="w-full bg-slate-900/80 border border-slate-800 rounded-lg p-2 text-sm text-slate-200">
              <option>A4</option><option>4x6</option><option>Letter</option>
            </select>
          </div>
        </div>
        <button @click="genPrint()" class="w-full py-3 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-semibold rounded-xl transition flex items-center justify-center gap-2 text-sm">
          📄 Generate Print Sheet Preview
        </button>
      </div>

    </div>

    <!-- RIGHT PREVIEW PANEL (7 Cols) -->
    <div class="lg:col-span-7 glass-card p-6 rounded-2xl min-h-[550px] flex flex-col justify-between">
      
      <template x-if="!rawOrig">
        <div class="flex-1 flex flex-col items-center justify-center text-center p-8">
          <div class="w-16 h-16 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-2xl mb-4">
            🖨️
          </div>
          <h3 class="text-xl font-bold text-slate-300 mb-1">Passport Studio Preview</h3>
          <p class="text-sm text-slate-500 max-w-sm">Upload photo to open Zoom &amp; Crop box, RemBG &amp; Direct One-Click Printing!</p>
        </div>
      </template>

      <template x-if="final">
        <div class="space-y-6">
          <div class="flex items-center justify-between">
            <h3 class="text-lg font-bold text-slate-200">Result Comparison Slider</h3>
            <span class="text-xs text-purple-400 font-medium bg-purple-500/10 px-3 py-1 rounded-full border border-purple-500/20">300 DPI Studio Quality</span>
          </div>

          <!-- SLIDER -->
          <div class="cmp"
               @mousedown="dragging=true;slide($event)"
               @mousemove="slide($event)"
               @mouseup="dragging=false"
               @mouseleave="dragging=false"
               @touchstart.prevent="dragging=true;slideT($event)"
               @touchmove.prevent="slideT($event)"
               @touchend="dragging=false">
            <div class="cmp-b"><img :src="croppedOrig" alt="Cropped"></div>
            <div class="cmp-a" :style="`clip-path:inset(0 ${100-pos}% 0 0)`"><img :src="final" alt="Processed"></div>
            <div class="cmp-div" :style="`left:${pos}%`"></div>
            <div class="cmp-hdl" :style="`left:${pos}%`">↔</div>
          </div>

          <div class="flex gap-4">
            <a :href="final" download="passport_photo.jpg" class="flex-1 py-3.5 glow-button text-white text-center font-bold rounded-xl text-sm block">
              ⬇️ Download Passport Photo
            </a>
          </div>

          <template x-if="printImg">
            <div class="pt-6 border-t border-slate-800/80 space-y-4">
              <div class="flex items-center justify-between">
                <h4 class="text-md font-bold text-slate-300">Print Sheet Ready</h4>
                <span class="text-xs text-slate-500">Auto Grid Layout</span>
              </div>
              <div class="p-2 rounded-xl bg-slate-950 border border-slate-800">
                <img :src="printImg" class="max-h-64 mx-auto rounded">
              </div>

              <!-- DIRECT PRINT BUTTON -->
              <div class="flex gap-3">
                <button @click="directPrint()" class="flex-1 py-3.5 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded-xl text-sm shadow-lg shadow-emerald-600/30 flex items-center justify-center gap-2">
                  🖨️ DIRECT PRINT NOW
                </button>
                <a :href="printImg" download="passport_print_sheet.jpg" class="py-3.5 px-5 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 font-semibold rounded-xl text-sm block">
                  ⬇️ Save File
                </a>
              </div>

            </div>
          </template>
        </div>
      </template>

    </div>

  </div>

  <footer class="mt-12 text-center text-xs text-slate-600">
    AI Passport Photo Studio PRO &nbsp;&middot;&nbsp; Direct Printer Driver Integration
  </footer>

</div>

<script>
function App() {
  return {
    rawOrig: null, croppedOrig: null, final: null, printImg: null,
    cropper: null, croppedConfirmed: false,
    processing: false, dragging: false, pos: 50,
    enhance: true, rmbg: true, bg: 'white',
    copies: '8', paper: 'A4',
    toast: { show: false, msg: '' },

    showToast(msg) {
      this.toast = { show: true, msg };
      setTimeout(() => this.toast.show = false, 3000);
    },

    onFile(e) {
      const f = e.target.files[0];
      if (!f) return;
      const r = new FileReader();
      r.onload = evt => {
        this.rawOrig = evt.target.result;
        this.croppedConfirmed = false;
        this.final = null;
        this.printImg = null;

        this.$nextTick(() => {
          this.initCropper();
        });
      };
      r.readAsDataURL(f);
    },

    initCropper() {
      if (this.cropper) {
        this.cropper.destroy();
      }
      const image = document.getElementById('cropTarget');
      this.cropper = new Cropper(image, {
        aspectRatio: 3 / 4,
        viewMode: 1,
        dragMode: 'move',
        autoCropArea: 0.8,
        restore: false,
        guides: true,
        center: true,
        highlight: false,
        cropBoxMovable: true,
        cropBoxResizable: true,
        toggleDragModeOnDblclick: false,
      });
    },

    confirmCrop() {
      if (!this.cropper) return;
      const canvas = this.cropper.getCroppedCanvas({
        width: 600,
        height: 800,
      });
      this.croppedOrig = canvas.toDataURL('image/jpeg', 0.95);
      this.croppedConfirmed = true;
      this.showToast('✓ Crop & Zoom Confirmed!');
    },

    reCrop() {
      this.croppedConfirmed = false;
      this.$nextTick(() => {
        this.initCropper();
      });
    },

    async process() {
      if (!this.croppedOrig) return;
      this.processing = true;
      const blob = await (await fetch(this.croppedOrig)).blob();
      const fd = new FormData();
      fd.append('file', blob, 'upload.jpg');
      fd.append('remove_bg', this.rmbg);
      fd.append('bg_color', this.bg);
      fd.append('do_enhance', this.enhance);
      fd.append('do_crop', 'false');
      try {
        const res = await fetch('/api/process', { method: 'POST', body: fd });
        const d = await res.json();
        if (d.status === 'success') {
          this.final = d.final_image;
          this.pos = 50;
          this.showToast('✨ Studio RemBG Processed!');
        } else {
          this.showToast('❌ Processing Error');
        }
      } catch (e) {
        this.showToast('❌ Connection Error');
      } finally {
        this.processing = false;
      }
    },

    async genPrint() {
      if (!this.final) return;
      const blob = await (await fetch(this.final)).blob();
      const fd = new FormData();
      fd.append('file', blob, 'passport.jpg');
      fd.append('copies', this.copies);
      fd.append('paper_size', this.paper);
      try {
        const res = await fetch('/api/print', { method: 'POST', body: fd });
        const d = await res.json();
        if (d.status === 'success') {
          this.printImg = d.print_layout;
          this.showToast('🖨️ Print Sheet Generated!');
        }
      } catch (e) {
        this.showToast('❌ Print Generation Error');
      }
    },

    directPrint() {
      if (!this.printImg) return;
      this.showToast('🖨️ Opening Printer Dialog...');
      setTimeout(() => {
        window.print();
      }, 300);
    },

    slide(e) { if (!this.dragging) return; const r = e.currentTarget.getBoundingClientRect(); this.pos = Math.max(2, Math.min(98, (e.clientX - r.left) / r.width * 100)); },
    slideT(e) { const r = e.currentTarget.getBoundingClientRect(); const t = e.touches[0]; this.pos = Math.max(2, Math.min(98, (t.clientX - r.left) / r.width * 100)); }
  }
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8090, reload=False)