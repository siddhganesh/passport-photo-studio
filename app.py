import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AI Passport Studio Pro</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>
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
header{padding:36px 0 28px;text-align:center;}
.badge{display:inline-flex;align-items:center;gap:8px;background:rgba(111,78,55,.08);border:1px solid rgba(111,78,55,.2);border-radius:100px;padding:7px 18px;font-size:12px;font-weight:600;color:var(--br);}
.dot{width:6px;height:6px;background:var(--br-lt);border-radius:50%;}
h1{font-family:'Playfair Display',serif;font-size:clamp(30px,4vw,52px);font-weight:700;color:var(--br-dk);margin-bottom:8px;}
.gt{background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sub{font-size:14px;color:var(--txt2);max-width:560px;margin:0 auto 24px;line-height:1.5;}
.grid{display:grid;grid-template-columns:390px 1fr;gap:20px;align-items:start;}
@media(max-width:1020px){.grid{grid-template-columns:1fr;}}
.card{background:var(--white);border:1px solid var(--bdr);border-radius:var(--r);padding:24px;box-shadow:var(--sh);}
.ct{font-family:'Playfair Display',serif;font-size:16px;font-weight:600;color:var(--br-dk);margin-bottom:18px;display:flex;align-items:center;gap:10px;}
.cn{width:26px;height:26px;background:var(--grad);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#fff;}
.uz{border:2px dashed rgba(111,78,55,.25);border-radius:14px;padding:24px 20px;text-align:center;cursor:pointer;background:var(--cream);}
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
.btn-p{width:100%;padding:14px;background:var(--grad);border:none;border-radius:13px;color:#fff;font-size:15px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:10px;margin-top:16px;text-decoration:none;}
.btn-s{padding:10px 18px;background:var(--cream);border:1px solid var(--bdr);border-radius:10px;color:var(--txt2);font-size:13px;font-weight:500;cursor:pointer;display:inline-flex;align-items:center;gap:8px;text-decoration:none;}
.preview{min-height:480px;display:flex;flex-direction:column;}
.pempty{flex:1;display:flex;align-items:center;justify-content:center;}
.cmp{position:relative;border-radius:14px;overflow:hidden;height:300px;cursor:ew-resize;user-select:none;background:var(--cream2);border:1px solid var(--bdr);touch-action:none;}
.cmp-b,.cmp-a{position:absolute;inset:0;}
.cmp-b img,.cmp-a img{width:100%;height:100%;object-fit:contain;}
.cmp-a{clip-path:inset(0 50% 0 0);}
.cmp-div{position:absolute;top:0;bottom:0;width:2px;background:var(--br);z-index:10;transform:translateX(-50%);}
.cmp-hdl{position:absolute;top:50%;width:36px;height:36px;background:var(--white);border-radius:50%;z-index:11;display:flex;align-items:center;justify-content:center;box-shadow:var(--sh2);transform:translate(-50%,-50%);}
.cmp-lbl{position:absolute;top:10px;padding:4px 10px;border-radius:6px;font-size:10px;font-weight:700;text-transform:uppercase;}
.lbl-b{left:10px;background:rgba(62,32,16,.5);color:#fff;}
.lbl-a{right:10px;background:var(--br);color:#fff;}
footer{text-align:center;padding:28px 0 20px;color:var(--txt3);font-size:12px;margin-top:24px;}
</style>
</head>
<body>
<div x-data="App()" class="wrap">

<header>
  <div class="badge"><span class="dot"></span>&nbsp;AI Selfie Segmentation &amp; Smart Crop</div>
  <h1>Passport Photo <span class="gt">Studio Pro</span></h1>
  <p class="sub">AI Background Remover + Passport Studio Generator</p>
</header>

<div class="grid">
<!-- LEFT PANEL -->
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
      <div class="tlabel">🤖 AI Remove Background</div>
      <div class="tog" :class="{on:rmbg}" @click="rmbg=!rmbg;process()"></div>
    </div>
    <div class="trow">
      <div class="tlabel">✨ Auto Enhance</div>
      <div class="tog" :class="{on:enhance}" @click="enhance=!enhance;process()"></div>
    </div>
    <div class="trow">
      <div class="tlabel">🎯 Passport 3:4 Crop</div>
      <div class="tog" :class="{on:crop}" @click="crop=!crop;process()"></div>
    </div>
    <div class="trow">
      <div class="tlabel">🎨 Background Color</div>
    </div>
    <div class="sw-wrap">
      <template x-for="s in swatches" :key="s.v">
        <div class="sw" :class="{act:bg===s.v,'sw-trans':s.v==='transparent'}"
             :style="s.v!=='transparent'?`background:${s.h}`:''"
             @click="bg=s.v;process()" :title="s.n"></div>
      </template>
    </div>
  </div>

  <div class="card" x-show="final">
    <div class="ct"><div class="cn">3</div> Print Sheet Setup</div>
    <div style="display:flex;gap:10px;margin-bottom:10px">
      <select x-model="copies" @change="genPrint()" style="flex:1;padding:8px;border-radius:8px"><option>4</option><option selected>8</option><option>12</option></select>
      <select x-model="paper" @change="genPrint()" style="flex:1;padding:8px;border-radius:8px"><option>A4</option><option>4x6</option></select>
    </div>
    <button @click="genPrint()" class="btn-s" style="width:100%;justify-content:center">Refresh Print Sheet</button>
  </div>
</div>

<!-- RIGHT PANEL -->
<div>
<div class="card preview">
  <div class="pempty" x-show="!orig">
    <h3 style="font-family:'Playfair Display',serif;font-size:18px;color:var(--txt3)">Upload Photo to Preview</h3>
  </div>

  <div x-show="final">
    <p style="font-weight:600;margin-bottom:12px;color:var(--br-dk)">✨ Result Preview:</p>
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
      <a :href="final" download="passport_photo.jpg" class="btn-p">Download Passport Photo</a>
    </div>

    <template x-if="printImg">
      <div style="margin-top:20px;padding-top:16px;border-top:1px solid var(--bdr)">
        <p style="font-weight:600;margin-bottom:10px">🖨️ Print Sheet Ready:</p>
        <img :src="printImg" style="width:100%;max-height:220px;object-fit:contain;border-radius:8px">
        <a :href="printImg" download="print_sheet.jpg" class="btn-s" style="margin-top:10px;display:inline-block">Download Print Sheet</a>
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
  orig:null,final:null,printImg:null,imgObj:null,segmenter:null,
  dragging:false,pos:50,
  rmbg:true,enhance:true,crop:true,bg:'white',
  copies:'8',paper:'A4',

  swatches:[
    {n:'White',v:'white',h:'#FFFFFF'},
    {n:'Off White',v:'offwhite',h:'#F5F5F0'},
    {n:'Official Blue',v:'blue',h:'#224987'},
    {n:'Light Blue',v:'lightblue',h:'#4A90D9'},
    {n:'Grey',v:'grey',h:'#E0E0E0'},
    {n:'Transparent',v:'transparent',h:'transparent'},
  ],

  init(){
    if(window.SelfieSegmentation){
      this.segmenter = new SelfieSegmentation({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`});
      this.segmenter.setOptions({modelSelection: 1});
    }
  },

  onFile(e){
    const f=e.target.files[0];
    if(!f)return;
    const r=new FileReader();
    r.onload=evt=>{
      this.orig=evt.target.result;
      this.imgObj=new Image();
      this.imgObj.onload=()=>{ this.process(); };
      this.imgObj.src=evt.target.result;
    };
    r.readAsDataURL(f);
  },

  async process(){
    if(!this.imgObj)return;
    
    const rawCanvas = document.createElement('canvas');
    const rawCtx = rawCanvas.getContext('2d');
    let w = this.imgObj.width;
    let h = this.imgObj.height;
    rawCanvas.width = w;
    rawCanvas.height = h;
    rawCtx.drawImage(this.imgObj, 0, 0);

    if(this.rmbg && this.segmenter){
      this.segmenter.onResults((results) => {
        this.renderCanvas(results.segmentationMask);
      });
      await this.segmenter.send({image: rawCanvas});
    } else {
      this.renderCanvas(null);
    }
  },

  renderCanvas(mask){
    const canvas=document.createElement('canvas');
    const ctx=canvas.getContext('2d');
    
    let w=this.imgObj.width;
    let h=this.imgObj.height;
    
    let cropX=0, cropY=0, cropW=w, cropH=h;
    if(this.crop){
      const tr=3/4;
      if((w/h)>tr){
        cropW=h*tr;
        cropX=(w-cropW)/2;
      }else{
        cropH=w/tr;
        cropY=Math.max(0,(h-cropH)/4);
      }
    }
    
    canvas.width=cropW;
    canvas.height=cropH;
    
    if(this.bg!=='transparent'){
      const colors={'white':'#FFFFFF','offwhite':'#F5F5F0','blue':'#224987','lightblue':'#4A90D9','grey':'#E0E0E0'};
      ctx.fillStyle=colors[this.bg]||'#FFFFFF';
      ctx.fillRect(0,0,cropW,cropH);
    }
    
    if(this.enhance){
      ctx.filter='contrast(112%) brightness(105%) saturate(105%)';
    }

    if(mask && this.rmbg){
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = w;
      tempCanvas.height = h;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.drawImage(mask, 0, 0, w, h);
      tempCtx.globalCompositeOperation = 'source-in';
      tempCtx.drawImage(this.imgObj, 0, 0, w, h);
      
      ctx.drawImage(tempCanvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
    } else {
      ctx.drawImage(this.imgObj, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
    }

    this.final=canvas.toDataURL('image/jpeg',0.95);
    this.genPrint();
  },

  genPrint(){
    if(!this.final)return;
    const pImg=new Image();
    pImg.onload=()=>{
      const canvas=document.createElement('canvas');
      const ctx=canvas.getContext('2d');
      
      const dpi=300;
      const pw=(this.paper==='A4')?Math.floor(8.27*dpi):Math.floor(6*dpi);
      const ph=(this.paper==='A4')?Math.floor(11.69*dpi):Math.floor(4*dpi);
      
      canvas.width=pw;
      canvas.height=ph;
      ctx.fillStyle='#FFFFFF';
      ctx.fillRect(0,0,pw,ph);
      
      const photoW=Math.floor(35/25.4*dpi);
      const photoH=Math.floor(45/25.4*dpi);
      const margin=20, spacing=10;
      
      const cols=Math.floor((pw-2*margin)/(photoW+spacing));
      const rows=Math.floor((ph-2*margin)/(photoH+spacing));
      let count=0;
      
      for(let r=0; r<rows; r++){
        for(let c=0; c<cols; c++){
          if(count>=parseInt(this.copies))break;
          ctx.drawImage(pImg, margin+c*(photoW+spacing), margin+r*(photoH+spacing), photoW, photoH);
          count++;
        }
        if(count>=parseInt(this.copies))break;
      }
      this.printImg=canvas.toDataURL('image/jpeg',0.9);
    };
    pImg.src=this.final;
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