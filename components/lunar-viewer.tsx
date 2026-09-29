'use client';
import { useEffect, useRef, useState } from 'react';
import { Crosshair, Hand, Maximize, Minus, Plus, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import type { TiePoint } from '@/lib/lunar-data';

export type ViewMode = 'comparison' | 'tie-points' | 'overlay' | 'registered';
type Raster = { image: HTMLImageElement; pixels: ImageData };
type Dimensions = { source: { width: number; height: number }; reference: { width: number; height: number } };
type Props = { mode: ViewMode; source: string; reference: string; registered: string | null; simulated: boolean; points: TiePoint[]; selected: TiePoint | null; onSelect: (p: TiePoint) => void; dimensions?: Dimensions };
export function LunarViewer({ mode, source, reference, registered, simulated, points, selected, onSelect, dimensions }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const surface = useRef<HTMLDivElement>(null);
  const [rasters, setRasters] = useState<Record<string, Raster>>({});
  const [size, setSize] = useState({ w: 800, h: 480 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [panMode, setPanMode] = useState(false);
  const [reveal, setReveal] = useState(50);
  const [opacity, setOpacity] = useState(50);
  const [hover, setHover] = useState('Move over image to inspect pixels');
  const [error, setError] = useState('');
  const drag = useRef<{ kind: 'split' | 'pan'; x: number; y: number; px: number; py: number } | null>(null);
  const aligned = registered || reference;
  const empty = !source && !reference;
  useEffect(() => {
    let live = true; setError(''); setRasters({}); setZoom(1); setPan({ x: 0, y: 0 });
    for (const url of new Set([source, reference, aligned])) {
      if (!url) continue;
      const image = new Image();
      image.onload = () => {
        if (!live) return;
        const c = document.createElement('canvas'); c.width = image.naturalWidth; c.height = image.naturalHeight;
        const ctx = c.getContext('2d', { willReadFrequently: true }); if (!ctx) return;
        ctx.drawImage(image, 0, 0);
        try { const pixels = ctx.getImageData(0, 0, c.width, c.height); setRasters(r => ({ ...r, [url]: { image, pixels } })); }
        catch { setError('Pixel inspection unavailable for this image. Load a local image or a backend preview.'); }
      };
      image.onerror = () => { if (live) setError('Preview could not be loaded. TIFF / IMG previews require the processing service.'); };
      image.src = url;
    }
    return () => { live = false; };
  }, [source, reference, aligned]);
  useEffect(() => {
    if (!surface.current) return;
    const observer = new ResizeObserver(entries => { const { width, height } = entries[0].contentRect; setSize({ w: width, h: height }); });
    observer.observe(surface.current); return () => observer.disconnect();
  }, []);
  const refRaster = rasters[reference] || rasters[source];
  const width = dimensions?.reference.width || refRaster?.image.naturalWidth || 1300, height = dimensions?.reference.height || refRaster?.image.naturalHeight || 1300;
  const frameWidth = mode === 'tie-points' ? size.w / 2 : size.w;
  const scale = Math.min(frameWidth / width, size.h / height) * zoom;
  const origin = { x: (frameWidth - width * scale) / 2 + pan.x, y: (size.h - height * scale) / 2 + pan.y };
  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const dpr = window.devicePixelRatio || 1; canvas.width = Math.round(size.w * dpr); canvas.height = Math.round(size.h * dpr);
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    ctx.scale(dpr, dpr); ctx.fillStyle = '#03080c'; ctx.fillRect(0, 0, size.w, size.h);
    const draw = (url: string, offset = 0, isSource = false) => {
      const raster = rasters[url]; if (!raster) return;
      ctx.save(); ctx.beginPath(); ctx.rect(offset, 0, frameWidth, size.h); ctx.clip();
      ctx.translate(origin.x + offset, origin.y); ctx.scale(scale, scale);
      if (simulated && isSource) { ctx.translate(18, -12); ctx.filter = 'brightness(0.86) contrast(1.08)'; }
      const original = isSource ? dimensions?.source : dimensions?.reference;
      ctx.drawImage(raster.image, 0, 0, original?.width || raster.image.naturalWidth, original?.height || raster.image.naturalHeight); ctx.restore();
    };
    if (mode === 'tie-points') {
      draw(source, 0, true); draw(reference, frameWidth);
      ctx.strokeStyle = '#29424b'; ctx.beginPath(); ctx.moveTo(frameWidth, 0); ctx.lineTo(frameWidth, size.h); ctx.stroke();
      for (const p of points) {
        const x1 = origin.x + p.source_x * scale, y1 = origin.y + p.source_y * scale;
        const x2 = origin.x + frameWidth + p.reference_x * scale, y2 = origin.y + p.reference_y * scale;
        const chosen = selected?.id === p.id;
        ctx.strokeStyle = p.status === 'accepted' ? '#5ee397' : '#ff7584'; ctx.lineWidth = chosen ? 2 : 1;
        if (x1 < 0 || x1 > frameWidth || x2 < frameWidth || x2 > size.w || y1 < 0 || y1 > size.h || y2 < 0 || y2 > size.h) continue;
        ctx.globalAlpha = chosen ? .95 : .22; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.globalAlpha = 1;
        for (const [x,y] of [[x1,y1],[x2,y2]]) {
          ctx.beginPath(); if (p.status === 'accepted') ctx.arc(x, y, chosen ? 7 : 4, 0, Math.PI * 2);
          else { ctx.moveTo(x-5, y-5); ctx.lineTo(x+5,y+5); ctx.moveTo(x+5,y-5); ctx.lineTo(x-5,y+5); } ctx.stroke();
          if (chosen) { ctx.fillStyle = '#051016'; ctx.fillRect(x+10,y-12,66,19); ctx.fillStyle = '#fff'; ctx.font='11px monospace'; ctx.fillText(p.id,x+13,y+1); }
        }
      }
    } else if (mode === 'registered') draw(aligned);
    else if (mode === 'overlay') { draw(reference); ctx.globalAlpha = opacity / 100; draw(source, 0, true); ctx.globalAlpha = 1; }
    else { draw(aligned); ctx.save(); ctx.beginPath(); ctx.rect(0, 0, size.w * reveal / 100, size.h); ctx.clip(); draw(source, 0, true); ctx.restore(); }
  }, [rasters, size, scale, origin.x, origin.y, frameWidth, mode, source, reference, aligned, simulated, reveal, opacity, points, selected, dimensions]);
  const position = (e: React.PointerEvent) => { const r = surface.current!.getBoundingClientRect(); return { x: e.clientX-r.left, y: e.clientY-r.top }; };
  const reset = () => { setZoom(1); setPan({ x:0, y:0 }); setReveal(50); setOpacity(50); setHover('Move over image to inspect pixels'); };
  const move = (e: React.PointerEvent) => {
    const p = position(e);
    if (drag.current) { if (drag.current.kind === 'split') setReveal(Math.max(0,Math.min(100,p.x/size.w*100))); else setPan({ x:drag.current.px+p.x-drag.current.x, y:drag.current.py+p.y-drag.current.y }); return; }
    const side = mode === 'tie-points' && p.x >= frameWidth ? frameWidth : 0;
    const x = Math.floor((p.x-side-origin.x)/scale), y = Math.floor((p.y-origin.y)/scale);
    const isSource = mode === 'tie-points' ? !side : mode === 'comparison' && p.x < size.w*reveal/100;
    const url = isSource ? source : mode === 'tie-points' || mode === 'overlay' ? reference : aligned;
    const raster = rasters[url]; const original = isSource ? dimensions?.source : dimensions?.reference;
    const sx = Math.floor((x-(simulated&&isSource?18:0)) * (raster?.pixels.width || 1) / (original?.width || raster?.pixels.width || 1));
    const sy = Math.floor((y+(simulated&&isSource?12:0)) * (raster?.pixels.height || 1) / (original?.height || raster?.pixels.height || 1));
    if (!raster || sx<0 || sy<0 || sx>=raster.pixels.width || sy>=raster.pixels.height) { setHover('Outside raster extent'); return; }
    const i = (sy*raster.pixels.width+sx)*4; const dn = Math.round(.299*raster.pixels.data[i]+.587*raster.pixels.data[i+1]+.114*raster.pixels.data[i+2]);
    setHover(`${isSource?'SRC':'REF'}  X ${x}  Y ${y} px  ·  ${dn} / 255${mode==='overlay'?' · reference sample':''}`);
  };
  const down = (e: React.PointerEvent) => {
    if (e.button !== 0) return; const p = position(e);
    if (mode === 'comparison' && !panMode && Math.abs(p.x-size.w*reveal/100)<24) drag.current={ kind:'split', ...p, px:pan.x,py:pan.y };
    else if (panMode) drag.current={kind:'pan',...p,px:pan.x,py:pan.y};
    else if (mode === 'tie-points') {
      const side = p.x >= frameWidth; let best: TiePoint | null = null, distance = 15;
      for(const t of points) { const x=origin.x+(side?frameWidth:0)+(side?t.reference_x:t.source_x)*scale, y=origin.y+(side?t.reference_y:t.source_y)*scale; const d=Math.hypot(p.x-x,p.y-y); if(d<distance){best=t;distance=d;} }
      if(best) onSelect(best);
    }
    if(drag.current) e.currentTarget.setPointerCapture(e.pointerId);
  };
  return <>
    <div className="viewer-toolbar"><div className="tool-group"><Button variant="ghost" size="sm" aria-label="Zoom out" disabled={empty} onClick={()=>setZoom(z=>Math.max(.25,z/1.25))}><Minus/></Button><span className="zoom-value">{Math.round(zoom*100)}%</span><Button variant="ghost" size="sm" aria-label="Zoom in" disabled={empty} onClick={()=>setZoom(z=>Math.min(8,z*1.25))}><Plus/></Button><i/><Button variant={panMode?'secondary':'ghost'} size="sm" aria-pressed={panMode} disabled={empty} onClick={()=>setPanMode(!panMode)}><Hand/> Pan</Button><Button variant="ghost" size="sm" disabled={empty} onClick={()=>{setZoom(1);setPan({x:0,y:0});}}><Maximize/> Fit</Button><Button variant="ghost" size="sm" disabled={empty} onClick={reset}><RotateCcw/> Reset</Button></div><span className="raster-meta">{empty?'No raster loaded':`${width} × ${height} px · grayscale preview`}</span></div>
    <div ref={surface} className={`comparison scientific-viewer ${panMode?'panning':''}`} onPointerDown={down} onPointerMove={move} onPointerUp={()=>{drag.current=null;}} onPointerCancel={()=>{drag.current=null;}} onLostPointerCapture={()=>{drag.current=null;}}>
      <canvas ref={canvasRef} className="image-layer" aria-label={`${mode} lunar imagery; use toolbar to zoom and pan`}/>
      {empty&&<div className="viewer-message">Load source and reference images, or choose Load demo data.</div>}
      {!empty&&(!source || !reference)&&<div className="viewer-message">TIFF / IMG previews require the processing service. Run Registration with a connected backend.</div>}
      {source&&reference&&!Object.keys(rasters).length&&!error&&<div className="viewer-message">Loading lunar raster…</div>}
      {error&&<div className="viewer-message" role="alert">{error}</div>}
      {!empty&&mode==='comparison'&&<div className="reveal-line" style={{left:`${reveal}%`}}><span/></div>}
      {!empty&&<span className="image-tag left-tag">{mode==='registered'?'REGISTERED':mode==='overlay'?'SOURCE + REFERENCE':'SOURCE'}</span>}
      {!empty&&mode!=='registered'&&mode!=='overlay'&&<span className="image-tag right-tag">{mode==='tie-points'||!registered?'REFERENCE':'REGISTERED'}</span>}
      {!empty&&<span className="preview-tag">{simulated?'SIMULATED ALIGNMENT':registered?'BACKEND PREVIEW':'INPUT PREVIEW'}</span>}
    </div>
    <div className="coordinate-bar"><Crosshair/><output>{hover}</output><span>Pixel coordinates · origin top left</span></div>
    {!empty&&mode==='comparison'&&<div className="viewer-controls"><span>Source</span><Slider aria-label="Source registered comparison" value={[reveal]} min={0} max={100} onValueChange={v=>setReveal(Array.isArray(v)?v[0]:v)}/><span>Registered</span></div>}
    {mode==='overlay'&&<div className="viewer-controls"><span>Reference</span><Slider aria-label="Source overlay opacity" value={[opacity]} min={0} max={100} onValueChange={v=>setOpacity(Array.isArray(v)?v[0]:v)}/><span>Source {opacity}%</span></div>}
    {mode==='tie-points'&&<div className="viewer-legend"><span className="accepted">○ Accepted</span><span className="rejected">× Rejected</span><span>Click either endpoint to inspect a match.</span></div>}
    {mode==='registered'&&<div className="viewer-legend">Aligned to the reference image grid · {simulated?'demonstration output':'backend output'}</div>}
  </>;
}
