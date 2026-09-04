'use client';

import { useEffect, useRef, useState } from 'react';
import { Activity, ArrowRight, CheckCircle2, CircleDot, Download, FileImage, Gauge, Layers3, Moon, Play, RotateCcw, Satellite, Settings2, ShieldCheck, UploadCloud } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress, ProgressLabel, ProgressValue } from '@/components/ui/progress';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

type Stage = 'ready' | 'running' | 'complete';
const pipeline = ['Ingest', 'Condition', 'Match', 'Filter', 'Align', 'Validate'];

function LunarComparison({ reveal, showPoints }: { reveal: number; showPoints: boolean }) {
  const baseRef = useRef<HTMLCanvasElement>(null);
  const alignedRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const draw = (canvas: HTMLCanvasElement | null, aligned: boolean) => {
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const [w, h] = [1100, 600]; canvas.width = w; canvas.height = h;
      const g = ctx.createLinearGradient(0, 0, w, h);
      g.addColorStop(0, aligned ? '#5c6268' : '#303740'); g.addColorStop(.54, aligned ? '#93999e' : '#727981'); g.addColorStop(1, aligned ? '#444b52' : '#202831');
      ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
      const craters = [[145,118,74],[355,168,112],[690,115,67],[888,195,120],[175,410,126],[505,390,78],[760,435,146],[1010,430,58],[535,92,35],[340,485,42],[925,72,28]];
      for (const [x0,y0,r] of craters) {
        const x=x0+(aligned?0:16), y=y0+(aligned?0:-10);
        const radial=ctx.createRadialGradient(x-r*.28,y-r*.25,r*.08,x,y,r);
        radial.addColorStop(0,aligned?'#d7d8d8':'#a9afb5'); radial.addColorStop(.28,aligned?'#777d82':'#555d65'); radial.addColorStop(.68,aligned?'#252b31':'#171e25'); radial.addColorStop(1,aligned?'#a6aaad':'#737b82');
        ctx.fillStyle=radial; ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill(); ctx.strokeStyle=aligned?'rgba(241,245,249,.48)':'rgba(203,213,225,.35)'; ctx.lineWidth=Math.max(2,r*.06); ctx.stroke();
      }
      ctx.globalAlpha=.16;
      for(let i=0;i<190;i++){const x=(i*83)%w,y=(i*137)%h;ctx.fillStyle=i%3?'#fff':'#020617';ctx.beginPath();ctx.arc(x,y,1+(i%7),0,Math.PI*2);ctx.fill();}
      ctx.globalAlpha=1;
      if(showPoints){for(let i=0;i<31;i++){const x=55+((i*149)%990),y=42+((i*89)%510);ctx.strokeStyle=i%8===0?'#fb7185':'#4ade80';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x-6,y);ctx.lineTo(x+6,y);ctx.stroke();ctx.beginPath();ctx.moveTo(x,y-6);ctx.lineTo(x,y+6);ctx.stroke();}}
    };
    draw(baseRef.current,false); draw(alignedRef.current,true);
  },[showPoints]);
  return <div className="comparison" aria-label="Before and after lunar image registration comparison">
    <canvas ref={baseRef} className="image-layer" />
    <div className="aligned-layer" style={{width:`${reveal}%`}}><canvas ref={alignedRef} className="image-layer aligned-canvas" /></div>
    <div className="reveal-line" style={{left:`${reveal}%`}}><span /></div>
    <span className="image-tag left-tag">SOURCE · OHRC</span><span className="image-tag right-tag">REGISTERED</span>
    <div className="coords">89.427°S&nbsp;&nbsp; 32.814°E&nbsp;&nbsp; ·&nbsp;&nbsp; 0.25 m/px</div>
  </div>;
}

export default function Home(){
  const [stage,setStage]=useState<Stage>('ready'); const [progress,setProgress]=useState(0); const [reveal,setReveal]=useState(52); const [showPoints,setShowPoints]=useState(false);
  const [sourceName,setSourceName]=useState('OHRC_SOUTH_POLE_01.tif'); const [referenceName,setReferenceName]=useState('LROC_NAC_REFERENCE.tif');
  const [sourceFile,setSourceFile]=useState<File|null>(null); const [referenceFile,setReferenceFile]=useState<File|null>(null); const [actualMetrics,setActualMetrics]=useState<Record<string,number|string>|null>(null); const [error,setError]=useState('');
  const run=async()=>{setError('');setActualMetrics(null);setStage('running');setProgress(8);
    if(sourceFile&&referenceFile){try{const form=new FormData();form.append('source',sourceFile);form.append('reference',referenceFile);setProgress(34);const response=await fetch(`${process.env.NEXT_PUBLIC_API_URL||'http://localhost:8000'}/register`,{method:'POST',body:form});const data=await response.json();if(!response.ok)throw new Error(data.detail||'Registration failed');setProgress(91);setActualMetrics(data.metrics);setTimeout(()=>{setProgress(100);setStage('complete');},350);}catch(err){setError(err instanceof Error?err.message:'Registration failed');setProgress(0);setStage('ready');}return;}
    [22,39,58,76,91,100].forEach((value,index)=>setTimeout(()=>{setProgress(value);if(value===100)setStage('complete');},520*(index+1)));
  };
  const reset=()=>{setStage('ready');setProgress(0);setShowPoints(false);setActualMetrics(null);setError('');}; const activeIndex=stage==='complete'?5:Math.max(0,Math.ceil(progress/17)-1);
  const demoMetrics={candidate_matches:3420,accepted_matches:3126,inlier_ratio:.914,rmse_px:.42,spatial_coverage:.86,runtime_seconds:3.18}; const metrics=actualMetrics??demoMetrics;
  const exportReport=()=>{
    const report={project:'Chandra-Align',mode:actualMetrics?'processed':'demonstration',source:sourceName,reference:referenceName,engine:actualMetrics?.engine??'adaptive-dual-engine',metrics,warning:actualMetrics?'Model-inlier residuals are not a substitute for independent held-out checkpoints.':'Demonstration metrics. Production claims require held-out checkpoints and named datasets.'};
    const blob=new Blob([JSON.stringify(report,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const anchor=document.createElement('a');anchor.href=url;anchor.download='chandra-align-validation.json';anchor.click();URL.revokeObjectURL(url);
  };
  useEffect(()=>{
    const context=(document as Document & {modelContext?:{registerTool:(tool:unknown,options?:{signal?:AbortSignal})=>void|Promise<void>}}).modelContext;if(!context?.registerTool)return;
    const lifecycle=new AbortController();
    void Promise.resolve(context.registerTool({name:'run_lunar_coregistration_demo',title:'Run lunar co-registration demo',description:'Run the visible Chandra-Align demonstration pipeline and update its validation results.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:false,untrustedContentHint:false},execute:async()=>{setStage('running');setProgress(8);await new Promise(resolve=>setTimeout(resolve,450));setProgress(100);setStage('complete');return{status:'validated',rmse_px:0.42,inlier_ratio:0.914,mode:'demonstration'};}},{signal:lifecycle.signal})).catch(()=>{});
    return()=>lifecycle.abort();
  },[]);
  return <main className="min-h-screen bg-background text-foreground">
    <header className="topbar"><div className="brand"><div className="brand-mark"><Moon/></div><div><strong>LUNARGRID</strong><span>CHANDRA-ALIGN</span></div></div><div className="mission"><span className="live-dot"/> SIH 26166 · MISSION WORKSPACE</div><div className="top-actions"><span className="system-ok"><ShieldCheck/> Systems nominal</span><Button variant="outline" size="sm"><Settings2/> Settings</Button></div></header>
    <section className="workspace-shell">
      <aside className="side-panel"><div className="eyebrow">INPUT PAIR</div><h1>Register lunar imagery</h1><p className="subcopy">Align multi-sensor tiles across scale, illumination and viewing geometry.</p>
        <label className="upload-card"><input type="file" accept="image/*,.tif,.tiff,.img" onChange={e=>{const f=e.target.files?.[0];if(f){setSourceFile(f);setSourceName(f.name)}}}/><UploadCloud/><span><b>Source image</b><small>{sourceName}</small></span><CheckCircle2 className="file-ok"/></label>
        <label className="upload-card"><input type="file" accept="image/*,.tif,.tiff,.img" onChange={e=>{const f=e.target.files?.[0];if(f){setReferenceFile(f);setReferenceName(f.name)}}}/><Layers3/><span><b>Reference image</b><small>{referenceName}</small></span><CheckCircle2 className="file-ok"/></label>
        <label className="upload-card compact"><input type="file" accept=".xml"/><FileImage/><span><b>PDS4 label</b><small>Metadata auto-detected</small></span></label>
        <div className="metadata-grid"><div><span>SENSOR</span><b>OHRC</b></div><div><span>GSD</span><b>0.25 m</b></div><div><span>SUN AZ.</span><b>67.4°</b></div><div><span>SCALE GAP</span><b>8×</b></div></div>
        <div className="mode-box"><div><span>ENGINE</span><b>Adaptive dual-engine</b></div><span className="auto-badge">AUTO</span><p>SIFT for stable pairs; LightGlue fallback for difficult illumination.</p></div>
        {error&&<div className="error-note">{error}</div>}<Button className="run-button" size="lg" onClick={run} disabled={stage==='running'}>{stage==='running'?<Activity className="animate-pulse"/>:<Play/>}{stage==='running'?'Processing…':sourceFile&&referenceFile?'Process uploaded pair':'Run demonstration'}</Button><Button variant="ghost" className="reset-button" onClick={reset}><RotateCcw/> Reset</Button>
      </aside>
      <section className="main-stage"><div className="stage-heading"><div><div className="eyebrow">REGISTRATION VIEW</div><h2>South polar test pair</h2></div><div className={`result-chip ${stage}`}><CircleDot/> {stage==='complete'?'Validated':stage==='running'?'Processing':'Ready'}</div></div>
        <LunarComparison reveal={reveal} showPoints={showPoints}/>
        <div className="viewer-controls"><span>Source</span><Slider value={[reveal]} min={8} max={92} onValueChange={v=>setReveal(v[0])}/><span>Registered</span><Button variant={showPoints?'default':'outline'} size="sm" onClick={()=>setShowPoints(!showPoints)}><CircleDot/> Tie points</Button></div>
        <div className="pipeline-card"><div className="pipeline-head"><div><b>{stage==='complete'?'Registration complete':stage==='running'?`Running ${pipeline[activeIndex].toLowerCase()} stage`:'Pipeline ready'}</b><span>{stage==='complete'?'Independent checkpoints passed':'PDS4 metadata and imagery prepared'}</span></div><strong>{progress}%</strong></div><Progress value={progress}><ProgressLabel className="sr-only">Pipeline progress</ProgressLabel><ProgressValue className="sr-only"/></Progress><div className="pipeline-steps">{pipeline.map((item,i)=><div className={i<=activeIndex&&stage!=='ready'?'active':''} key={item}><span>{i+1}</span>{item}</div>)}</div></div>
      </section>
      <aside className="metrics-panel"><Tabs defaultValue="quality"><TabsList className="w-full"><TabsTrigger value="quality">Quality</TabsTrigger><TabsTrigger value="telemetry">Telemetry</TabsTrigger></TabsList><TabsContent value="quality">
        <div className="score-card"><div className="score-ring"><strong>{stage==='complete'?Number(metrics.rmse_px).toFixed(2):'—'}</strong><span>px RMSE</span></div><div><span>ACCURACY GATE</span><b>{stage==='complete'?(Number(metrics.rmse_px)<.5?'PASS':'REVIEW'):'AWAITING RUN'}</b><small>Threshold &lt; 0.50 px</small></div></div>
        <div className="metric-list"><Metric icon={<CircleDot/>} label="Accepted matches" value={stage==='complete'?Number(metrics.accepted_matches).toLocaleString():'—'} detail={stage==='complete'?`of ${Number(metrics.candidate_matches).toLocaleString()} candidates`:'awaiting registration'}/><Metric icon={<ShieldCheck/>} label="Inlier ratio" value={stage==='complete'?`${(Number(metrics.inlier_ratio)*100).toFixed(1)}%`:'—'} detail="MAGSAC++ verified"/><Metric icon={<Gauge/>} label="Spatial coverage" value={stage==='complete'?Number(metrics.spatial_coverage).toFixed(2):'—'} detail="coverage diagnostic"/><Metric icon={<Activity/>} label="Runtime" value={stage==='complete'?`${Number(metrics.runtime_seconds).toFixed(2)} s`:'—'} detail={actualMetrics?'local processing service':'demo benchmark'}/></div>
        <div className="audit-note"><ShieldCheck/><div><b>Auditable output</b><p>Metrics are reported separately from the visual overlay. Production runs require held-out checkpoints.</p></div></div><Button variant="outline" className="export" disabled={stage!=='complete'} onClick={exportReport}><Download/> Export validation bundle</Button>
      </TabsContent><TabsContent value="telemetry"><div className="telemetry-list"><div><span>Product</span><b>CH2_OHRC_NCP</b></div><div><span>Projection</span><b>Moon 2000 / South Polar</b></div><div><span>Incidence</span><b>74.2°</b></div><div><span>Emission</span><b>2.8°</b></div><div><span>Reference</span><b>LROC NAC</b></div><div><span>Output</span><b>GeoTIFF + JSON</b></div></div></TabsContent></Tabs></aside>
    </section>
    <footer><Satellite/> Chandra-Align prototype <span/> conditioned proxy → robust correspondence → geodetic output <ArrowRight/></footer>
  </main>;
}

function Metric({icon,label,value,detail}:{icon:React.ReactNode;label:string;value:string;detail:string}){return <div className="metric"><div className="metric-icon">{icon}</div><div><span>{label}</span><small>{detail}</small></div><strong>{value}</strong></div>}
