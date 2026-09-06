'use client';
import { useEffect, useRef, useState } from 'react';
import { DEMO_IMAGE, DEMO_METRICS, DEMO_POINTS, LOG_MESSAGES, PIPELINE, downloadFile, tiePointCSV, type TiePoint } from '@/lib/lunar-data';
import type { ViewMode } from '@/components/lunar-viewer';
type Fields = { sensor:string; gsd:string; sun:string; incidence:string; emission:string };
type Dataset = {name:string;file:File|null;preview:string;metadata:Fields;label:string};
type Result = {metrics:typeof DEMO_METRICS;aligned_preview:string;source_preview?:string;reference_preview?:string;tie_points?:TiePoint[];registered_geotiff_base64?:string};
const empty:Fields={sensor:'Not supplied',gsd:'Not supplied',sun:'Not supplied',incidence:'Not supplied',emission:'Not supplied'};
const sampleSource:Dataset={name:'OHRC_SOUTH_POLE_01.tif',file:null,preview:DEMO_IMAGE,metadata:{sensor:'OHRC',gsd:'0.25 m',sun:'67.4°',incidence:'74.2°',emission:'2.8°'},label:'Example PDS4 metadata'};
const sampleReference:Dataset={name:'LROC_NAC_REFERENCE.tif',file:null,preview:DEMO_IMAGE,metadata:{sensor:'LROC NAC',gsd:'2.0 m',sun:'72.1°',incidence:'69.3°',emission:'1.2°'},label:'Example PDS4 metadata'};
export function useLunarWorkspace(){
  const [source,setSource]=useState(sampleSource),[reference,setReference]=useState(sampleReference);
  const [stage,setStage]=useState<'ready'|'running'|'complete'>('complete'),[progress,setProgress]=useState(100),[activeIndex,setActiveIndex]=useState(7);
  const [mode,setMode]=useState<ViewMode>('comparison'),[selected,setSelected]=useState<TiePoint|null>(null),[showRejected,setShowRejected]=useState(true);
  const [result,setResult]=useState<Result|null>(null),[error,setError]=useState('');
  const [logs,setLogs]=useState(['[DEMO] Simulated example loaded. Run Registration to replay.']);
  const [pipelineOpen,setPipelineOpen]=useState(false),[logOpen,setLogOpen]=useState(false);
  const [settingsOpen,setSettingsOpen]=useState(false),[exportNotice,setExportNotice]=useState(false);
  const [endpoint,setEndpoint]=useState(process.env.NEXT_PUBLIC_API_URL||''),[draftEndpoint,setDraftEndpoint]=useState(endpoint);
  const [labelTarget,setLabelTarget]=useState<'source'|'reference'>('source');
  const runId=useRef(0),request=useRef<AbortController|null>(null),urls=useRef<string[]>([]);
  const uploaded=!!(source.file||reference.file),simulated=!result,completed=stage==='complete';
  const metrics=result?.metrics||DEMO_METRICS,points=completed?(result?.tie_points||(uploaded?[]:DEMO_POINTS)):[];
  const sourcePreview=result?.source_preview||source.preview,referencePreview=result?.reference_preview||reference.preview;
  useEffect(()=>()=>{runId.current++;request.current?.abort();urls.current.forEach(URL.revokeObjectURL);},[]);
  const log=(line:string)=>setLogs(l=>[...l,`[${new Date().toLocaleTimeString('en-GB',{hour12:false})}] ${line}`]);
  const invalidate=()=>{runId.current++;request.current?.abort();setStage('ready');setProgress(0);setActiveIndex(-1);setResult(null);setSelected(null);setError('');setMode('comparison');};
  const loadImage=(file:File|undefined,target:'source'|'reference')=>{
    if(!file)return;if(file.size>64*1024*1024){setError('Use an image tile smaller than 64 MB.');return;}invalidate();
    const preview=/\.(png|jpe?g|webp|bmp)$/i.test(file.name)?URL.createObjectURL(file):'';if(preview)urls.current.push(preview);
    (target==='source'?setSource:setReference)({name:file.name,file,preview,metadata:{...empty},label:''});log(`Loaded ${target}: ${file.name}${preview?'':' · preview requires backend'}`);
  };
  const loadMetadata=async(file:File|undefined)=>{
    if(!file)return;try{
      if(file.size>5*1024*1024)throw Error('PDS4 label must be smaller than 5 MB.');
      const xml=new DOMParser().parseFromString(await file.text(),'application/xml');
      if(xml.querySelector('parsererror')||!xml.documentElement.namespaceURI?.includes('pds.nasa.gov/pds4'))throw Error('Select a valid PDS4 XML label.');
      const elements=Array.from(xml.getElementsByTagName('*'));
      const read=(names:string[],angle=false)=>{const el=elements.find(e=>names.includes(e.localName.toLowerCase()));if(!el?.textContent?.trim())return 'Not supplied';const unit=el.getAttribute('unit');return `${el.textContent.trim()}${angle?(unit==='rad'?' rad':'°'):unit?' '+unit:''}`;};
      const metadata={sensor:read(['instrument_name','instrument_id','sensor_name']),gsd:read(['ground_sampling_distance','pixel_resolution','horizontal_pixel_scale']),sun:read(['solar_azimuth','solar_azimuth_angle','sun_azimuth'],true),incidence:read(['incidence_angle'],true),emission:read(['emission_angle'],true)};
      invalidate();(labelTarget==='source'?setSource:setReference)(d=>({...d,metadata,label:file.name}));log(`PDS4 metadata read: ${file.name} (${labelTarget})`);
    }catch(e){setError(e instanceof Error?e.message:'Metadata could not be read.');}
  };
  const run=async()=>{
    if(stage==='running')return;
    if(uploaded&&(!source.file||!reference.file)){setError('Load both source and reference images to process your pair.');return;}
    if(uploaded&&!endpoint){setError('Connect your processing service in Settings to register uploaded images. The sample pair runs locally as a simulation.');return;}
    const id=++runId.current;setError('');setResult(null);setSelected(null);setStage('running');setProgress(0);setActiveIndex(0);setPipelineOpen(true);setLogs([]);setMode('comparison');
    if(uploaded){
      log('Sending image pair to processing service');setProgress(15);setActiveIndex(1);const controller=new AbortController();request.current=controller;
      const timeout=setTimeout(()=>controller.abort(),120000);
      try{
        const form=new FormData();form.append('source',source.file!);form.append('reference',reference.file!);
        const response=await fetch(`${endpoint.replace(/\/$/,'')}/register`,{method:'POST',body:form,signal:controller.signal});const data=await response.json() as Result & {detail?:string};
        if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:'Registration service returned an error.');
        if(!data?.metrics||typeof data.aligned_preview!=='string'||!(['rmse_px','accepted_matches','candidate_matches','inlier_ratio','spatial_coverage','runtime_seconds'] as const).every(k=>Number.isFinite(data.metrics[k])))throw Error('The service response is missing a registered preview or numeric quality metrics.');
        if(data.tie_points&&!data.tie_points.every(p=>typeof p.id==='string'&&[p.source_x,p.source_y,p.reference_x,p.reference_y].every(Number.isFinite)&&['accepted','rejected'].includes(p.status)&&(p.confidence===null||Number.isFinite(p.confidence))))throw Error('The service returned invalid tie-point coordinates.');
        if(id!==runId.current)return;setResult(data);setStage('complete');setProgress(100);setActiveIndex(7);log(`Registration complete · ${data.metrics.registration_method||data.metrics.engine}`);log('Metrics returned by backend. Inspect model residuals and coverage.');
      }catch(e){if(id!==runId.current)return;setStage('ready');setProgress(0);setActiveIndex(-1);const message=e instanceof Error?e.message:'Registration failed';setError(message);log(`Failed: ${message}`);}finally{clearTimeout(timeout);}return;
    }
    log('SIMULATED PROTOTYPE RUN · example metadata and correspondences');
    for(let i=0;i<PIPELINE.length;i++){if(id!==runId.current)return;setActiveIndex(i);log(LOG_MESSAGES[i]);await new Promise(r=>setTimeout(r,[400,430,500,570,420,470,390][i]));if(id!==runId.current)return;setProgress(Math.round((i+1)/7*100));}
    setActiveIndex(7);setStage('complete');log('Registration complete · simulated results');
  };
  const resetDemo=()=>{invalidate();setSource(sampleSource);setReference(sampleReference);setStage('complete');setProgress(100);setActiveIndex(7);setLogs(['[DEMO] Simulated example loaded.']);};
  const demoRun=useRef(run);
  demoRun.current=async()=>{if(uploaded)throw Error('Load the sample pair before running the demonstration tool.');await run();};
  useEffect(()=>{
    const context=(document as Document & {modelContext?:{registerTool:(tool:unknown,options?:{signal?:AbortSignal})=>void|Promise<void>}}).modelContext;
    if(!context?.registerTool)return;const lifecycle=new AbortController();
    void Promise.resolve(context.registerTool({name:'run_lunar_coregistration_demo',title:'Run lunar co-registration demo',description:'Replay the visible simulated registration pipeline. Requires the sample pair.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:false,untrustedContentHint:false},execute:async()=>{await demoRun.current();return{mode:'demonstration',notice:'SIMULATED PROTOTYPE RESULTS'};}},{signal:lifecycle.signal})).catch(()=>{});
    return()=>lifecycle.abort();
  },[]);
  const exportReport=()=>{
    const report={title:'LunarGrid / Chandra-Align Registration Report',result_type:simulated?'SIMULATED PROTOTYPE RESULTS':'BACKEND RESULTS',created_at:new Date().toISOString(),source_dataset:source.name,reference_dataset:reference.name,source_metadata:source.metadata,reference_metadata:reference.metadata,source_pds4_label:source.label||null,reference_pds4_label:reference.label||null,registration_method:metrics.registration_method||metrics.engine,matching_engine:metrics.engine,accepted_matches:metrics.accepted_matches,rejected_matches:metrics.candidate_matches-metrics.accepted_matches,inlier_ratio:metrics.inlier_ratio,rmse_px:metrics.rmse_px,spatial_coverage:metrics.spatial_coverage,runtime_seconds:metrics.runtime_seconds,tie_point_count:points.length,coordinate_system:'Raster pixels; top-left origin; no lunar georeferencing asserted',validation_note:simulated?'Simulated values, not experimentally achieved. Dataset names and acquisition metadata are illustrative. Photograph: Tycho crater, LROC WAC, NASA/GSFC/Arizona State University.':metrics.validation_note};
    downloadFile(JSON.stringify(report,null,2),`lunargrid-${simulated?'SIMULATED-':''}registration-report.json`,'application/json');log('Exported registration report');
  };
  const exportCSV=()=>{downloadFile(tiePointCSV(points,simulated),`lunargrid-${simulated?'SIMULATED-':''}tie-points.csv`,'text/csv');log('Exported tie points CSV');};
  const exportGeoTiff=()=>{if(!result?.registered_geotiff_base64){setExportNotice(true);return;}try{const bytes=Uint8Array.from(atob(result.registered_geotiff_base64),c=>c.charCodeAt(0));if(!((bytes[0]===73&&bytes[1]===73)||(bytes[0]===77&&bytes[1]===77)))throw Error();downloadFile(new Blob([bytes],{type:'image/tiff'}),'lunargrid-registered.tif');log('Exported backend registered GeoTIFF');}catch{setError('The backend GeoTIFF could not be decoded.');}};
  const saveSettings=()=>{try{if(draftEndpoint){const url=new URL(draftEndpoint);if(!['http:','https:'].includes(url.protocol)||url.username||url.password||url.search||url.hash)throw Error();}setEndpoint(draftEndpoint.replace(/\/$/,''));setSettingsOpen(false);setError('');}catch{setError('Use an HTTP or HTTPS service URL without embedded credentials or query parameters.');setSettingsOpen(false);}};
  const meters=(s:string)=>{const match=s.match(/^([\d.]+)\s*(m|km|cm)(?:\s*\/\s*(?:px|pixel))?$/i);return match?Number(match[1])*({m:1,km:1000,cm:.01}[match[2].toLowerCase()]||1):NaN;};
  const a=meters(source.metadata.gsd),b=meters(reference.metadata.gsd),scaleGap=a>0&&b>0?`${(Math.max(a,b)/Math.min(a,b)).toFixed(1)}×`:'—';
  return {source,reference,stage,progress,activeIndex,mode,setMode,selected,setSelected,showRejected,setShowRejected,result,error,logs,pipelineOpen,setPipelineOpen,logOpen,setLogOpen,settingsOpen,setSettingsOpen,exportNotice,setExportNotice,endpoint,draftEndpoint,setDraftEndpoint,labelTarget,setLabelTarget,uploaded,simulated,completed,metrics,points,sourcePreview,referencePreview,loadImage,loadMetadata,run,resetDemo,exportReport,exportCSV,exportGeoTiff,saveSettings,scaleGap};
}
