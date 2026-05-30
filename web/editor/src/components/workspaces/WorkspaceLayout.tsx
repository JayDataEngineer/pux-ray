import type { WorkflowRun, WorkflowSpec } from '../../types'
import { useState } from 'react'
import { SidebarProvider, Sidebar, SidebarHeader, SidebarContent, SidebarGroup, SidebarGroupLabel, SidebarMenu, SidebarMenuItem, SidebarMenuButton, SidebarTrigger, useSidebar } from '../ui/sidebar'
import { FolderOpen, Plus, Music, Play, Trash2, ChevronDown, ChevronRight, Image, Video, Mic, Volume2, Wand2, Layers } from 'lucide-react'
import { useAssetStore, CATEGORY_LABEL, CATEGORY_ORDER, type AssetCategory } from '../../stores/assets'
import { useTimelineStore } from '../../stores/timeline'
import { useToastStore } from '../../stores/toast'


type Tab = 'assets' | 'video'

const CATEGORY_ICONS: Record<AssetCategory, typeof Image> = {
  image: Image, music: Music, voice: Mic, sfx: Volume2, video: Video, other: FolderOpen,
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────
function AppSidebar() {
  const { state } = useSidebar()
  const collapsed = state === 'collapsed'
  const assets = useAssetStore((s) => s.assets)
  const removeAsset = useAssetStore((s) => s.removeAsset)
  const addAsset = useAssetStore((s) => s.addAsset)
  const addAudioCue = useTimelineStore((s) => s.addAudioCue)
  const toast = useToastStore((s) => s.addToast)
  const [expandedCats, setExpandedCats] = useState<Set<AssetCategory>>(new Set(['image']))
  const [playingId, setPlayingId] = useState<string | null>(null)

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const dataUrl = reader.result as string
      const t = file.type.startsWith('image/') ? 'image' as const : file.type.startsWith('audio/') ? 'audio' as const : file.type.startsWith('video/') ? 'video' as const : 'other' as const
      const cat: AssetCategory = t === 'audio' ? 'other' : t
      addAsset({ name: file.name, type: t, category: cat, mediaType: file.type, url: dataUrl, sizeBytes: file.size, source: 'uploaded' })
      if (t === 'audio') addAudioCue({ track: 'sfx', start: 0, duration: 5, label: file.name, audioUrl: dataUrl, volume: 0.8, waveformPeaks: null, sourceStepId: null })
      toast('info', `"${file.name}" added`)
    }
    reader.readAsDataURL(file)
  }

  return (
    <>
      <SidebarHeader>
        {!collapsed && <span className="text-xs font-medium tracking-wider text-accent uppercase">ASSETS</span>}
        <div className="flex gap-0.5 ml-auto">
          <label className="flex h-7 w-7 items-center justify-center rounded-sm text-muted-foreground hover:text-accent cursor-pointer"><Plus size={16} /><input type="file" className="hidden" accept="image/*,audio/*,video/*" onChange={handleImport} /></label>
          <SidebarTrigger />
        </div>
      </SidebarHeader>
      <SidebarContent>
        {assets.length === 0 && !collapsed && <div className="px-3 py-6 text-xs text-muted-foreground text-center">No assets yet<br/>Generate or Import</div>}
        {CATEGORY_ORDER.map((cat) => {
          const items = assets.filter((a) => a.category === cat)
          if (items.length === 0) return null
          const Icon = CATEGORY_ICONS[cat]
          const expanded = expandedCats.has(cat)
          if (collapsed) return <SidebarMenu key={cat}><SidebarMenuItem><SidebarMenuButton isActive={cat==='image'}><Icon size={16} /></SidebarMenuButton></SidebarMenuItem></SidebarMenu>
          return (
            <SidebarGroup key={cat}>
              <SidebarGroupLabel className="cursor-pointer flex items-center gap-1 text-[10px]" onClick={() => setExpandedCats((p) => {const n=new Set(p); n.has(cat)?n.delete(cat):n.add(cat); return n})}>
                {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}<Icon size={12} />{CATEGORY_LABEL[cat]}<span className="ml-auto text-[9px] bg-muted px-1 rounded">{items.length}</span>
              </SidebarGroupLabel>
              {expanded && (cat === 'image' ? (
                <div className="grid grid-cols-2 gap-1">
                  {items.map((a) => (
                    <div key={a.id} className="relative group cursor-pointer border border-border hover:border-accent/50"
                      draggable onDragStart={(e) => { e.dataTransfer.setData('application/tech-noir-asset', JSON.stringify({url:a.url,type:a.type,name:a.name,id:a.id})); e.dataTransfer.setData('text/plain',a.url); e.dataTransfer.effectAllowed='copy' }}
                      onDoubleClick={() => { const o=document.createElement('div');o.className='fixed inset-0 z-50 bg-black/90 flex items-center justify-center cursor-pointer';o.onclick=()=>o.remove();const i=document.createElement('img');i.src=a.url;i.className='max-w-[90vw] max-h-[90vh] object-contain';o.appendChild(i);document.body.appendChild(o) }}>
                      <img src={a.url} alt={a.name} className="w-full aspect-[4/5] object-cover" draggable={false} />
                      <span className="absolute bottom-0 inset-x-0 bg-black/70 px-1 py-0.5 text-[8px] truncate">{a.name.slice(0,14)}</span>
                      <button className="absolute top-0.5 right-0.5 hidden group-hover:flex bg-black/70 rounded p-0.5 text-destructive" onClick={(e)=>{e.stopPropagation();removeAsset(a.id)}}><Trash2 size={10}/></button>
                    </div>
                  ))}
                </div>
              ) : (
                items.map((a) => (
                  <div key={a.id} className={`flex items-center gap-1.5 px-1 py-0.5 text-xs hover:bg-accent/10 rounded cursor-pointer ${playingId===a.id?'bg-accent/10':''}`}>
                    <button className="p-0.5 hover:text-accent" onClick={() => setPlayingId((p)=>p===a.id?null:a.id)}>{playingId===a.id?'⏸':<Play size={10}/>}</button>
                    <Icon size={12} /><span className="flex-1 truncate">{a.name}</span>
                    <button className="p-0.5 hover:text-destructive opacity-0 group-hover:opacity-100" onClick={()=>removeAsset(a.id)}><Trash2 size={10}/></button>
                  </div>
                ))
              ))}
            </SidebarGroup>
          )
        })}
      </SidebarContent>
    </>
  )
}

// ─── Audio DAG Panel ──────────────────────────────────────────────────────────
const AUDIO_TASKS: any[] = [
  { id: 'ace_step', label: 'ACE-Step Music', icon: Music, pipeline: undefined, service: 'wan2gp', model: 'tts/ace_step_v1_5', params: [{n:'input_prompt',t:'string',l:'Prompt',p:'epic cinematic orchestral, 120bpm'},{n:'duration_seconds',t:'number',l:'Duration (s)',p:'30',d:30}] },
  { id: 'moss_sfx', pipeline: undefined, label: 'MOSS Sound Effect', icon: Volume2, service: 'wan2gp', model: 'moss/moss-soundeffect', params: [{n:'input_prompt',t:'string',l:'Sound Description',p:'rain and thunder'},{n:'duration_seconds',t:'number',l:'Duration (s)',p:'5',d:5}] },
  { id: 'moss_voice_clone', pipeline: undefined, label: 'Voice Clone', icon: Mic, service: 'wan2gp', model: 'moss/moss-tts', params: [{n:'text',t:'string',l:'Text',p:'Hello world'},{n:'reference_audio_b64',t:'audio',l:'Reference Audio',p:'Upload voice sample'}] },
  { id: 'moss_voice_gen', pipeline: undefined, label: 'Voice Generator', icon: Wand2, service: 'wan2gp', model: 'moss/moss-voicegenerator', params: [{n:'input_prompt',t:'string',l:'Voice Description',p:'deep British male voice, authoritative'}] },
  { id: 'kokoro', pipeline: undefined, label: 'Kokoro TTS', icon: Mic, service: 'wan2gp', model: 'kokoro', params: [{n:'text',t:'string',l:'Text',p:'Hello world'},{n:'voice',t:'string',l:'Voice',p:'af_bella',d:'af_bella'}] },
]

const IMAGE_TASKS: any[] = [
  { id: 'z_image', label: 'Z-Image Turbo', icon: Image, pipeline: 'tech-noir/generate', params: [{n:'prompt',t:'string',l:'Prompt',p:'A cyberpunk samurai...'},{n:'quality',t:'select',l:'Quality',opts:['turbo','standard'],d:'turbo'}] },
  { id: 'z_image_base', label: 'Z-Image Base', icon: Image, pipeline: 'tech-noir/generate', params: [{n:'prompt',t:'string',l:'Prompt',p:'A cyberpunk samurai...'},{n:'quality',t:'select',l:'Quality',opts:['turbo','standard'],d:'standard'}] },
  { id: 'vnccs_pose', label: 'VNCCS Pose Edit', icon: Layers, pipeline: 'vnccs/pose-edit', params: [{n:'character_image_b64',t:'image',l:'Character Image',p:'Select or drag from assets'},{n:'rotations',t:'string',l:'Rotations JSON',p:'{}',d:'{}'}] },
  { id: 'ltx_video', label: 'LTX Video', icon: Video, pipeline: undefined, service: 'wan2gp', model: 'ltx2', params: [{n:'input_prompt',t:'string',l:'Prompt',p:'walking forward, cinematic'},{n:'image_b64',t:'image',l:'Start Frame',p:'Select keyframe image'},{n:'frame_num',t:'number',l:'Frames',p:'97',d:97},{n:'fps',t:'number',l:'FPS',p:'24',d:24}] },
]

// ─── Assets Tab ───────────────────────────────────────────────────────────────
function AssetsTab() {
  const toast = useToastStore((s) => s.addToast)
  const addAsset = useAssetStore((s) => s.addAsset)
  const [tabCat, setTabCat] = useState<'audio'|'image'>('image')
  const tasks = tabCat === 'audio' ? AUDIO_TASKS : IMAGE_TASKS
  const [selected, setSelected] = useState(tasks[0])
  const [paramVals, setParamVals] = useState<Record<string, string|number>>(() => { const d:Record<string,string|number>={}; selected.params.forEach((p: any)=> {if(p.d!==undefined)d[p.n]=p.d}); return d })
  const [generating, setGenerating] = useState(false)

  const handleTaskChange = (task: typeof tasks[0]) => {
    setSelected(task)
    const d: Record<string,string|number> = {}
    task.params.forEach((p: any) => { if (p.d !== undefined) d[p.n] = p.d })
    setParamVals(d)
  }

  const handleGenerate = async () => {
    setGenerating(true)
    try {
      const payload = selected.pipeline
        ? { pipeline: selected.pipeline, params: paramVals }
        : { service: selected.service!, model: selected.model!, ...paramVals }
      const res = await fetch('/v1/run', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) })
      const text = await res.text()
      let result: Record<string,unknown> = {}
      try { result = JSON.parse(text) } catch { toast('error', text.slice(0, 200)); setGenerating(false); return }
      if (result.status === 'ok' || result.status === 'success') {
        if (result.data) {
          const mt = result.media_type as string || 'image/png'
          const ext = mt.includes('audio') ? 'wav' : mt.includes('video') ? 'mp4' : 'png'
          const dataUrl = `data:${mt};base64,${result.data}`
          const cat: AssetCategory = selected.id.includes('ace') ? 'music' : selected.id.includes('sfx') ? 'sfx' : selected.id.includes('voice') ? 'voice' : 'image'
          addAsset({ name: `${selected.label} ${new Date().toLocaleTimeString()}`, type: ext==='wav'?'audio':'image', category: cat, mediaType: mt, url: dataUrl, sizeBytes: Math.round((result.data as string).length*0.75), source: 'generated' })
          toast('success', `${selected.label} generated`)
        }
      } else {
        toast('error', `Failed: ${result.error || result.message || 'Unknown'}`)
      }
    } catch (e) { toast('error', `Error: ${e instanceof Error ? e.message : String(e)}`) }
    finally { setGenerating(false) }
  }

  return (
    <div className="flex-1 flex flex-col p-4 overflow-y-auto bg-background">
      <div className="flex gap-1 mb-4">
        <button className={`px-4 py-1.5 text-xs font-medium rounded ${tabCat==='image'?'bg-accent/20 text-accent':'text-muted-foreground hover:text-foreground'}`} onClick={()=>setTabCat('image')}><Image size={14} className="inline mr-1.5"/>Image</button>
        <button className={`px-4 py-1.5 text-xs font-medium rounded ${tabCat==='audio'?'bg-accent/20 text-accent':'text-muted-foreground hover:text-foreground'}`} onClick={()=>setTabCat('audio')}><Music size={14} className="inline mr-1.5"/>Audio</button>
      </div>

      <div className="flex flex-col gap-1 mb-4">
        {tasks.map((t) => { const Icon = t.icon; return (
          <button key={t.id} className={`flex items-center gap-2 px-3 py-2 text-left text-sm rounded border transition-colors ${selected.id===t.id?'border-accent bg-accent/10 text-accent':'border-border text-muted-foreground hover:border-input'}`} onClick={()=>handleTaskChange(t)}>
            <Icon size={16} />{t.label}
          </button>
        )})}
      </div>

      <div className="flex flex-col gap-3 flex-1 overflow-y-auto">
        {selected.params.map((p: any) => (
          <div key={p.n} className="flex flex-col gap-1">
            <label className="text-[11px] font-medium text-muted-foreground">{p.l}</label>
            {p.t === 'select' && p.opts ? (
              <select className="bg-card border border-border rounded px-2 py-1.5 text-sm text-foreground focus:border-accent outline-none" value={String(paramVals[p.n]??p.d??'')} onChange={e=>setParamVals(prev=>({...prev,[p.n]:e.target.value}))}>
                {p.opts.map((o: string)=><option key={o} value={o}>{o}</option>)}
              </select>
            ) : p.t === 'number' ? (
              <input type="number" className="bg-card border border-border rounded px-2 py-1.5 text-sm text-foreground focus:border-accent outline-none" value={paramVals[p.n]??p.d??''} onChange={e=>setParamVals(prev=>({...prev,[p.n]:parseInt(e.target.value)||0}))} placeholder={p.p} />
            ) : p.t === 'image' || p.t === 'audio' ? (
              <label className="flex items-center justify-center h-20 border border-dashed border-input rounded text-xs text-muted-foreground hover:border-accent/50 cursor-pointer">
                {paramVals[p.n] ? 'File loaded' : p.p}
                <input type="file" className="hidden" accept={p.t==='image'?'image/*':'audio/*'} onChange={e=>{const f=e.target.files?.[0];if(!f)return;const r=new FileReader();r.onload=()=>{const d=r.result as string;setParamVals(prev=>({...prev,[p.n]:d.includes(',')?d.split(',')[1]:d}))};r.readAsDataURL(f)}} />
              </label>
            ) : (
              <textarea className="bg-card border border-border rounded px-2 py-1.5 text-sm text-foreground focus:border-accent outline-none resize-none" rows={3} value={String(paramVals[p.n]??'')} onChange={e=>setParamVals(prev=>({...prev,[p.n]:e.target.value}))} placeholder={p.p} />
            )}
          </div>
        ))}
        <button className="mt-2 bg-accent text-accent-foreground font-semibold py-2 rounded text-sm hover:bg-accent/80 transition-colors disabled:opacity-50" disabled={generating} onClick={handleGenerate}>
          {generating ? 'Generating...' : `Generate ${selected.label}`}
        </button>
      </div>
    </div>
  )
}

// ─── Video Tab ────────────────────────────────────────────────────────────────
function VideoTab() {
  const segments = useTimelineStore((s) => s.segments)
  const audioCues = useTimelineStore((s) => s.audioCues)
  const addSegment = useTimelineStore((s) => s.addSegment)
  const updateSegment = useTimelineStore((s) => s.updateSegment)
  const selectedSegmentId = useTimelineStore((s) => s.selectedSegmentId)
  const setSelectedSegment = useTimelineStore((s) => s.setSelectedSegment)
  const toast = useToastStore((s) => s.addToast)
  const addAsset = useAssetStore((s) => s.addAsset)
  const [generating, setGenerating] = useState(false)
  const selectedSegment = segments.find((s) => s.id === selectedSegmentId)

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    try { const d = JSON.parse(e.dataTransfer.getData('application/tech-noir-asset'))
      if (d.type === 'image') { const s = addSegment({ prompt: d.name, firstFrameB64: d.url, thumbnailUrl: d.url, status: 'empty' }); setSelectedSegment(s.id); toast('info', `Added keyframe: ${d.name}`) }
    } catch {}
  }

  const handleGenerateVideo = async () => {
    if (!selectedSegment?.firstFrameB64) { toast('error', 'Select a keyframe with an image first'); return }
    setGenerating(true)
    const seg = selectedSegment
    try {
      const b64 = seg.firstFrameB64!.includes(',') ? seg.firstFrameB64!.split(',')[1] : seg.firstFrameB64!
      const res = await fetch('/v1/run', { method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ service: 'wan2gp', model: 'ltx2', input_prompt: seg.prompt || 'cinematic motion', image_b64: b64, seed: 42, fps: 24, frame_num: Math.round(seg.duration * 24), guide_scale: 3.0, width: 768, height: 512, sample_solver: 'euler' }) })
      const text = await res.text()
      let result: Record<string,unknown> = {}
      try { result = JSON.parse(text) } catch { toast('error', text.slice(0,200)); setGenerating(false); return }
      if (result.status === 'ok' && result.data) {
        const videoUrl = `data:video/mp4;base64,${result.data}`
        updateSegment(seg.id, { videoUrl, status: 'ready' })
        addAsset({ name: `LTX Video ${seg.order+1}`, type: 'video', category: 'video', mediaType: 'video/mp4', url: videoUrl, sizeBytes: Math.round((result.data as string).length*0.75), source: 'generated' })
        toast('success', `Video K_${String(seg.order+1).padStart(2,'0')} generated`)
      } else {
        toast('error', `Video failed: ${result.error || 'Unknown'}`)
      }
    } catch (e) { toast('error', `Error: ${e instanceof Error ? e.message : String(e)}`) }
    finally { setGenerating(false) }
  }

  return (
    <div className="flex-1 flex flex-col p-4 overflow-hidden bg-background">
      <div className="flex-1 flex flex-col items-center justify-center border border-border rounded bg-card mb-3" onDragOver={e=>e.preventDefault()} onDrop={onDrop}>
        {selectedSegment?.videoUrl ? (
          <video src={selectedSegment.videoUrl} controls className="max-w-full max-h-full" />
        ) : segments.length === 0 ? (
          <p className="text-muted-foreground text-sm">Drag images from Assets sidebar to add keyframes</p>
        ) : (
          <p className="text-muted-foreground text-sm">{segments.length} keyframe(s) — select one and generate video</p>
        )}
      </div>
      <div className="h-32 border border-border rounded bg-card p-2 flex items-center gap-1 overflow-x-auto">
        {segments.map((seg) => (
          <div key={seg.id} className={`h-full min-w-[60px] flex items-center justify-center bg-muted border rounded text-[10px] cursor-pointer relative overflow-hidden ${seg.id===selectedSegmentId?'border-accent':'border-input'}`}
            style={{width:`${seg.duration*40}px`}} onClick={()=>setSelectedSegment(seg.id)}>
            {seg.thumbnailUrl && <img src={seg.thumbnailUrl} alt="" className="absolute inset-0 w-full h-full object-cover opacity-40" />}
            <span className="relative z-10 text-foreground">K_{String(seg.order+1).padStart(2,'0')}</span>
          </div>
        ))}
        <button className="h-full min-w-[30px] flex items-center justify-center border border-dashed border-input rounded text-muted-foreground hover:border-accent/50" onClick={()=>{const s=addSegment({duration:5,status:'empty'});setSelectedSegment(s.id)}}>+</button>
      </div>
      {selectedSegment && (
        <div className="mt-3 flex flex-col gap-2">
          <input className="bg-card border border-border rounded px-2 py-1 text-sm text-foreground" placeholder="Video prompt (e.g. walking forward, cinematic)" value={selectedSegment.prompt}
            onChange={(e) => updateSegment(selectedSegment.id, { prompt: e.target.value })} />
          <div className="flex gap-2">
            <input type="number" className="bg-card border border-border rounded px-2 py-1 w-20 text-sm text-foreground" placeholder="Duration" value={selectedSegment.duration}
              onChange={(e) => updateSegment(selectedSegment.id, { duration: parseFloat(e.target.value) || 5 })} />
            <button className="flex-1 bg-accent text-accent-foreground font-semibold py-1 rounded text-sm hover:bg-accent/80 disabled:opacity-50" disabled={generating || !selectedSegment.firstFrameB64} onClick={handleGenerateVideo}>
              {generating ? 'Generating...' : 'Generate LTX Video'}
            </button>
          </div>
        </div>
      )}
      {audioCues.length > 0 && <div className="mt-1 text-[10px] text-muted-foreground">{audioCues.length} audio cue(s) from generated assets</div>}
    </div>
  )
}

// ─── Layout ───────────────────────────────────────────────────────────────────
interface Props {
  spec: WorkflowSpec; run: WorkflowRun | null; onSpecChange: (name: string) => void
  allSpecs: { name: string; description: string; steps: number }[]
}

export function WorkspaceLayout({ spec: _spec, run: _run, onSpecChange: _oc, allSpecs: _as }: Props) {
  const [tab, setTab] = useState<Tab>('assets')
  return (
    <SidebarProvider defaultOpen={true}>
      <div className="flex h-screen w-full bg-background text-foreground">
        <Sidebar><AppSidebar /></Sidebar>
        <div className="flex-1 flex flex-col min-w-0">
          <header className="flex items-center justify-between h-11 px-3 border-b border-border flex-shrink-0">
            <div className="flex items-center gap-6">
              <span className="text-sm font-bold text-accent tracking-tight">TECH NOIR</span>
              <nav className="flex gap-0">
                <button className={`px-3 h-11 text-xs font-medium border-b-2 transition-colors ${tab==='assets'?'border-accent text-accent':'border-transparent text-muted-foreground hover:text-foreground'}`} onClick={()=>setTab('assets')}>Assets</button>
                <button className={`px-3 h-11 text-xs font-medium border-b-2 transition-colors ${tab==='video'?'border-accent text-accent':'border-transparent text-muted-foreground hover:text-foreground'}`} onClick={()=>setTab('video')}>Video</button>
              </nav>
            </div>
          </header>
          {tab === 'assets' ? <AssetsTab /> : <VideoTab />}
        </div>
      </div>
    </SidebarProvider>
  )
}
