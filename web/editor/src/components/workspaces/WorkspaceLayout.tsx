import { useState } from 'react'
import { SidebarProvider, Sidebar, SidebarHeader, SidebarContent, SidebarGroup, SidebarGroupLabel, SidebarMenu, SidebarMenuItem, SidebarMenuButton, SidebarTrigger, useSidebar } from '@/components/ui/sidebar'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { ScrollArea } from '@/components/ui/scroll-area'
import { FolderOpen, Plus, Music, Play, Trash2, ChevronDown, ChevronRight, Image, Video, Mic, Volume2, Wand2, Layers } from 'lucide-react'
import { useAssetStore, CATEGORY_LABEL, CATEGORY_ORDER, type AssetCategory } from '@/stores/assets'
import { useTimelineStore } from '@/stores/timeline'
import { useToastStore } from '@/stores/toast'

const CATEGORY_ICONS: Record<AssetCategory, typeof Image> = { image: Image, music: Music, voice: Mic, sfx: Volume2, video: Video, other: FolderOpen }
const AUDIO_TASKS: any[] = [
  { id: 'ace_step', label: 'ACE-Step Music', icon: Music, model: 'tts/ace_step_v1_5', params: [{n:'input_prompt',t:'text',l:'Prompt',p:'epic cinematic orchestral, 120bpm'},{n:'duration_seconds',t:'number',l:'Duration (s)',p:'30',d:30}] },
  { id: 'moss_sfx', label: 'MOSS Sound Effect', icon: Volume2, model: 'moss/moss-soundeffect', params: [{n:'input_prompt',t:'text',l:'Description',p:'rain and thunder'},{n:'duration_seconds',t:'number',l:'Duration (s)',p:'5',d:5}] },
  { id: 'moss_voice_clone', label: 'Voice Clone', icon: Mic, model: 'moss/moss-tts', params: [{n:'text',t:'text',l:'Text',p:'Hello world'},{n:'reference_audio_b64',t:'file',l:'Reference Audio',p:'Upload voice sample'}] },
  { id: 'moss_voice_gen', label: 'Voice Generator', icon: Wand2, model: 'moss/moss-voicegenerator', params: [{n:'input_prompt',t:'text',l:'Voice Description',p:'deep British male voice, authoritative'}] },
  { id: 'kokoro', label: 'Kokoro TTS', icon: Mic, model: 'kokoro', params: [{n:'text',t:'text',l:'Text',p:'Hello world'},{n:'voice',t:'string',l:'Voice',p:'af_bella',d:'af_bella'}] },
]
const IMAGE_TASKS: any[] = [
  { id: 'z_image', label: 'Z-Image Turbo', icon: Image, pipeline: 'tech-noir/generate', params: [{n:'prompt',t:'text',l:'Prompt',p:'A cyberpunk samurai...'},{n:'quality',t:'select',l:'Quality',opts:['turbo','standard'],d:'turbo'}] },
  { id: 'z_image_base', label: 'Z-Image Base', icon: Image, pipeline: 'tech-noir/generate', params: [{n:'prompt',t:'text',l:'Prompt',p:'A cyberpunk samurai...'},{n:'quality',t:'select',l:'Quality',opts:['turbo','standard'],d:'standard'}] },
  { id: 'vnccs_pose', label: 'VNCCS Pose Edit', icon: Layers, pipeline: 'vnccs/pose-edit', params: [{n:'character_image_b64',t:'file',l:'Character Image',p:'Select from assets'},{n:'rotations',t:'string',l:'Rotations JSON',p:'{}',d:'{}'}] },
]

// ─── Sidebar ──────────────────────────────────────────────────────────────
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
      const t = file.type.startsWith('image/') ? 'image' as const : file.type.startsWith('audio/') ? 'audio' as const : 'video' as const
      const cat: AssetCategory = t === 'audio' ? 'other' : t
      addAsset({ name: file.name, type: t, category: cat, mediaType: file.type, url: dataUrl, sizeBytes: file.size, source: 'uploaded' })
      if (t === 'audio') addAudioCue({ track: 'sfx', start: 0, duration: 5, label: file.name, audioUrl: dataUrl, volume: 0.8, waveformPeaks: null, sourceStepId: null })
      toast('info', `Added: ${file.name}`)
    }
    reader.readAsDataURL(file)
  }

  return (
    <>
      <SidebarHeader>
        {!collapsed && <span className="font-semibold text-sm tracking-tight">Assets</span>}
        <div className="flex gap-1 ml-auto">
          <label className="cursor-pointer inline-flex items-center justify-center h-7 w-7 rounded-md hover:bg-accent hover:text-accent-foreground"><Plus className="h-4 w-4" /><input type="file" className="hidden" accept="image/*,audio/*,video/*" onChange={handleImport} /></label>
          <SidebarTrigger />
        </div>
      </SidebarHeader>
      <SidebarContent>
        <ScrollArea>
          {assets.length === 0 && !collapsed && (
            <div className="px-4 py-8 text-center text-sm text-muted-foreground">No assets yet.<br/>Generate or import to begin.</div>
          )}
          {CATEGORY_ORDER.map((cat) => {
            const items = assets.filter((a) => a.category === cat)
            if (items.length === 0) return null
            const Icon = CATEGORY_ICONS[cat]
            const expanded = expandedCats.has(cat)
            if (collapsed) return (
              <SidebarMenu key={cat}>
                <SidebarMenuItem><SidebarMenuButton isActive={cat==='image'}><Icon className="h-4 w-4" /></SidebarMenuButton></SidebarMenuItem>
              </SidebarMenu>
            )
            return (
              <SidebarGroup key={cat}>
                <SidebarGroupLabel className="cursor-pointer" onClick={() => setExpandedCats((p) => {const n=new Set(p); n.has(cat)?n.delete(cat):n.add(cat); return n})}>
                  {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                  <Icon className="h-3 w-3" />{CATEGORY_LABEL[cat]}
                  <Badge variant="secondary" className="ml-auto h-4 px-1 text-[10px]">{items.length}</Badge>
                </SidebarGroupLabel>
                {expanded && (cat === 'image' ? (
                  <div className="grid grid-cols-2 gap-1 mt-1">
                    {items.map((a) => (
                      <div key={a.id} className="relative group cursor-pointer rounded-md border border-border hover:border-primary/50 overflow-hidden"
                        draggable onDragStart={(e) => { e.dataTransfer.setData('application/tech-noir-asset', JSON.stringify({url:a.url,type:a.type,name:a.name,id:a.id})); e.dataTransfer.effectAllowed='copy' }}
                        onDoubleClick={() => { const o=document.createElement('div');o.className='fixed inset-0 z-50 bg-background/95 flex items-center justify-center cursor-zoom-out';o.onclick=()=>o.remove();const i=document.createElement('img');i.src=a.url;i.className='max-w-[90vw] max-h-[90vh] object-contain rounded-lg';o.appendChild(i);document.body.appendChild(o) }}>
                        <img src={a.url} alt={a.name} className="w-full aspect-[4/5] object-cover" draggable={false} />
                        <span className="absolute bottom-0 inset-x-0 bg-background/80 px-1.5 py-0.5 text-[9px] truncate">{a.name.slice(0,14)}</span>
                        <Button variant="ghost" size="icon" className="absolute top-0.5 right-0.5 h-5 w-5 hidden group-hover:flex bg-background/80" onClick={(e)=>{e.stopPropagation();removeAsset(a.id)}}><Trash2 className="h-3 w-3 text-destructive" /></Button>
                      </div>
                    ))}
                  </div>
                ) : (
                  items.map((a) => (
                    <div key={a.id} className={`flex items-center gap-2 px-2 py-1 text-sm rounded-md hover:bg-accent hover:text-accent-foreground cursor-pointer ${playingId===a.id?'bg-accent/20':''}`}>
                      <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setPlayingId((p)=>p===a.id?null:a.id)}>
                        {playingId===a.id ? <span className="text-xs">⏸</span> : <Play className="h-3 w-3" />}
                      </Button>
                      <Icon className="h-3 w-3 shrink-0" />
                      <span className="flex-1 truncate">{a.name}</span>
                      <Button variant="ghost" size="icon" className="h-6 w-6 opacity-0 group-hover:opacity-100" onClick={()=>removeAsset(a.id)}><Trash2 className="h-3 w-3 text-destructive" /></Button>
                    </div>
                  ))
                ))}
              </SidebarGroup>
            )
          })}
        </ScrollArea>
      </SidebarContent>
    </>
  )
}

// ─── Assets Tab ───────────────────────────────────────────────────────────
function AssetsTab() {
  const toast = useToastStore((s) => s.addToast)
  const addAsset = useAssetStore((s) => s.addAsset)
  const [selected, setSelected] = useState(IMAGE_TASKS[0])
  const [paramVals, setParamVals] = useState<Record<string, string|number>>(() => { const d:Record<string,string|number>={}; selected.params.forEach((p: any)=>{if(p.d!==undefined)d[p.n]=p.d}); return d })
  const [generating, setGenerating] = useState(false)
  

  const handleTaskChange = (task: any) => {
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
        : { service: 'wan2gp', model: selected.model, ...paramVals }
      const res = await fetch('/v1/run', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) })
      const text = await res.text()
      let result: Record<string,unknown> = {}
      try { result = JSON.parse(text) } catch { toast('error', text.slice(0,200)); setGenerating(false); return }
      if (result.status === 'ok' || result.status === 'success') {
        if (result.data) {
          const mt = result.media_type as string || 'image/png'
          const isAud = mt.includes('audio')
          const dataUrl = `data:${mt};base64,${result.data}`
          const cat: AssetCategory = selected.id.includes('ace') ? 'music' : selected.id.includes('sfx') ? 'sfx' : selected.id.includes('voice') ? 'voice' : 'image'
          addAsset({ name: `${selected.label} ${new Date().toLocaleTimeString()}`, type: isAud?'audio':'image', category: cat, mediaType: mt, url: dataUrl, sizeBytes: Math.round((result.data as string).length*0.75), source: 'generated' })
          toast('success', selected.label)
        }
      } else {
        toast('error', String(result.error || result.message || 'Unknown error'))
      }
    } catch (e) { toast('error', e instanceof Error ? e.message : String(e)) }
    finally { setGenerating(false) }
  }

  return (
    <div className="flex-1 flex gap-6 p-6 overflow-hidden">
      <div className="w-56 shrink-0">
        <div className="text-sm font-semibold mb-3">Models</div>
        <Tabs defaultValue="image" className="w-full">
          <TabsList className="w-full">
            <TabsTrigger value="image" className="flex-1">Image</TabsTrigger>
            <TabsTrigger value="audio" className="flex-1">Audio</TabsTrigger>
          </TabsList>
          <TabsContent value="image" className="mt-2">
            <div className="flex flex-col gap-1">
              {IMAGE_TASKS.map((t: any) => { const Icon = t.icon; return (
                <Button key={t.id} variant={selected.id===t.id ? 'secondary' : 'ghost'} className="justify-start gap-2 h-auto py-2" onClick={()=>handleTaskChange(t)}>
                  <Icon className="h-4 w-4" />{t.label}
                </Button>
              )})}
            </div>
          </TabsContent>
          <TabsContent value="audio" className="mt-2">
            <div className="flex flex-col gap-1">
              {AUDIO_TASKS.map((t: any) => { const Icon = t.icon; return (
                <Button key={t.id} variant={selected.id===t.id ? 'secondary' : 'ghost'} className="justify-start gap-2 h-auto py-2" onClick={()=>handleTaskChange(t)}>
                  <Icon className="h-4 w-4" />{t.label}
                </Button>
              )})}
            </div>
          </TabsContent>
        </Tabs>
      </div>

      <Separator orientation="vertical" />

      <Card className="flex-1 max-w-lg">
        <CardHeader><CardTitle className="text-base">{selected.label}</CardTitle></CardHeader>
        <CardContent className="flex flex-col gap-4">
          {selected.params.map((p: any) => (
            <div key={p.n} className="flex flex-col gap-1.5">
              <Label htmlFor={p.n}>{p.l}</Label>
              {p.t === 'select' && p.opts ? (
                <Select value={String(paramVals[p.n]??p.d??'')} onValueChange={(v)=>setParamVals(prev=>({...prev,[p.n]:v}))}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>{p.opts.map((o: string)=><SelectItem key={o} value={o}>{o}</SelectItem>)}</SelectContent>
                </Select>
              ) : p.t === 'number' ? (
                <Input id={p.n} type="number" value={paramVals[p.n]??p.d??''} onChange={e=>setParamVals(prev=>({...prev,[p.n]:parseInt(e.target.value)||0}))} placeholder={p.p} />
              ) : p.t === 'file' ? (
                <Label htmlFor={p.n} className="flex items-center justify-center h-20 border-2 border-dashed rounded-lg cursor-pointer hover:border-primary/50 transition-colors text-sm text-muted-foreground">
                  {paramVals[p.n] ? 'File loaded' : p.p}
                  <Input id={p.n} type="file" className="hidden" accept="image/*" onChange={e=>{const f=e.target.files?.[0];if(!f)return;const r=new FileReader();r.onload=()=>{const d=r.result as string;setParamVals(prev=>({...prev,[p.n]:d.includes(',')?d.split(',')[1]:d}))};r.readAsDataURL(f)}} />
                </Label>
              ) : p.t === 'text' ? (
                <Textarea id={p.n} value={String(paramVals[p.n]??'')} onChange={e=>setParamVals(prev=>({...prev,[p.n]:e.target.value}))} placeholder={p.p} rows={3} />
              ) : (
                <Input id={p.n} value={String(paramVals[p.n]??p.d??'')} onChange={e=>setParamVals(prev=>({...prev,[p.n]:e.target.value}))} placeholder={p.p} />
              )}
            </div>
          ))}
          <Button className="mt-2" disabled={generating} onClick={handleGenerate}>
            {generating ? 'Generating...' : `Generate ${selected.label}`}
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}

// ─── Video Tab ────────────────────────────────────────────────────────────
function VideoTab() {
  const segments = useTimelineStore((s) => s.segments)
  const audioCues = useTimelineStore((s) => s.audioCues)
  const addSegment = useTimelineStore((s) => s.addSegment)
  const updateSegment = useTimelineStore((s) => s.updateSegment)
  const selectedSegmentId = useTimelineStore((s) => s.selectedSegmentId)
  const setSelectedSegment = useTimelineStore((s) => s.setSelectedSegment)
  const toast = useToastStore((s) => s.addToast)
  const [generating, setGenerating] = useState(false)
  const selectedSegment = segments.find((s) => s.id === selectedSegmentId)

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    try { const d = JSON.parse(e.dataTransfer.getData('application/tech-noir-asset'))
      if (d.type === 'image') { const s = addSegment({ prompt: d.name, firstFrameB64: d.url, thumbnailUrl: d.url, status:'empty' }); setSelectedSegment(s.id); toast('info', `Keyframe: ${d.name}`) }
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
        toast('success', `Video generated: K_${String(seg.order+1).padStart(2,'0')}`)
      } else {
        toast('error', String(result.error || 'Unknown error'))
      }
    } catch (e) { toast('error', e instanceof Error ? e.message : String(e)) }
    finally { setGenerating(false) }
  }

  return (
    <div className="flex-1 flex gap-6 p-6 overflow-hidden">
      <div className="flex-1 flex flex-col gap-4">
        <Card className="flex-1 flex items-center justify-center" onDragOver={e=>e.preventDefault()} onDrop={onDrop}>
          <CardContent className="flex items-center justify-center h-full">
            {selectedSegment?.videoUrl ? (
              <video src={selectedSegment.videoUrl} controls className="max-w-full max-h-full rounded-lg" />
            ) : segments.length === 0 ? (
              <p className="text-muted-foreground text-sm">Drag images from the Assets sidebar to add keyframes</p>
            ) : (
              <p className="text-muted-foreground text-sm">{segments.length} keyframe(s) — select one to generate video</p>
            )}
          </CardContent>
        </Card>
        <div className="h-24 flex items-center gap-1 p-2 border rounded-lg overflow-x-auto bg-muted/30">
          {segments.map((seg) => (
            <div key={seg.id} className={`h-full flex items-center justify-center rounded-md text-[10px] cursor-pointer relative overflow-hidden shrink-0 border-2 transition-colors ${seg.id===selectedSegmentId?'border-primary':'border-border hover:border-primary/50'}`}
              style={{width:`${seg.duration*40}px`}} onClick={()=>setSelectedSegment(seg.id)}>
              {seg.thumbnailUrl && <img src={seg.thumbnailUrl} alt="" className="absolute inset-0 w-full h-full object-cover opacity-30" />}
              <span className="relative z-10 font-medium">K_{String(seg.order+1).padStart(2,'0')}</span>
            </div>
          ))}
          <Button variant="outline" size="icon" className="h-full w-8 shrink-0" onClick={()=>{const s=addSegment({duration:5,status:'empty'});setSelectedSegment(s.id)}}>+</Button>
        </div>
        {selectedSegment && (
          <Card>
            <CardContent className="flex flex-col gap-3 pt-4">
              <div className="flex flex-col gap-1.5">
                <Label>Prompt</Label>
                <Textarea value={selectedSegment.prompt} onChange={(e) => updateSegment(selectedSegment.id, { prompt: e.target.value })} placeholder="Describe the motion..." rows={2} />
              </div>
              <div className="flex gap-3">
                <div className="flex flex-col gap-1.5">
                  <Label>Duration (s)</Label>
                  <Input type="number" className="w-24" value={selectedSegment.duration} onChange={(e) => updateSegment(selectedSegment.id, { duration: parseFloat(e.target.value) || 5 })} />
                </div>
                <Button className="flex-1 self-end" disabled={generating || !selectedSegment.firstFrameB64} onClick={handleGenerateVideo}>
                  {generating ? 'Generating...' : 'Generate LTX Video'}
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
        {audioCues.length > 0 && <p className="text-xs text-muted-foreground">{audioCues.length} audio cue(s) ready</p>}
      </div>
    </div>
  )
}

// ─── Root Layout ──────────────────────────────────────────────────────────
export function WorkspaceLayout(_props: any = {}) {
  return (
    <SidebarProvider defaultOpen={true}>
      <div className="flex h-screen w-full">
        <Sidebar><AppSidebar /></Sidebar>
        <div className="flex-1 flex flex-col min-w-0">
          <header className="flex items-center h-11 px-4 border-b shrink-0 gap-6">
            <span className="font-bold text-sm tracking-tight">TECH NOIR</span>
            <Tabs defaultValue="assets" className="h-full -mb-px">
              <TabsList className="h-full bg-transparent gap-0">
                <TabsTrigger value="assets" className="h-full rounded-none data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none">Assets</TabsTrigger>
                <TabsTrigger value="video" className="h-full rounded-none data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none">Video</TabsTrigger>
              </TabsList>
            </Tabs>
          </header>
          <Tabs defaultValue="assets" className="flex-1 flex flex-col min-h-0">
            <TabsContent value="assets" className="flex-1 overflow-hidden m-0 data-[state=active]:flex"><AssetsTab /></TabsContent>
            <TabsContent value="video" className="flex-1 overflow-hidden m-0 data-[state=active]:flex"><VideoTab /></TabsContent>
          </Tabs>
        </div>
      </div>
    </SidebarProvider>
  )
}
