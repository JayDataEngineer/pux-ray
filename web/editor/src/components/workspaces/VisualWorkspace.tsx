import { useState, useCallback } from 'react'
import { Image, Wand2 } from 'lucide-react'
import { useAssetStore } from '../../stores/assets'
import { useToastStore } from '../../stores/toast'
import type { WorkflowSpec, WorkflowRun } from '../../types'

async function runPipeline(pipelineId: string, params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const res = await fetch('/v1/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pipeline: pipelineId, params }),
  })
  const text = await res.text()
  if (!res.ok) {
    // Try to extract error from JSON body, fallback to status text
    try { const err = JSON.parse(text); throw new Error(err.error || err.message || `HTTP ${res.status}`) }
    catch { if (text) throw new Error(text.slice(0, 200)); throw new Error(`Pipeline ${pipelineId} failed: HTTP ${res.status}`) }
  }
  try { return JSON.parse(text) }
  catch { throw new Error(`Invalid response from pipeline: ${text.slice(0, 200)}`) }
}

interface Props {
  spec: WorkflowSpec
  run: WorkflowRun | null
  allSpecs: { name: string; description: string; steps: number }[]
  onSpecChange: (name: string) => void
}

type PipelineStep = 'character' | 'mesh' | 'compose'
type CharMode = 'generate' | 'existing'

export function VisualWorkspace({ spec: _spec, run: _run, allSpecs: _allSpecs, onSpecChange: _onSpecChange }: Props) {
  const toast = useToastStore((s) => s.addToast)
  const allAssets = useAssetStore((s) => s.assets)
  const [activeStep, setActiveStep] = useState<PipelineStep>('character')
  const [charMode, setCharMode] = useState<CharMode>('generate')
  const [charPrompt, setCharPrompt] = useState('')
  const [charModel, setCharModel] = useState('z_image')
  const [generating, setGenerating] = useState(false)
  const [charImage, setCharImage] = useState<string | null>(null)
  const [poseImage, setPoseImage] = useState<string | null>(null)
  const [composedImage, setComposedImage] = useState<string | null>(null)

  const pipelineSteps: { id: PipelineStep; label: string; tag: string; num: number }[] = [
    { id: 'character', label: 'Character Gen', tag: 'Z-Image', num: 1 },
    { id: 'mesh', label: 'Mesh Posing', tag: 'Kimodo', num: 2 },
    { id: 'compose', label: 'Composition', tag: 'VNCCS', num: 3 },
  ]

  const handleGenerateChar = useCallback(async () => {
    if (!charPrompt) { toast('error', 'Enter a character prompt'); return }
    setGenerating(true)
    try {
      const quality = charModel === 'z_image_base' ? 'standard' : 'turbo'
      const result = await runPipeline('tech-noir/generate', {
        prompt: charPrompt,
        quality,
        seed: 42,
        width: 1024,
        height: 1024,
      })
      if (result.status === 'error' || result.status === 'failed') {
        toast('error', `Generate: ${String(result.error || result.message || 'Failed')}`)
        setGenerating(false)
        return
      }
      if (result.data) {
        const url = `data:image/png;base64,${result.data}`
        setCharImage(url)
        setActiveStep('mesh')
        toast('success', 'Character generated')
      } else {
        toast('error', `Generate returned no image: ${JSON.stringify(result).slice(0, 200)}`)
      }
    } catch (e) {
      toast('error', `Generate failed: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setGenerating(false)
    }
  }, [charPrompt, charModel, toast])

  const handleLaunchKimodo = useCallback(() => {
    window.open('/kimodo/', '_blank')
  }, [])

  const handleUploadPose = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => setPoseImage(reader.result as string)
    reader.readAsDataURL(file)
  }, [])

  const handleCompose = useCallback(async () => {
    if (!charImage) { toast('error', 'Select a character image first (Step 1 or Use Existing)'); return }
    if (!poseImage) { toast('error', 'Upload a pose image first (Kimodo or file)'); return }
    setGenerating(true)
    try {
      const strip = (url: string): string => {
        const idx = url.indexOf(',')
        return idx > 0 ? url.slice(idx + 1) : url
      }
      const result = await runPipeline('vnccs/pose-edit', {
        character_image_b64: strip(charImage),
        rotations: {},
        model_rotation_y: 0.0,
        seed: 42,
      })
      if (result.status === 'error' || result.status === 'failed') {
        toast('error', `VNCCS pose-edit: ${String(result.error || result.message || 'Failed')}`)
        setGenerating(false)
        return
      }
      if (result.data) {
        setComposedImage(`data:image/png;base64,${result.data}`)
        toast('success', 'VNCCS pose edit complete — character posed successfully')
      } else {
        toast('error', `VNCCS returned no image. Response: ${JSON.stringify(result).slice(0, 300)}`)
      }
    } catch (err) {
      toast('error', `VNCCS pipeline error: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setGenerating(false)
    }
  }, [charImage, poseImage, toast])

  const displayImage = activeStep === 'compose' && composedImage
    ? composedImage
    : activeStep === 'mesh' && poseImage
    ? poseImage
    : charImage

  return (
    <div className="visuals-workspace">
      {/* Keyframe Canvas */}
      <main className="visuals-main">
        <div className="visuals-canvas-header">
          <span className="canvas-title">KEYFRAME CANVAS</span>
          <span className="canvas-res">1920×1080</span>
        </div>
        <div className="visuals-canvas">
          {displayImage ? (
            <div className="canvas-frame">
              <img src={displayImage} alt="Keyframe" />
              <div className="canvas-grid" />
            </div>
          ) : (
            <div className="canvas-empty">
              <span>Generate a character to begin</span>
            </div>
          )}
        </div>
        <div className="visuals-canvas-footer">
          <span>RENDER: <span className="text-primary">{generating ? 'PROCESSING' : 'READY'}</span></span>
          {composedImage && (
            <span className="text-primary" style={{fontSize:11}}>Ready — switch to Video tab to use this keyframe</span>
          )}
        </div>
      </main>

      {/* Pipeline Panel */}
      <aside className="workspace-panel">
        <div className="panel-title">VISUAL PIPELINE</div>

        {/* Pipeline Steps — all clickable, no gates */}
        <div className="pipeline-steps-visual">
          {pipelineSteps.map((step) => (
            <div key={step.id} className={`pipeline-step ${activeStep === step.id ? 'pipeline-step--active' : ''}`}
              onClick={() => setActiveStep(step.id)}>
              <div className="pipeline-step-num">{step.num}</div>
              <div className="pipeline-step-label">
                {step.label}
                <span className="pipeline-step-tag">{step.tag}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Step Controls */}
        <div className="pipeline-controls">
          {activeStep === 'character' && (
            <>
              <div className="mode-toggle">
                <button className={`btn btn-sm ${charMode === 'generate' ? 'btn-primary' : 'btn-ghost'}`} onClick={() => setCharMode('generate')}>
                  <Wand2 size={14} /> Generate
                </button>
                <button className={`btn btn-sm ${charMode === 'existing' ? 'btn-primary' : 'btn-ghost'}`} onClick={() => setCharMode('existing')}>
                  <Image size={14} /> Use Existing
                </button>
              </div>

              {charMode === 'generate' ? (
                <>
                  <div className="form-group">
                    <label className="form-label">Character Prompt</label>
                    <textarea className="form-input form-textarea" value={charPrompt} onChange={(e) => setCharPrompt(e.target.value)} rows={4} placeholder="Describe the character..." />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Model</label>
                    <select className="form-input" value={charModel} onChange={(e) => setCharModel(e.target.value)}>
                      <option value="z_image">Z-Image (Turbo)</option>
                      <option value="z_image_base">Z-Image (Base)</option>
                    </select>
                  </div>
                  <button className="btn btn-primary btn-block" disabled={generating || !charPrompt} onClick={handleGenerateChar}>
                    {generating ? 'GENERATING...' : 'GENERATE CHARACTER'}
                  </button>
                </>
              ) : (
                <>
                  <div className="form-group">
                    <label className="form-label">Pick from Assets</label>
                    {(() => {
                      const imgs = allAssets.filter((a) => a.type === 'image')
                      if (imgs.length === 0) return <div className="sidebar-empty">No images in asset folder — generate or import first</div>
                      return (
                        <div className="asset-picker-grid">
                          {imgs.slice(0, 12).map((a) => (
                            <div key={a.id}
                              className={`asset-picker-thumb ${charImage === a.url ? 'asset-picker-thumb--active' : ''}`}
                              onClick={() => { setCharImage(a.url); setActiveStep('mesh'); toast('info', `Using "${a.name}" as character`) }}>
                              <img src={a.url} alt={a.name} />
                              <span>{a.name.slice(0, 14)}</span>
                            </div>
                          ))}
                        </div>
                      )})()}
                  </div>
                  <button className="btn btn-secondary btn-block" onClick={() => setCharMode('generate')}>
                    Or generate new
                  </button>
                </>
              )}
            </>
          )}

          {activeStep === 'mesh' && (
            <>
              <div className="form-group">
                <label className="form-label">Source Character</label>
                {!charImage && allAssets.filter(a => a.type === 'image').length > 0 && (
                  <div className="asset-picker-grid">
                    {allAssets.filter(a => a.type === 'image').slice(0, 6).map((a) => (
                      <div key={a.id}
                        className={`asset-picker-thumb ${charImage === a.url ? 'asset-picker-thumb--active' : ''}`}
                        onClick={() => setCharImage(a.url)}>
                        <img src={a.url} alt={a.name} />
                        <span>{a.name.slice(0, 12)}</span>
                      </div>
                    ))}
                  </div>
                )}
                <div className="asset-input" style={{ marginTop: 8 }}>
                  {charImage ? <img src={charImage} alt="Source" className="asset-thumb" /> : <span className="sidebar-empty">Pick from above or upload below</span>}
                  <span>{charImage ? 'Selected' : 'No character selected'}</span>
                </div>
                <label className="btn btn-ghost btn-sm" style={{ cursor: 'pointer', marginTop: 4 }}>
                  Upload Character
                  <input type="file" accept="image/*" style={{ display: 'none' }} onChange={(e) => {
                    const file = e.target.files?.[0]; if (!file) return
                    const reader = new FileReader()
                    reader.onload = () => setCharImage(reader.result as string)
                    reader.readAsDataURL(file)
                  }} />
                </label>
              </div>
              <div className="form-group">
                <label className="form-label">Pose Reference (Mesh)</label>
                <div className="asset-input">
                  {poseImage ? (
                    <img src={poseImage} alt="Pose" className="asset-thumb" />
                  ) : (
                    <span>No pose uploaded</span>
                  )}
                  <span>{poseImage ? 'Pose Ready' : 'Launch Kimodo or upload'}</span>
                </div>
              </div>
              <button className="btn btn-primary btn-block" onClick={handleLaunchKimodo}>
                Open Kimodo Director
              </button>
              <label className="btn btn-ghost btn-block" style={{ cursor: 'pointer' }}>
                Upload Pose Image
                <input type="file" accept="image/*" style={{ display: 'none' }} onChange={handleUploadPose} />
              </label>
              <button className="btn btn-secondary btn-block" onClick={() => setActiveStep('compose')}>
                Continue to Composition
              </button>
            </>
          )}

          {activeStep === 'compose' && (
            <>
              <div className="form-group">
                <label className="form-label">Source Character</label>
                {charImage ? (
                  <div className="asset-input">
                    <img src={charImage} alt="Source" className="asset-thumb" />
                    <span>Selected</span>
                    <button className="btn btn-ghost btn-sm" onClick={() => setCharImage(null)}>Clear</button>
                  </div>
                ) : (
                  <div className="asset-picker-grid">
                    {allAssets.filter(a => a.type === 'image').slice(0, 6).map((a) => (
                      <div key={a.id} className="asset-picker-thumb" onClick={() => setCharImage(a.url)}>
                        <img src={a.url} alt={a.name} />
                        <span>{a.name.slice(0, 12)}</span>
                      </div>
                    ))}
                  </div>
                )}
                <label className="btn btn-ghost btn-sm" style={{ cursor: 'pointer', marginTop: 4 }}>
                  Upload Character
                  <input type="file" accept="image/*" style={{ display: 'none' }} onChange={(e) => { const f=e.target.files?.[0]; if(!f)return; const r=new FileReader(); r.onload=()=>setCharImage(r.result as string); r.readAsDataURL(f) }} />
                </label>
              </div>
              <div className="form-group">
                <label className="form-label">Pose Reference (Kimodo output)</label>
                {poseImage ? (
                  <div className="asset-input">
                    <img src={poseImage} alt="Pose" className="asset-thumb" />
                    <span>Selected</span>
                    <button className="btn btn-ghost btn-sm" onClick={() => setPoseImage(null)}>Clear</button>
                  </div>
                ) : (
                  <div className="asset-picker-grid">
                    {allAssets.filter(a => a.type === 'image').slice(0, 6).map((a) => (
                      <div key={a.id} className="asset-picker-thumb" onClick={() => setPoseImage(a.url)}>
                        <img src={a.url} alt={a.name} />
                        <span>{a.name.slice(0, 12)}</span>
                      </div>
                    ))}
                  </div>
                )}
                <label className="btn btn-ghost btn-sm" style={{ cursor: 'pointer', marginTop: 4 }}>
                  Upload Pose
                  <input type="file" accept="image/*" style={{ display: 'none' }} onChange={(e) => { const f=e.target.files?.[0]; if(!f)return; const r=new FileReader(); r.onload=()=>setPoseImage(r.result as string); r.readAsDataURL(f) }} />
                </label>
              </div>
              <button className="btn btn-primary btn-block" disabled={generating || !charImage || !poseImage} onClick={handleCompose}>
                {generating ? 'COMPOSING...' : 'COMPOSE (VNCCS Pose Edit)'}
              </button>
            </>
          )}
        </div>
      </aside>
    </div>
  )
}
