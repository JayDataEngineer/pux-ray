import { useState, useCallback } from 'react'
import { Image, Wand2 } from 'lucide-react'
import { executeStep, getRun } from '../../api'
import { useWorkflowStore } from '../../stores/workflow'
import { useAssetStore } from '../../stores/assets'
import { useToastStore } from '../../stores/toast'
import type { WorkflowSpec, WorkflowRun } from '../../types'

interface Props {
  spec: WorkflowSpec
  run: WorkflowRun | null
  allSpecs: { name: string; description: string; steps: number }[]
  onSpecChange: (name: string) => void
}

type PipelineStep = 'character' | 'mesh' | 'compose'
type CharMode = 'generate' | 'existing'

export function VisualWorkspace({ spec: _spec, run, allSpecs: _allSpecs, onSpecChange: _onSpecChange }: Props) {
  const setRun = useWorkflowStore((s) => s.setRun)
  const toast = useToastStore((s) => s.addToast)
  const allAssets = useAssetStore((s) => s.assets)
  const [activeStep, setActiveStep] = useState<PipelineStep>('character')
  const [charMode, setCharMode] = useState<CharMode>('generate')
  const [charPrompt, setCharPrompt] = useState('')
  const [charModel, setCharModel] = useState('z_image')
  const [scenePrompt, setScenePrompt] = useState('')
  const [refineStrength, setRefineStrength] = useState(0.85)
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
    if (!run || !charPrompt) return
    setGenerating(true)
    try {
      const result = await executeStep(run.spec_name, run.run_id, 'generate_character', {
        input_prompt: charPrompt,
        model: charModel,
      }) as Record<string, unknown>
      if (result.status === 'error') {
        toast('error', String(result.error || 'Failed'))
        return
      }
      const updated = await getRun(run.spec_name, run.run_id)
      setRun(updated)
      const art = Object.values(updated.artifacts).find((a) => a.step_id === 'generate_character' && a.media_type.startsWith('image/'))
      if (art) {
        const url = `/v1/wf/${updated.spec_name}/runs/${updated.run_id}/artifacts/${art.step_id}/${art.name}.png`
        setCharImage(url)
      }
      setActiveStep('mesh')
      toast('success', 'Character generated')
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Generation failed')
    } finally {
      setGenerating(false)
    }
  }, [run, charPrompt, charModel, setRun, toast])

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
    if (!run || !charImage || !poseImage || !scenePrompt) return
    setGenerating(true)
    try {
      const result = await executeStep(run.spec_name, run.run_id, 'scene_compose', {
        input_prompt: scenePrompt,
        image_b64: charImage,
        reference_images: [poseImage],
      }) as Record<string, unknown>
      if (result.status === 'error') {
        toast('error', String(result.error || 'Failed'))
        return
      }
      const updated = await getRun(run.spec_name, run.run_id)
      setRun(updated)
      const art = Object.values(updated.artifacts).find((a) => a.step_id === 'scene_compose' && a.media_type.startsWith('image/'))
      if (art) {
        const url = `/v1/wf/${updated.spec_name}/runs/${updated.run_id}/artifacts/${art.step_id}/${art.name}.png`
        setComposedImage(url)
      }
      toast('success', 'Scene composed')
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Composition failed')
    } finally {
      setGenerating(false)
    }
  }, [run, charImage, poseImage, scenePrompt, setRun, toast])

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
            <button className="btn btn-primary" onClick={() => {
              if (run) {
                import('../../api').then(({ getRun }) => getRun(run.spec_name, run.run_id).then(setRun))
              }
            }}>
              Export to Video
            </button>
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
                <div className="asset-input">
                  {charImage && <img src={charImage} alt="Source" className="asset-thumb" />}
                  <span>{charImage ? 'CHAR_01' : '—'}</span>
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Pose Reference</label>
                <div className="asset-input">
                  {poseImage && <img src={poseImage} alt="Pose" className="asset-thumb" />}
                  <span>{poseImage ? 'Pose_Uploaded' : '—'}</span>
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Scene Description</label>
                <textarea className="form-input form-textarea" value={scenePrompt} onChange={(e) => setScenePrompt(e.target.value)} rows={3} placeholder="Describe the scene/background..." />
              </div>
              <div className="form-group">
                <label className="form-label">Refinement Strength <span className="param-value">{refineStrength.toFixed(2)}</span></label>
                <input type="range" min={0} max={1} step={0.01} value={refineStrength} onChange={(e) => setRefineStrength(parseFloat(e.target.value))} />
              </div>
              <button className="btn btn-primary btn-block" disabled={generating || !scenePrompt} onClick={handleCompose}>
                {generating ? 'COMPOSING...' : 'UPDATE PREVIEW'}
              </button>
            </>
          )}
        </div>
      </aside>
    </div>
  )
}
