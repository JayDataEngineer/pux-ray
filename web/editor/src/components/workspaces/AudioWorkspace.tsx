import { useState, useCallback, useRef } from 'react'
import { callTool } from '../../mcp'
import { useTimelineStore } from '../../stores/timeline'
import { useToastStore } from '../../stores/toast'

type ModelTarget = 'ace_step' | 'moss_sfx' | 'moss_voice'

interface GeneratedClip {
  id: string
  model: ModelTarget
  prompt: string
  duration: number
  audioUrl: string | null
  status: 'idle' | 'generating' | 'ready' | 'error'
  error?: string
}

export function AudioWorkspace({ run: _run }: { run: import('../../types').WorkflowRun | null }) {
  const [model, setModel] = useState<ModelTarget>('ace_step')
  const [prompt, setPrompt] = useState('')
  const [duration, setDuration] = useState(30)
  const [bpm, setBpm] = useState(128)
  const [generating, setGenerating] = useState(false)
  const [clips, setClips] = useState<GeneratedClip[]>([])
  const [playingId, setPlayingId] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const addAudioCue = useTimelineStore((s) => s.addAudioCue)
  const toast = useToastStore((s) => s.addToast)

  const modelConfig: Record<ModelTarget, { label: string; service: string; modelId: string; paramKey: string; track: string }> = {
    ace_step: { label: 'ACE-Step (Music)', service: 'wan2gp', modelId: 'tts/ace_step_v1_5', paramKey: 'input_prompt', track: 'music' },
    moss_sfx: { label: 'MOSS-SFX (Sound Effects)', service: 'wan2gp', modelId: 'moss/moss-soundeffect', paramKey: 'input_prompt', track: 'sfx' },
    moss_voice: { label: 'MOSS-Voice (Clone/Creation)', service: 'wan2gp', modelId: 'kokoro', paramKey: 'text', track: 'voice' },
  }

  const handleGenerate = useCallback(async () => {
    if (!prompt) return
    setGenerating(true)
    const cfg = modelConfig[model]
    const clip: GeneratedClip = { id: `clip_${Date.now()}`, model, prompt, duration, audioUrl: null, status: 'generating' }
    setClips((prev) => [clip, ...prev])

    try {
      const params: Record<string, unknown> = {
        [cfg.paramKey]: prompt,
        duration_seconds: duration,
      }
      if (model === 'moss_voice') {
        params.voice = 'af_bella'
        params.text = prompt
      }
      const result = await callTool<Record<string, unknown>>('run', {
        service: cfg.service,
        params: { model: cfg.modelId, ...params },
      })

      if (result.status === 'ok' || result.status === 'success') {
        const data = result.data as string | undefined
        const audioUrl = data ? `data:audio/wav;base64,${data}` : null
        setClips((prev) => prev.map((c) => c.id === clip.id ? { ...c, audioUrl, status: 'ready' as const } : c))
        if (audioUrl) {
          addAudioCue({
            track: cfg.track,
            start: 0,
            duration,
            label: `${cfg.label.split(' ')[0]}: ${prompt.slice(0, 30)}`,
            audioUrl,
            volume: model === 'ace_step' ? 0.4 : 0.8,
            waveformPeaks: null,
            sourceStepId: null,
          })
        }
        toast('success', `${cfg.label} generated`)
      } else {
        const err = String(result.error || 'Unknown error')
        setClips((prev) => prev.map((c) => c.id === clip.id ? { ...c, status: 'error', error: err } : c))
        toast('error', err)
      }
    } catch (e) {
      setClips((prev) => prev.map((c) => c.id === clip.id ? { ...c, status: 'error', error: String(e) } : c))
      toast('error', e instanceof Error ? e.message : 'Generation failed')
    } finally {
      setGenerating(false)
    }
  }, [prompt, model, duration, addAudioCue, toast])

  const handlePlay = (clip: GeneratedClip) => {
    if (!clip.audioUrl) return
    if (playingId === clip.id) {
      audioRef.current?.pause()
      setPlayingId(null)
    } else {
      if (audioRef.current) {
        audioRef.current.src = clip.audioUrl
        audioRef.current.play()
        setPlayingId(clip.id)
      }
    }
  }

  const cfg = modelConfig[model]

  return (
    <div className="audio-workspace">
      <audio ref={audioRef} onEnded={() => setPlayingId(null)} style={{ display: 'none' }} />
      {/* Asset Explorer Sidebar */}
      <aside className="workspace-sidebar">
        <div className="sidebar-section">
          <div className="sidebar-title">ASSET EXPLORER</div>
          <button className="btn btn-primary btn-block">NEW GEN</button>
        </div>
        <div className="sidebar-nav">
          <a className="sidebar-link"><span className="icon">folder_open</span>Assets</a>
          <a className="sidebar-link"><span className="icon">upload_file</span>Uploads</a>
          <a className="sidebar-link"><span className="icon">history</span>History</a>
          <a className="sidebar-link"><span className="icon">library_music</span>Library</a>
        </div>
        <div className="sidebar-section">
          <div className="sidebar-subtitle">CLIPS</div>
          {clips.filter((c) => c.status === 'ready').map((c) => (
            <div key={c.id} className={`sidebar-clip ${playingId === c.id ? 'sidebar-clip--active' : ''}`} onClick={() => handlePlay(c)}>
              <span className="sidebar-clip-label">{c.prompt.slice(0, 25)}</span>
              <span className="sidebar-clip-dur">{c.duration}s</span>
            </div>
          ))}
        </div>
      </aside>

      {/* Main Content */}
      <main className="audio-main">
        <div className="audio-main-header">
          <h1 className="audio-title">AUDIO SUITE</h1>
        </div>

        {/* Model Selector */}
        <div className="audio-section">
          <label className="audio-section-label">MODEL_TARGET</label>
          <div className="model-selector">
            {Object.entries(modelConfig).map(([key, cfg]) => (
              <label key={key} className={`model-option ${model === key ? 'model-option--active' : ''}`}>
                <input type="radio" name="model" checked={model === key} onChange={() => setModel(key as ModelTarget)} />
                <span className="model-option-label">{cfg.label}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Composition Prompt */}
        <div className="audio-section">
          <label className="audio-section-label">
            COMPOSITION_PROMPT
            <span className="audio-char-count">{prompt.length}/500</span>
          </label>
          <textarea
            className="audio-textarea"
            placeholder={model === 'ace_step' ? 'Describe the sonic architecture... e.g. 120bpm cyberpunk synthwave with heavy distorted bassline' :
                        model === 'moss_sfx' ? 'e.g. rain and thunder, distant explosions, enemy footsteps' :
                        'e.g. Hello world, my name is...'}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value.slice(0, 500))}
            rows={6}
          />
        </div>

        {/* Parameters */}
        <div className="audio-section">
          <label className="audio-section-label">PARAMETERS</label>
          <div className="audio-params">
            <div className="param-row">
              <span className="param-label">Duration (s)</span>
              <span className="param-value">{duration}</span>
              <input type="range" min={1} max={120} value={duration} onChange={(e) => setDuration(parseInt(e.target.value))} />
            </div>
            {model === 'ace_step' && (
              <div className="param-row">
                <span className="param-label">BPM</span>
                <span className="param-value">{bpm}</span>
                <input type="range" min={60} max={200} value={bpm} onChange={(e) => setBpm(parseInt(e.target.value))} />
              </div>
            )}
          </div>
        </div>

        {/* Generate */}
        <button className="btn btn-primary btn-block" disabled={generating || !prompt} onClick={handleGenerate}>
          {generating ? 'GENERATING...' : 'INITIALIZE RENDER'}
        </button>

        {/* Master Track Preview */}
        {clips.length > 0 && (
          <div className="audio-section">
            <label className="audio-section-label">MASTER TRACK</label>
            <div className="master-track">
              <div className="master-track-header">
                <span>{cfg.label}</span>
                <span>{clips.filter((c) => c.status === 'ready').length} clips</span>
              </div>
              <div className="master-waveform">
                {Array.from({ length: 80 }).map((_, i) => (
                  <div key={i} className="waveform-bar" style={{
                    height: `${Math.random() * 60 + 20}%`,
                    animationDelay: `${Math.random()}s`,
                  }} />
                ))}
              </div>
              <div className="master-controls">
                <button className="btn btn-ghost btn-sm" onClick={() => clips[0] && handlePlay(clips[0])}>
                  {playingId ? '⏸' : '▶'} Play
                </button>
                <span className="master-time">00:00:00</span>
              </div>
            </div>
          </div>
        )}

        {/* Clip List */}
        {clips.length > 0 && (
          <div className="audio-section">
            <label className="audio-section-label">GENERATED CLIPS</label>
            {clips.map((clip) => (
              <div key={clip.id} className={`clip-item clip-item--${clip.status}`}>
                <button className="btn btn-ghost btn-sm" onClick={() => handlePlay(clip)}>
                  {playingId === clip.id ? '⏸' : '▶'}
                </button>
                <span className="clip-label">{clip.prompt.slice(0, 40)}</span>
                <span className="clip-model">{modelConfig[clip.model].label}</span>
                <span className="clip-dur">{clip.duration}s</span>
                {clip.status === 'generating' && <span className="clip-status">Generating...</span>}
                {clip.status === 'error' && <span className="clip-status clip-status--error">{clip.error}</span>}
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}
