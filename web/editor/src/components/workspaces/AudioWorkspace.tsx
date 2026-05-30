import { useState, useCallback, useRef } from 'react'
import { Music, Volume2, Mic, Wand2, Play, Trash2, Save } from 'lucide-react'
import { callTool } from '../../mcp'
import { useTimelineStore } from '../../stores/timeline'
import { useVoiceStore } from '../../stores/voices'
import { useAssetStore } from '../../stores/assets'
import { useToastStore } from '../../stores/toast'

// Audio task definitions — each is either a DAG pipeline or a direct service call.
// Parameters are discovered dynamically from the API, these are just defaults.
interface AudioTask {
  id: string
  label: string
  icon: typeof Music
  pipeline?: string        // DAG pipeline ID (preferred)
  service?: string          // Direct service call (fallback)
  model?: string            // Model ID for direct call
  params: AudioParam[]      // Parameter definitions
}

interface AudioParam {
  name: string
  type: 'string' | 'integer' | 'number'
  label: string
  placeholder: string
  default?: string | number
}

const AUDIO_TASKS: AudioTask[] = [
  {
    id: 'ace_step', label: 'ACE-Step (Music)', icon: Music,
    service: 'wan2gp', model: 'tts/ace_step_v1_5',
    params: [
      { name: 'input_prompt', type: 'string', label: 'Music Prompt', placeholder: 'epic cinematic orchestral, 120bpm, dark synthwave...' },
      { name: 'duration_seconds', type: 'integer', label: 'Duration (s)', placeholder: '30', default: 30 },
    ],
  },
  {
    id: 'moss_sfx', label: 'MOSS Sound Effect', icon: Volume2,
    service: 'wan2gp', model: 'moss/moss-soundeffect',
    params: [
      { name: 'input_prompt', type: 'string', label: 'Sound Description', placeholder: 'rain and thunder, distant explosions...' },
      { name: 'duration_seconds', type: 'integer', label: 'Duration (s)', placeholder: '5', default: 5 },
    ],
  },
  {
    id: 'moss_voice_clone', label: 'MOSS Voice Clone', icon: Mic,
    service: 'wan2gp', model: 'moss/moss-tts',
    params: [
      { name: 'text', type: 'string', label: 'Text to Speak', placeholder: 'Hello, my name is...' },
      { name: 'reference_audio_b64', type: 'string', label: 'Reference Audio (base64)', placeholder: 'Upload a reference voice sample' },
    ],
  },
  {
    id: 'moss_voice_gen', label: 'MOSS Voice Generator', icon: Wand2,
    service: 'wan2gp', model: 'moss/moss-voicegenerator',
    params: [
      { name: 'text', type: 'string', label: 'Voice Description', placeholder: 'Deep male voice, British accent, authoritative...' },
      { name: 'input_prompt', type: 'string', label: 'Text to Speak', placeholder: 'The text to synthesize with the generated voice' },
    ],
  },
  {
    id: 'kokoro', label: 'Kokoro TTS', icon: Mic,
    service: 'wan2gp', model: 'kokoro',
    params: [
      { name: 'text', type: 'string', label: 'Text', placeholder: 'Hello world' },
      { name: 'voice', type: 'string', label: 'Voice', placeholder: 'af_bella', default: 'af_bella' },
      { name: 'speed', type: 'number', label: 'Speed', placeholder: '1.0', default: 1.0 },
    ],
  },
]

interface GeneratedClip {
  id: string
  taskId: string
  label: string
  duration: number
  audioUrl: string | null
  status: 'generating' | 'ready' | 'error'
  error?: string
}

export function AudioWorkspace({ run: _run }: { run: import('../../types').WorkflowRun | null }) {
  const [selectedTask, setSelectedTask] = useState(AUDIO_TASKS[0])
  const [paramValues, setParamValues] = useState<Record<string, string | number>>({})
  const [generating, setGenerating] = useState(false)
  const [clips, setClips] = useState<GeneratedClip[]>([])
  const [playingId, setPlayingId] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const addAudioCue = useTimelineStore((s) => s.addAudioCue)
  const savedVoices = useVoiceStore((s) => s.voices)
  const addVoice = useVoiceStore((s) => s.addVoice)
  const removeVoice = useVoiceStore((s) => s.removeVoice)
  const addAsset = useAssetStore((s) => s.addAsset)
  const toast = useToastStore((s) => s.addToast)

  const handleSaveVoice = (clip: GeneratedClip) => {
    if (!clip.audioUrl) return
    const name = prompt('Voice name:') || `Voice ${savedVoices.length + 1}`
    if (!name) return
    const b64 = clip.audioUrl.includes(',') ? clip.audioUrl.split(',')[1] : clip.audioUrl
    const dataUrl = `data:audio/wav;base64,${b64}`
    addVoice({ name, audioB64: b64, modelId: clip.taskId })
    addAsset({ name, type: 'audio' as const, category: 'voice' as const, mediaType: 'audio/wav', url: dataUrl, sizeBytes: Math.round(b64.length * 0.75), source: 'generated' as const })
    toast('success', `Voice "${name}" saved to Assets`)
  }

  const handleUseSavedVoice = (voiceId: string) => {
    const voice = savedVoices.find((v) => v.id === voiceId)
    if (voice) {
      setParamValues((prev) => ({ ...prev, reference_audio_b64: voice.audioB64 }))
      toast('info', `Using saved voice: ${voice.name}`)
    }
  }

  const handleGenerate = useCallback(async () => {
    setGenerating(true)
    const task = selectedTask
    const clip: GeneratedClip = {
      id: `clip_${Date.now()}`, taskId: task.id,
      label: task.label, duration: Number(paramValues.duration_seconds || paramValues.duration || 5) || 5,
      audioUrl: null, status: 'generating',
    }
    setClips((prev) => [clip, ...prev])

    try {
      const result = task.pipeline
        ? await callTool<Record<string, unknown>>('run', { pipeline: task.pipeline, params: paramValues })
        : await callTool<Record<string, unknown>>('run', { service: task.service!, params: { model: task.model!, ...paramValues } })

      if (result.status === 'ok' || result.status === 'success') {
        const data = result.data as string | undefined
        const audioUrl = data ? `data:audio/wav;base64,${data}` : null
        setClips((prev) => prev.map((c) => c.id === clip.id ? { ...c, audioUrl, status: 'ready' as const } : c))
        if (audioUrl) {
          const track = task.id === 'ace_step' ? 'music' : task.id.includes('sfx') ? 'sfx' : 'voice'
          addAudioCue({
            track, start: 0, duration: clip.duration,
            label: `${task.label.split('(')[0].trim()}: ${String(paramValues.input_prompt || paramValues.text || '').slice(0, 25)}`,
            audioUrl, volume: task.id === 'ace_step' ? 0.4 : 0.8,
            waveformPeaks: null, sourceStepId: null,
          })
        }
        toast('success', `${task.label.split('(')[0].trim()} generated`)
      } else {
        const err = String(result.error || result.message || 'Unknown error')
        setClips((prev) => prev.map((c) => c.id === clip.id ? { ...c, status: 'error' as const, error: err } : c))
        toast('error', `${task.label}: ${err}`)
      }
    } catch (e) {
      setClips((prev) => prev.map((c) => c.id === clip.id ? { ...c, status: 'error' as const, error: String(e) } : c))
      toast('error', `Failed: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setGenerating(false)
    }
  }, [selectedTask, paramValues, addAudioCue, toast])

  const handlePlay = (clip: GeneratedClip) => {
    if (!clip.audioUrl) return
    if (playingId === clip.id) { audioRef.current?.pause(); setPlayingId(null) }
    else { if (audioRef.current) { audioRef.current.src = clip.audioUrl; audioRef.current.play(); setPlayingId(clip.id) } }
  }

  const handleTaskChange = (task: AudioTask) => {
    setSelectedTask(task)
    // Reset params to defaults
    const defaults: Record<string, string | number> = {}
    task.params.forEach((p) => { if (p.default !== undefined) defaults[p.name] = p.default })
    setParamValues(defaults)
  }

  return (
    <div className="audio-workspace">
      <audio ref={audioRef} onEnded={() => setPlayingId(null)} style={{ display: 'none' }} />

      <main className="audio-main">
        <div className="audio-main-header">
          <h1 className="audio-title">AUDIO SUITE</h1>
        </div>

        {/* Model Selector */}
        <div className="audio-section">
          <label className="audio-section-label">MODEL</label>
          <div className="model-selector">
            {AUDIO_TASKS.map((task) => {
              const Icon = task.icon
              return (
                <label key={task.id} className={`model-option ${selectedTask.id === task.id ? 'model-option--active' : ''}`}>
                  <input type="radio" name="audioModel" checked={selectedTask.id === task.id} onChange={() => handleTaskChange(task)} />
                  <Icon size={16} />
                  <span className="model-option-label">{task.label}</span>
                </label>
              )
            })}
          </div>
        </div>

        {/* Saved Voices — shown for voice clone */}
        {selectedTask.id === 'moss_voice_clone' && savedVoices.length > 0 && (
          <div className="audio-section">
            <label className="audio-section-label">SAVED VOICES ({savedVoices.length})</label>
            <div className="model-selector">
              {savedVoices.map((v) => (
                <div key={v.id} className="sidebar-clip" style={{ cursor: 'pointer' }} onClick={() => handleUseSavedVoice(v.id)}>
                  <Mic size={14} className="sidebar-icon" />
                  <span className="sidebar-clip-label">{v.name}</span>
                  <span className="sidebar-clip-dur">{v.createdAt.slice(0, 10)}</span>
                  <button className="btn btn-ghost btn-sm" onClick={(e) => { e.stopPropagation(); removeVoice(v.id) }}>
                    <Trash2 size={10} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Dynamic Parameters */}
        <div className="audio-section">
          <label className="audio-section-label">PARAMETERS</label>
          {selectedTask.params.map((param) => (
            <div key={param.name} className="form-group">
              <label className="form-label">{param.label}</label>
              {param.type === 'integer' || param.type === 'number' ? (
                <input
                  className="form-input"
                  type="number"
                  value={paramValues[param.name] ?? param.default ?? ''}
                  onChange={(e) => setParamValues((prev) => ({ ...prev, [param.name]: param.type === 'integer' ? parseInt(e.target.value) || 0 : parseFloat(e.target.value) || 0 }))}
                  placeholder={param.placeholder}
                />
              ) : param.name.includes('b64') || param.name.includes('reference') ? (
                <label className="btn btn-ghost btn-block" style={{ cursor: 'pointer' }}>
                  {paramValues[param.name] ? 'Audio loaded' : param.placeholder}
                  <input type="file" accept="audio/*" style={{ display: 'none' }} onChange={(e) => {
                    const f = e.target.files?.[0]; if (!f) return
                    const r = new FileReader(); r.onload = () => setParamValues((prev) => ({ ...prev, [param.name]: (r.result as string).split(',')[1] || r.result as string }))
                    r.readAsDataURL(f)
                  }} />
                </label>
              ) : param.name === 'text' || param.name === 'input_prompt' ? (
                <textarea
                  className="form-input form-textarea"
                  value={paramValues[param.name] ?? ''}
                  onChange={(e) => setParamValues((prev) => ({ ...prev, [param.name]: e.target.value }))}
                  rows={3} placeholder={param.placeholder}
                />
              ) : (
                <input
                  className="form-input"
                  type="text"
                  value={paramValues[param.name] ?? param.default ?? ''}
                  onChange={(e) => setParamValues((prev) => ({ ...prev, [param.name]: e.target.value }))}
                  placeholder={param.placeholder}
                />
              )}
            </div>
          ))}
        </div>

        <button className="btn btn-primary btn-block" disabled={generating} onClick={handleGenerate}>
          {generating ? 'GENERATING...' : `Generate ${selectedTask.label.split('(')[0].trim()}`}
        </button>

        {/* Master Track Preview */}
        <div className="audio-section">
          <label className="audio-section-label">MASTER TRACK — {clips.filter((c) => c.status === 'ready').length} clips</label>
          <div className="master-track">
            <div className="master-waveform">
              {Array.from({ length: 60 }).map((_, i) => (
                <div key={i} className="waveform-bar" style={{ height: `${Math.random() * 50 + 20}%`, animationDelay: `${Math.random()}s` }} />
              ))}
            </div>
            <div className="master-controls">
              <button className="btn btn-ghost btn-sm" onClick={() => { const r = clips.find((c) => c.status === 'ready' && c.audioUrl); if (r) handlePlay(r) }}>
                {playingId ? '⏸' : <Play size={14} />} Play Latest
              </button>
              <span className="master-time">{clips.filter((c) => c.status === 'ready').length} ready</span>
            </div>
          </div>
        </div>

        {/* Generated Clips */}
        {clips.length > 0 && (
          <div className="audio-section">
            <label className="audio-section-label">GENERATED</label>
            {clips.map((clip) => (
              <div key={clip.id} className={`clip-item clip-item--${clip.status}`}>
                <button className="btn btn-ghost btn-sm" onClick={() => handlePlay(clip)}>
                  {playingId === clip.id ? '⏸' : <Play size={12} />}
                </button>
                <span className="clip-label">{clip.label.split('(')[0].trim()}</span>
                <span className="clip-dur">{clip.duration}s</span>
                {clip.status === 'generating' && <span className="clip-status">Generating...</span>}
                {clip.status === 'error' && <span className="clip-status clip-status--error">{clip.error}</span>}
                {clip.status === 'ready' && clip.taskId === 'moss_voice_gen' && (
                  <button className="btn btn-ghost btn-sm" onClick={() => handleSaveVoice(clip)} title="Save this voice for reuse">
                    <Save size={12} />
                  </button>
                )}
                <button className="btn btn-ghost btn-sm" onClick={() => setClips((prev) => prev.filter((c) => c.id !== clip.id))}>
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}
