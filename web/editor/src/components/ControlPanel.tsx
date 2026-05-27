import { useState, useCallback, useMemo, useRef } from 'react'
import type { WorkflowSpec, WorkflowRun, InputSpec } from '../types'
import { rerunStep, executeStep, getRun } from '../api'
import { useWorkflowStore } from '../stores/workflow'
import { useToastStore } from '../stores/toast'
import { getInputsForStep, buildInputStepMap } from '../utils/stepUtils'

interface Props {
  spec: WorkflowSpec
  run: WorkflowRun | null
  onStart: (inputs: Record<string, unknown>) => void
}

type InputEntry = [string, InputSpec]
interface InputGroup { label: string; inputs: InputEntry[] }

/** Whether a text prompt input should also offer an image upload option */
function isVisualPrompt(key: string): boolean {
  const k = key.toLowerCase()
  return k.includes('character') || k.includes('scene') || k.includes('style') || k.includes('reference')
}

/** File upload for a text prompt input — attaches an image alongside the text */
function ImageAttach({ inputKey, value, onChange }: {
  inputKey: string
  value: unknown
  onChange: (key: string, value: unknown) => void
}) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [preview, setPreview] = useState<string | null>(value && typeof value === 'string' && value.startsWith('data:') ? value : null)

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const base64 = reader.result as string
      onChange(inputKey, base64)
      setPreview(base64)
    }
    reader.readAsDataURL(file)
  }

  const clear = () => {
    onChange(inputKey, '')
    setPreview(null)
  }

  if (preview) {
    return (
      <div className="image-attach-preview">
        <img src={preview} alt="Attached" className="image-attach-thumb" />
        <button className="btn btn-ghost btn-sm" onClick={clear}>Remove</button>
      </div>
    )
  }

  return (
    <button className="btn btn-ghost btn-sm image-attach-btn" onClick={() => fileRef.current?.click()}>
      <input ref={fileRef} type="file" style={{ display: 'none' }} accept="image/*" onChange={handleFile} />
      + Attach image
    </button>
  )
}
function isFileInput(key: string, spec: InputSpec): boolean {
  const k = key.toLowerCase()
  if (k.includes('image') || k.includes('file') || k.includes('reference') || k.includes('audio') || k.includes('video') || k.includes('upload') || k.includes('attachment')) return true
  if (k.includes('frame') && k.includes('image')) return true
  const d = (spec.description || '').toLowerCase()
  if (d.includes('base64') || d.includes('upload') || d.includes('image file') || d.includes('audio file')) return true
  return false
}

function InputField({ inputKey, spec, value, onChange }: {
  inputKey: string
  spec: InputSpec
  value: unknown
  onChange: (key: string, value: unknown) => void
}) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [preview, setPreview] = useState<string | null>(null)

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const base64 = reader.result as string
      onChange(inputKey, base64)
      setPreview(base64)
    }
    reader.readAsDataURL(file)
  }

  if (isFileInput(inputKey, spec)) {
    return (
      <div className="upload-zone" onClick={() => fileRef.current?.click()}>
        <input ref={fileRef} type="file" style={{ display: 'none' }} accept="image/*,video/*" onChange={handleFile} />
        {preview ? (
          <img src={preview} alt="Preview" className="upload-preview" />
        ) : (
          <div className="upload-placeholder">
            <span>+</span>
            <span>{spec.description || 'Drop or click to upload'}</span>
          </div>
        )}
      </div>
    )
  }

  if (spec.enum) {
    return (
      <select
        className="form-input"
        value={(value as string) ?? (spec.default as string) ?? ''}
        onChange={(e) => onChange(inputKey, e.target.value)}
      >
        {spec.enum.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
      </select>
    )
  }

  if (spec.type === 'integer' || spec.type === 'number') {
    return (
      <input
        className="form-input"
        type="number"
        value={(value as string | number) ?? spec.default ?? ''}
        onChange={(e) => onChange(inputKey, spec.type === 'integer' ? parseInt(e.target.value) : parseFloat(e.target.value))}
        placeholder={spec.description}
      />
    )
  }

  return (
    <div className="form-field-group">
      <textarea
        className="form-input form-textarea"
        value={(value as string) ?? (spec.default as string) ?? ''}
        onChange={(e) => onChange(inputKey, e.target.value)}
        placeholder={spec.description}
        rows={inputKey.includes('prompt') ? 3 : 1}
      />
      {isVisualPrompt(inputKey) && (
        <ImageAttach inputKey={inputKey} value={value} onChange={onChange} />
      )}
    </div>
  )
}

export function ControlPanel({ spec, run, onStart }: Props) {
  const [inputs, setInputs] = useState<Record<string, unknown>>({})
  const [actionLoading, setActionLoading] = useState(false)
  const [editParams, setEditParams] = useState<Record<string, unknown>>({})
  const [showParamEditor, setShowParamEditor] = useState(false)
  const selectedStepId = useWorkflowStore((s) => s.selectedStepId)
  const setRun = useWorkflowStore((s) => s.setRun)
  const toast = useToastStore((s) => s.addToast)

  const inputStepMap = useMemo(
    () => buildInputStepMap(spec.inputs, spec.steps),
    [spec.inputs, spec.steps],
  )

  const relevantInputs = useMemo(() => {
    if (!selectedStepId || run) return null
    const step = spec.steps.find((s) => s.id === selectedStepId)
    if (!step) return null
    const stepInputs = getInputsForStep(step)
    const shared = [...inputStepMap.entries()]
      .filter(([, steps]) => steps.length === 0 || steps.length > 1)
      .map(([name]) => name)
    return new Set([...stepInputs, ...shared])
  }, [selectedStepId, run, spec.steps, inputStepMap])

  const inputGroups = useMemo((): InputGroup[] | null => {
    if (run) return null

    if (relevantInputs) {
      const entries = Object.entries(spec.inputs).filter(([k]) => relevantInputs.has(k))
      return [{ label: selectedStepId!.replace(/_/g, ' '), inputs: entries }]
    }

    const groups = new Map<string, InputEntry[]>()
    const sharedEntries: InputEntry[] = []

    for (const [name, inputSpec] of Object.entries(spec.inputs)) {
      const steps = inputStepMap.get(name) || []
      if (steps.length === 0 || steps.length > 1) {
        sharedEntries.push([name, inputSpec])
      } else {
        const stepId = steps[0]
        if (!groups.has(stepId)) groups.set(stepId, [])
        groups.get(stepId)!.push([name, inputSpec])
      }
    }

    const result: InputGroup[] = []
    for (const step of spec.steps) {
      const entries = groups.get(step.id)
      if (entries) result.push({ label: step.id.replace(/_/g, ' '), inputs: entries })
    }
    if (sharedEntries.length > 0) result.push({ label: 'Shared', inputs: sharedEntries })

    return result
  }, [run, relevantInputs, selectedStepId, spec.inputs, spec.steps, inputStepMap])

  const handleChange = useCallback((key: string, value: unknown) => {
    setInputs((prev) => ({ ...prev, [key]: value }))
  }, [])

  const handleStart = useCallback(() => {
    const merged: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(spec.inputs)) {
      if (k in inputs && inputs[k] !== '') {
        merged[k] = inputs[k]
      } else if (v.default !== undefined && v.default !== null) {
        merged[k] = v.default
      } else if (v.required) {
        toast('error', `Missing required field: ${k.replace(/_/g, ' ')}`)
        return
      }
    }
    onStart(merged)
  }, [spec.inputs, inputs, onStart, toast])

  const handleRerun = useCallback(async () => {
    if (!run || !selectedStepId) return
    setActionLoading(true)
    try {
      const params = showParamEditor && Object.keys(editParams).length > 0 ? editParams : undefined
      await rerunStep(run.spec_name, run.run_id, selectedStepId, params)
      setShowParamEditor(false)
      setEditParams({})
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Retry failed')
    } finally {
      setActionLoading(false)
    }
  }, [run, selectedStepId, showParamEditor, editParams, toast])

  const handleExecute = useCallback(async () => {
    if (!run || !selectedStepId) return
    setActionLoading(true)
    try {
      const params = showParamEditor && Object.keys(editParams).length > 0 ? editParams : undefined
      const result = await executeStep(run.spec_name, run.run_id, selectedStepId, params) as Record<string, unknown>
      if (result.status !== 'error') {
        const updated = await getRun(run.spec_name, run.run_id)
        setRun(updated)
      }
      setShowParamEditor(false)
      setEditParams({})
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Execute failed')
    } finally {
      setActionLoading(false)
    }
  }, [run, selectedStepId, setRun, showParamEditor, editParams, toast])

  const selectedStep = run && selectedStepId
    ? { id: selectedStepId, state: run.step_states[selectedStepId] }
    : null

  const stepSpec = spec.steps.find((s) => s.id === selectedStepId)

  const headerLabel = run
    ? 'Step Actions'
    : relevantInputs
      ? `${selectedStepId!.replace(/_/g, ' ')} Config`
      : 'Configure'

  return (
    <div className="control-panel">
      <div className="panel-header">
        {headerLabel}
        {relevantInputs && (
          <button className="btn btn-ghost btn-sm" onClick={() => useWorkflowStore.getState().setSelectedStep(null)}>
            Show all
          </button>
        )}
      </div>

      {!run ? (
        <div className="control-form">
          {inputGroups?.map((group) => (
            <div key={group.label} className="config-group">
              {(inputGroups.length > 1 || relevantInputs) && (
                <div className="config-group-label">{group.label}</div>
              )}
              {group.inputs.map(([key, inputSpec]) => (
                <div key={key} className="form-group">
                  <label className="form-label">
                    {key.replace(/_/g, ' ')}
                    {inputSpec.required && <span className="form-required">*</span>}
                    {inputSpec.enum && (
                      <span className="form-hint">[{inputSpec.enum.join(', ')}]</span>
                    )}
                  </label>
                  <InputField
                    inputKey={key}
                    spec={inputSpec}
                    value={inputs[key]}
                    onChange={handleChange}
                  />
                </div>
              ))}
            </div>
          ))}
          <button className="btn btn-primary btn-block" onClick={handleStart}>
            Start Pipeline (Manual)
          </button>
        </div>
      ) : selectedStep && selectedStep.state ? (
        <div className="control-actions">
          <div className="selected-step-info">
            <strong>{selectedStep.id.replace(/_/g, ' ')}</strong>
            <span className={`run-status run-status--${selectedStep.state.status}`}>
              {selectedStep.state.status}
            </span>
          </div>

          {selectedStep.state.duration_ms != null && (
            <div className="step-detail">
              Duration: {(selectedStep.state.duration_ms / 1000).toFixed(1)}s
            </div>
          )}

          {selectedStep.state.error && (
            <div className="step-detail step-detail-error">
              {selectedStep.state.error}
            </div>
          )}

          {(selectedStep.state.status === 'completed' || selectedStep.state.status === 'failed') && stepSpec && (
            <>
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => setShowParamEditor(!showParamEditor)}
              >
                {showParamEditor ? 'Hide' : 'Edit'} params
              </button>

              {showParamEditor && (
                <div className="param-editor">
                  {Object.entries(stepSpec.params || {}).map(([key, defaultVal]) => (
                    <div key={key} className="form-group">
                      <label className="form-label">{key}</label>
                      <input
                        className="form-input"
                        type="text"
                        value={String(editParams[key] ?? (typeof defaultVal === 'string' ? defaultVal.replace(/\{\{.*?\}\}/g, '') : defaultVal ?? ''))}
                        onChange={(e) => setEditParams((prev) => ({ ...prev, [key]: e.target.value }))}
                      />
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          <div className="action-buttons">
            {selectedStep.state.status === 'completed' && (
              <button
                className="btn btn-secondary"
                onClick={handleExecute}
                disabled={actionLoading}
              >
                Regenerate
              </button>
            )}
            {selectedStep.state.status === 'failed' && (
              <button
                className="btn btn-primary"
                onClick={handleRerun}
                disabled={actionLoading}
              >
                Retry from here
              </button>
            )}
          </div>
        </div>
      ) : (
        <div className="control-empty">
          Select a step to see actions
        </div>
      )}
    </div>
  )
}
