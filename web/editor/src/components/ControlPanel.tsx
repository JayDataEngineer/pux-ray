import { useState, useCallback } from 'react'
import type { WorkflowSpec, WorkflowRun } from '../types'
import { rerunStep, executeStep, getRun } from '../api'
import { useWorkflowStore } from '../stores/workflow'

interface Props {
  spec: WorkflowSpec
  run: WorkflowRun | null
  onStart: (inputs: Record<string, unknown>) => void
}

export function ControlPanel({ spec, run, onStart }: Props) {
  const [inputs, setInputs] = useState<Record<string, unknown>>({})
  const [actionLoading, setActionLoading] = useState(false)
  const selectedStepId = useWorkflowStore((s) => s.selectedStepId)
  const setRun = useWorkflowStore((s) => s.setRun)

  const handleInputChange = useCallback((key: string, value: unknown) => {
    setInputs((prev) => ({ ...prev, [key]: value }))
  }, [])

  const handleStart = useCallback(() => {
    const merged: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(spec.inputs)) {
      if (k in inputs && inputs[k] !== '') {
        merged[k] = inputs[k]
      } else if (v.default !== undefined && v.default !== null) {
        merged[k] = v.default
      }
    }
    onStart(merged)
  }, [spec.inputs, inputs, onStart])

  const handleRerun = useCallback(async () => {
    if (!run || !selectedStepId) return
    setActionLoading(true)
    try {
      await rerunStep(run.spec_name, run.run_id, selectedStepId)
    } catch (e) {
      console.error('Rerun failed:', e)
    } finally {
      setActionLoading(false)
    }
  }, [run, selectedStepId])

  const handleExecute = useCallback(async () => {
    if (!run || !selectedStepId) return
    setActionLoading(true)
    try {
      const result = await executeStep(run.spec_name, run.run_id, selectedStepId) as Record<string, unknown>
      if (result.status !== 'error') {
        const updated = await getRun(run.spec_name, run.run_id)
        setRun(updated)
      }
    } catch (e) {
      console.error('Execute failed:', e)
    } finally {
      setActionLoading(false)
    }
  }, [run, selectedStepId, setRun])

  const selectedStep = run && selectedStepId
    ? { id: selectedStepId, state: run.step_states[selectedStepId] }
    : null

  return (
    <div className="control-panel">
      <div className="panel-header">
        {run ? 'Step Actions' : 'Configure'}
      </div>

      {!run ? (
        <div className="control-form">
          {Object.entries(spec.inputs).map(([key, inputSpec]) => (
            <div key={key} className="form-group">
              <label className="form-label">
                {key}
                {inputSpec.required && <span className="form-required">*</span>}
                {inputSpec.enum && (
                  <span className="form-hint">[{inputSpec.enum.join(', ')}]</span>
                )}
              </label>
              {inputSpec.enum ? (
                <select
                  className="form-input"
                  value={(inputs[key] as string) ?? (inputSpec.default as string) ?? ''}
                  onChange={(e) => handleInputChange(key, e.target.value)}
                >
                  {inputSpec.enum.map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              ) : inputSpec.type === 'integer' || inputSpec.type === 'number' ? (
                <input
                  className="form-input"
                  type="number"
                  value={(inputs[key] as string | number) ?? inputSpec.default ?? ''}
                  onChange={(e) => handleInputChange(key, inputSpec.type === 'integer' ? parseInt(e.target.value) : parseFloat(e.target.value))}
                  placeholder={inputSpec.description}
                />
              ) : (
                <textarea
                  className="form-input form-textarea"
                  value={(inputs[key] as string) ?? (inputSpec.default as string) ?? ''}
                  onChange={(e) => handleInputChange(key, e.target.value)}
                  placeholder={inputSpec.description}
                  rows={key.includes('prompt') ? 3 : 1}
                />
              )}
            </div>
          ))}
          <button className="btn btn-primary btn-block" onClick={handleStart}>
            Start Pipeline
          </button>
        </div>
      ) : selectedStep && selectedStep.state ? (
        <div className="control-actions">
          <div className="selected-step-info">
            <strong>{selectedStep.id}</strong>
            <span className={`run-status run-status--${selectedStep.state.status}`}>
              {selectedStep.state.status}
            </span>
          </div>

          {selectedStep.state.duration_ms && (
            <div className="step-detail">
              Duration: {(selectedStep.state.duration_ms / 1000).toFixed(1)}s
            </div>
          )}

          {selectedStep.state.error && (
            <div className="step-detail step-detail-error">
              {selectedStep.state.error}
            </div>
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
