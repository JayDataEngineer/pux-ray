import type { WorkflowSpec, WorkflowRun } from '../types'
import { cancelRun } from '../api'
import { useWorkflowStore, type ViewMode } from '../stores/workflow'
import { useToastStore } from '../stores/toast'

interface HeaderProps {
  spec: WorkflowSpec
  allSpecs: { name: string; description: string; steps: number }[]
  run: WorkflowRun | null
  onNewRun: () => void
  onSpecChange: (name: string) => void
}

export function Header({ spec, allSpecs, run, onNewRun, onSpecChange }: HeaderProps) {
  const setRun = useWorkflowStore((s) => s.setRun)
  const viewMode = useWorkflowStore((s) => s.viewMode)
  const setViewMode = useWorkflowStore((s) => s.setViewMode)
  const toast = useToastStore((s) => s.addToast)

  const handleCancel = async () => {
    if (!run) return
    try {
      await cancelRun(run.spec_name, run.run_id)
      setRun({ ...run, status: 'cancelled' })
      toast('info', 'Run cancelled')
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Could not cancel run')
    }
  }

  const toggleView = (mode: ViewMode) => {
    setViewMode(viewMode === mode ? 'pipeline' : mode)
  }

  return (
    <header className="editor-header">
      <div className="header-left">
        <span className="header-logo">Tech Noir</span>
        <span className="header-sep">/</span>
        <select
          className="spec-selector"
          value={spec.name}
          onChange={(e) => onSpecChange(e.target.value)}
        >
          {allSpecs.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name.replace(/_/g, ' ')} ({s.steps} steps)
            </option>
          ))}
        </select>
      </div>
      <div className="header-right">
        {run && (
          <>
            <span className={`run-status run-status--${run.status}`}>
              {run.status}
            </span>
            <span className="run-id">{run.run_id}</span>
            {(run.status === 'running') && (
              <button className="btn btn-ghost btn-sm" onClick={handleCancel}>Cancel</button>
            )}
            <button className="btn btn-ghost btn-sm" onClick={onNewRun}>New Run</button>
          </>
        )}
        <button
          className={`btn btn-ghost btn-sm ${viewMode === 'kimodo' ? 'btn-active' : ''}`}
          onClick={() => toggleView('kimodo')}
        >
          Kimodo
        </button>
        <a href="/studio/" target="_blank" rel="noreferrer" className="btn btn-ghost btn-sm">Studio</a>
        <a href="/dashboard" target="_blank" rel="noreferrer" className="btn btn-ghost btn-sm">Dashboard</a>
      </div>
    </header>
  )
}
