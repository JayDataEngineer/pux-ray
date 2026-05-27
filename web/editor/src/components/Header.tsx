import type { WorkflowSpec, WorkflowRun } from '../types'

interface HeaderProps {
  spec: WorkflowSpec
  run: WorkflowRun | null
  onNewRun: () => void
}

export function Header({ spec, run, onNewRun }: HeaderProps) {
  return (
    <header className="editor-header">
      <div className="header-left">
        <span className="header-logo">Tech Noir</span>
        <span className="header-sep">/</span>
        <span className="header-title">{spec.description || spec.name}</span>
      </div>
      <div className="header-right">
        {run && (
          <>
            <span className={`run-status run-status--${run.status}`}>
              {run.status}
            </span>
            <span className="run-id">{run.run_id}</span>
            <button className="btn btn-ghost" onClick={onNewRun}>New Run</button>
          </>
        )}
        <a href="/studio" className="btn btn-ghost">Studio</a>
        <a href="/dashboard" className="btn btn-ghost">Dashboard</a>
      </div>
    </header>
  )
}
