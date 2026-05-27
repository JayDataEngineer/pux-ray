import { useMemo, useCallback, useRef } from 'react'
import { Timeline, type TimelineState } from '@xzdarcy/react-timeline-editor'
import type { TimelineRow, TimelineAction, TimelineEffect } from '@xzdarcy/timeline-engine'
import type { WorkflowSpec, WorkflowRun } from '../types'
import { useWorkflowStore } from '../stores/workflow'
import '@xzdarcy/react-timeline-editor/dist/react-timeline-editor.css'

const TRACK_DEFS: Array<{ id: string; label: string; color: string; stepIds: string[] }> = [
  { id: 'video', label: 'Video', color: '#818cf8', stepIds: ['generate_video', 'lipsync', 'video_edit', 'upscale'] },
  { id: 'voice', label: 'Voice', color: '#4ade80', stepIds: ['voice'] },
  { id: 'sfx', label: 'SFX', color: '#facc15', stepIds: ['sound_fx'] },
  { id: 'music', label: 'Music', color: '#fb923c', stepIds: ['music'] },
  { id: 'mix', label: 'Mix', color: '#f87171', stepIds: ['mix_audio'] },
]

function artifactUrl(run: WorkflowRun, stepId: string): string | null {
  const key = Object.keys(run.artifacts).find((k) => k.startsWith(`${stepId}.`))
  if (!key) return null
  const art = run.artifacts[key]
  const name = art.name.includes('.') ? art.name : art.name + '.bin'
  return `/v1/wf/${run.spec_name}/runs/${run.run_id}/artifacts/${art.step_id}/${name}`
}

interface Props {
  spec: WorkflowSpec
  run: WorkflowRun | null
}

export function TimelineBar({ spec, run }: Props) {
  const setSelectedStep = useWorkflowStore((s) => s.setSelectedStep)
  const timelineRef = useRef<TimelineState>(null)

  const totalDuration = useMemo(() => {
    if (!run) return 5
    const frames = typeof run.inputs?.video_frames === 'number' ? run.inputs.video_frames : 121
    const fps = typeof run.inputs?.video_fps === 'number' ? run.inputs.video_fps : 24
    return frames / fps
  }, [run])

  const hasOutput = run
    ? TRACK_DEFS.some((td) =>
        td.stepIds.some((sid) => {
          const st = run.step_states[sid]
          return st?.status === 'completed' && Object.keys(run.artifacts).some((k) => k.startsWith(`${sid}.`))
        }),
      )
    : false

  const { rows, effects } = useMemo(() => {
    if (!run || !hasOutput) return { rows: [] as TimelineRow[], effects: {} as Record<string, TimelineEffect> }

    const rows: TimelineRow[] = []
    const effects: Record<string, TimelineEffect> = {}

    for (const td of TRACK_DEFS) {
      const actions: TimelineAction[] = []
      for (const stepId of td.stepIds) {
        const state = run.step_states[stepId]
        const url = artifactUrl(run, stepId)
        if (!url && state?.status !== 'completed') continue

        const actionId = `${td.id}_${stepId}`
        effects[actionId] = { id: actionId, name: stepId.replace(/_/g, ' ') }
        actions.push({
          id: actionId,
          start: 0,
          end: totalDuration,
          effectId: actionId,
        })
      }
      if (actions.length > 0) {
        rows.push({ id: td.id, actions })
      }
    }

    return { rows, effects }
  }, [run, hasOutput, totalDuration])

  const getActionRender = useCallback(
    (action: TimelineAction) => {
      const td = TRACK_DEFS.find((t) => action.id.startsWith(t.id + '_'))
      const stepId = action.id.split('_').slice(1).join('_')
      const state = run?.step_states[stepId]
      const url = run ? artifactUrl(run, stepId) : null
      const isCompleted = state?.status === 'completed' && !!url

      return (
        <div
          className={`timeline-clip ${isCompleted ? 'timeline-clip--ready' : 'timeline-clip--pending'}`}
          style={{ borderLeft: `3px solid ${td?.color || '#888'}` }}
        >
          <span className="timeline-clip-label">{stepId.replace(/_/g, ' ')}</span>
          {isCompleted && state?.duration_ms != null && (
            <span className="timeline-clip-dur">{(state.duration_ms / 1000).toFixed(1)}s</span>
          )}
        </div>
      )
    },
    [run],
  )

  const handleClickAction = useCallback(
    (_e: React.MouseEvent, { action }: { action: TimelineAction; row: TimelineRow }) => {
      const stepId = action.id.split('_').slice(1).join('_')
      setSelectedStep(stepId)
    },
    [setSelectedStep],
  )

  if (!hasOutput) {
    return <SimpleTimeline spec={spec} run={run} />
  }

  return (
    <div className="timeline-pro">
      <div className="timeline-pro-header">
        <span className="timeline-pro-title">Timeline</span>
        <span className="timeline-pro-duration">{totalDuration.toFixed(1)}s</span>
      </div>
      <div className="timeline-pro-body">
        <Timeline
          ref={timelineRef}
          editorData={rows}
          effects={effects}
          scale={1}
          scaleWidth={100}
          scaleSplitCount={10}
          minScaleCount={2}
          rowHeight={36}
          hideCursor={false}
          disableDrag={false}
          getActionRender={getActionRender}
          onClickAction={handleClickAction}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  )
}

function SimpleTimeline({ spec, run }: Props) {
  return (
    <div className="timeline-bar">
      <div className="timeline-stages">
        {spec.steps.map((step) => {
          const state = run?.step_states[step.id]
          const status = state?.status ?? 'pending'
          return (
            <div key={step.id} className={`timeline-stage timeline-stage--${status}`}>
              <div className="stage-dot" />
              <div className="stage-label">{step.id.replace(/_/g, ' ')}</div>
              {status === 'running' && <div className="stage-pulse" />}
            </div>
          )
        })}
      </div>
      <div className="timeline-progress">
        {run && (() => {
          const completed = Object.values(run.step_states).filter((s) => s.status === 'completed').length
          return `${completed}/${spec.steps.length} steps`
        })()}
      </div>
    </div>
  )
}
