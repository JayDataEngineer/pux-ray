export interface InputSpec {
  type: string
  required: boolean
  default?: unknown
  description?: string
  enum?: string[]
}

export interface StepSpecInfo {
  id: string
  type: string
  service?: string
  depends_on: string[]
  outputs: string[]
  interaction?: string | null
}

export interface WorkflowSpec {
  name: string
  version: string
  description: string
  inputs: Record<string, InputSpec>
  steps: StepSpecInfo[]
}

export interface StepState {
  step_id: string
  status: 'pending' | 'running' | 'waiting_input' | 'completed' | 'failed' | 'skipped'
  outputs: Record<string, string>
  error?: string | null
  duration_ms?: number | null
  started_at?: string | null
  completed_at?: string | null
}

export interface ArtifactRef {
  run_id: string
  step_id: string
  name: string
  file_path: string
  media_type: string
  url: string
  size_bytes: number
  created_at: string
}

export interface WorkflowRun {
  run_id: string
  spec_name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  inputs: Record<string, unknown>
  step_states: Record<string, StepState>
  artifacts: Record<string, ArtifactRef>
  created_at?: string
  updated_at?: string
}

export interface SSEEvent {
  event: string
  run_id?: string
  step_id?: string
  status?: string
  duration_ms?: number
  outputs?: Record<string, string>
  error?: string
  message?: string
  artifacts?: string[]
  ts?: number
}
