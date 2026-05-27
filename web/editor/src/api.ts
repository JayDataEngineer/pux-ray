import type { WorkflowSpec, WorkflowRun } from './types'

const BASE = '/v1/wf'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function listSpecs() {
  const res = await fetch(`${BASE}`)
  const data = await json<{ data: { name: string; description: string; steps: number }[] }>(res)
  return data.data
}

export async function getSpec(name: string): Promise<WorkflowSpec> {
  const res = await fetch(`${BASE}/${name}`)
  return json<WorkflowSpec>(res)
}

export async function startRun(specName: string, inputs: Record<string, unknown>): Promise<{ run_id: string; status: string }> {
  const res = await fetch(`${BASE}/${specName}/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(inputs),
  })
  return json(res)
}

export async function getRun(specName: string, runId: string): Promise<WorkflowRun> {
  const res = await fetch(`${BASE}/${specName}/runs/${runId}`)
  return json<WorkflowRun>(res)
}

export async function cancelRun(specName: string, runId: string) {
  const res = await fetch(`${BASE}/${specName}/runs/${runId}`, { method: 'DELETE' })
  return json(res)
}

export async function rerunStep(specName: string, runId: string, stepId: string, params?: Record<string, unknown>) {
  const res = await fetch(`${BASE}/${specName}/runs/${runId}/steps/${stepId}/rerun`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: params ? JSON.stringify(params) : '{}',
  })
  return json(res)
}

export async function executeStep(specName: string, runId: string, stepId: string, params?: Record<string, unknown>) {
  const res = await fetch(`${BASE}/${specName}/runs/${runId}/steps/${stepId}/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: params ? JSON.stringify(params) : '{}',
  })
  return json(res)
}

export async function approveStep(
  specName: string, runId: string, stepId: string,
  data: { file_data: string; name: string; media_type: string },
) {
  const res = await fetch(`${BASE}/${specName}/runs/${runId}/steps/${stepId}/approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  return json(res)
}

export function artifactUrl(specName: string, runId: string, stepId: string, filename: string) {
  return `${BASE}/${specName}/runs/${runId}/artifacts/${stepId}/${filename}`
}

export function sseUrl(specName: string, runId: string) {
  return `${BASE}/${specName}/runs/${runId}/events`
}
