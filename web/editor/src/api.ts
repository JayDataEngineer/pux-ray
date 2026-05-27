import type { WorkflowSpec, WorkflowRun } from './types'

const BASE = '/v1/wf'

const STATUS_SUMMARIES: Record<number, string> = {
  400: 'Invalid request',
  401: 'Authentication required',
  403: 'Permission denied',
  404: 'Not found',
  408: 'Request timed out',
  409: 'Conflict — resource already exists',
  422: 'Validation failed',
  429: 'Too many requests — slow down',
  500: 'Server error — try again',
  502: 'Service unavailable',
  503: 'Service overloaded — retry shortly',
  504: 'Gateway timed out',
}

function summarizeError(status: number, detail: string): string {
  if (detail && detail !== `HTTP ${status}`) return detail
  return STATUS_SUMMARIES[status] || `Request failed (${status})`
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: '' }))
    throw new Error(summarizeError(res.status, body.error || body.detail || ''))
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

export async function loadKimodo(): Promise<{ status: string }> {
  const res = await fetch('/forge', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'preload', service: 'kimodo_demo' }),
  })
  return json(res)
}

export function kimodoUrl() {
  return `${window.location.origin}/kimodo/`
}
