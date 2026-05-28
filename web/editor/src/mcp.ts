import type { WorkflowSpec, WorkflowRun } from './types'

const MCP_URL = '/mcp/wan2gp-studio/mcp'

let sessionId: string | null = null
let initPromise: Promise<void> | null = null

async function ensureInit() {
  if (sessionId) return
  if (initPromise) return initPromise

  initPromise = (async () => {
    const resp = await fetch(MCP_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream',
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: 1,
        method: 'initialize',
        params: {
          protocolVersion: '2024-11-05',
          capabilities: {},
          clientInfo: { name: 'video-editor', version: '1.0' },
        },
      }),
    })

    const text = await resp.text()
    const dataLine = text.split('\n').find((l) => l.startsWith('data: '))
    if (dataLine) {
      const msg = JSON.parse(dataLine.slice(6))
      sessionId = msg.result?.sessionId ?? null
    }

    // Send initialized notification
    await fetch(MCP_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream',
        ...(sessionId ? { 'Mcp-Session-Id': sessionId } : {}),
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'notifications/initialized',
      }),
    })
  })()

  return initPromise
}

let nextId = 10

export async function callTool<T = unknown>(name: string, args: Record<string, unknown> = {}): Promise<T> {
  await ensureInit()

  const resp = await fetch(MCP_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json, text/event-stream',
      ...(sessionId ? { 'Mcp-Session-Id': sessionId } : {}),
    },
    body: JSON.stringify({
      jsonrpc: '2.0',
      id: nextId++,
      method: 'tools/call',
      params: { name, arguments: args },
    }),
  })

  const text = await resp.text()
  const dataLine = text.split('\n').find((l) => l.startsWith('data: '))
  if (!dataLine) throw new Error('No response from MCP server')

  const msg = JSON.parse(dataLine.slice(6))
  if (msg.error) throw new Error(msg.error.message || 'MCP tool error')

  // FastMCP wraps result in content[{type:"text", text:"..."}]
  const content = msg.result?.content
  if (content?.[0]?.type === 'text') {
    return JSON.parse(content[0].text)
  }
  return msg.result as T
}

// ========== Convenience wrappers ==========

export async function listSpecs() {
  const data = await callTool<{ data: { name: string; description: string; steps: number }[] }>('workflow_list_specs', {})
  return data.data
}

export async function getSpec(name: string): Promise<WorkflowSpec> {
  return callTool<WorkflowSpec>('workflow_get_spec', { spec_name: name })
}

export async function startRun(specName: string, inputs: Record<string, unknown>, manual = false): Promise<{ run_id: string; status: string }> {
  return callTool<{ run_id: string; status: string }>('workflow_start_run', {
    spec_name: specName,
    inputs,
    manual,
  })
}

export async function getRun(specName: string, runId: string): Promise<WorkflowRun> {
  return callTool<WorkflowRun>('workflow_get_run', { spec_name: specName, run_id: runId })
}

export async function cancelRun(specName: string, runId: string) {
  return callTool('workflow_cancel_run', { spec_name: specName, run_id: runId })
}

export async function rerunStep(specName: string, runId: string, stepId: string, params?: Record<string, unknown>) {
  return callTool('workflow_rerun_step', {
    spec_name: specName, run_id: runId, step_id: stepId,
    ...(params ? { params } : {}),
  })
}

export async function executeStep(specName: string, runId: string, stepId: string, params?: Record<string, unknown>) {
  return callTool<{ run_id: string; step_id: string; status: string; duration_ms: number; outputs: Record<string, string> }>('workflow_execute_step', {
    spec_name: specName, run_id: runId, step_id: stepId,
    ...(params ? { params } : {}),
  })
}

export async function approveStep(
  specName: string, runId: string, stepId: string,
  data: { file_data: string; name: string; media_type: string },
) {
  return callTool('workflow_approve_step', {
    spec_name: specName, run_id: runId, step_id: stepId, data,
  })
}

export async function continueStep(specName: string, runId: string, stepId: string) {
  return callTool('workflow_approve_step', {
    spec_name: specName, run_id: runId, step_id: stepId,
  })
}

export function artifactUrl(specName: string, runId: string, stepId: string, filename: string) {
  return `/v1/wf/${specName}/runs/${runId}/artifacts/${stepId}/${filename}`
}

export function sseUrl(specName: string, runId: string) {
  return `/v1/wf/${specName}/runs/${runId}/events`
}

export async function loadKimodo(): Promise<{ status: string }> {
  const res = await fetch('/forge', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'preload', service: 'kimodo_demo' }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: '' }))
    throw new Error(body.error || `Failed to load Kimodo`)
  }
  return res.json()
}

export function kimodoUrl() {
  return `${window.location.origin}/kimodo/`
}
