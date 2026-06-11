import type { WorkflowSpec, WorkflowRun, ServiceInfo, ServiceResult } from './types'

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

    if (!resp.ok) {
      initPromise = null  // Allow retry on next call
      throw new Error(`MCP initialize failed: ${resp.status} ${await resp.text()}`)
    }

    const text = await resp.text()
    const dataLine = text.split('\n').find((l) => l.startsWith('data: '))
    if (!dataLine) {
      initPromise = null
      throw new Error('MCP initialize: no data line in response')
    }

    const msg = JSON.parse(dataLine.slice(6))
    if (msg.error) {
      initPromise = null
      throw new Error(`MCP initialize error: ${msg.error.message}`)
    }
    sessionId = msg.result?.sessionId ?? null

    // Send initialized notification (best-effort, stateless servers may ignore it)
    try {
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
    } catch {
      // Stateless servers don't require this notification
    }
  })()

  return initPromise
}

let nextId = 10

export async function callTool<T = unknown>(name: string, args: Record<string, unknown> = {}, signal?: AbortSignal): Promise<T> {
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
    signal,
  })

  const text = await resp.text()
  const dataLine = text.split('\n').find((l) => l.startsWith('data: '))
  if (!dataLine) throw new Error('No response from MCP server')

  const dataStr = dataLine.slice(6)
  let msg: any
  try {
    msg = JSON.parse(dataStr)
  } catch {
    // FastMCP sometimes returns plain text errors (e.g. "Error calling tool...")
    throw new Error(dataStr.trim() || 'MCP returned invalid JSON')
  }
  if (msg.error) throw new Error(msg.error.message || 'MCP tool error')

  // FastMCP wraps result in content[{type:"text", text:"..."}]
  const content = msg.result?.content
  if (content?.[0]?.type === 'text') {
    try {
      return JSON.parse(content[0].text)
    } catch {
      throw new Error(content[0].text?.slice(0, 200) || 'MCP returned unparseable result')
    }
  }
  return msg.result as T
}

export interface MCPTool {
  name: string
  description?: string
  inputSchema: {
    type?: string
    properties?: Record<string, {
      type?: string
      default?: unknown
      description?: string
      enum?: string[]
      anyOf?: { type?: string; default?: unknown; additionalProperties?: boolean }[]
      additionalProperties?: boolean
    }>
    required?: string[]
    additionalProperties?: boolean
  }
  outputSchema?: Record<string, unknown>
  _meta?: Record<string, unknown>
}

export async function listTools(): Promise<MCPTool[]> {
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
      method: 'tools/list',
      params: {},
    }),
  })
  const text = await resp.text()
  const dataLine = text.split('\n').find((l) => l.startsWith('data: '))
  if (!dataLine) throw new Error('No response from MCP server')
  const msg = JSON.parse(dataLine.slice(6))
  if (msg.error) throw new Error(msg.error.message || 'MCP listTools error')
  return msg.result?.tools ?? []
}

// ========== Convenience wrappers ==========

export async function listSpecs() {
  const data = await callTool<{ name: string; description: string; steps: number }[] | { data: { name: string; description: string; steps: number }[] }>('workflow_list_specs', {})
  return Array.isArray(data) ? data : (data as { data: unknown[] }).data as { name: string; description: string; steps: number }[]
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

// ========== Service Catalog (REST API — direct, no MCP wrapper needed) ==========

export async function listServices(): Promise<ServiceInfo[]> {
  const res = await fetch('/v1/services')
  if (!res.ok) throw new Error(`Failed to fetch services: ${res.status}`)
  return res.json()
}

export async function getServiceInfo(name: string): Promise<ServiceInfo & { default_model: string }> {
  const res = await fetch(`/v1/services/${name}`)
  if (!res.ok) throw new Error(`Failed to fetch service ${name}: ${res.status}`)
  return res.json()
}

export async function invokeService(
  service: string,
  params: Record<string, unknown>,
  model?: string,
): Promise<ServiceResult> {
  const payload: Record<string, unknown> = { service, ...params }
  if (model) payload.model = model
  const res = await fetch('/v1/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(`Service invoke failed: ${res.status}`)
  return res.json()
}

export async function invokeServiceFormData(
  service: string,
  formData: FormData,
): Promise<ServiceResult> {
  const res = await fetch('/v1/run', {
    method: 'POST',
    body: formData,
  })
  const text = await res.text()
  try { return JSON.parse(text) }
  catch { return { status: 'error', error: text.slice(0, 500) } }
}

export async function loadService(name: string, model?: string): Promise<ServiceResult> {
  const res = await fetch('/forge', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'preload', service: name, model }),
  })
  if (!res.ok) throw new Error(`Failed to load service: ${res.status}`)
  return res.json()
}

export async function forgeStatus(): Promise<{
  loaded: Record<string, number>
  vram_free_mb: number
  vram_total_mb: number
  gpu?: { device: string; total_mb: number; allocated_mb: number }
}> {
  const res = await fetch('/status')
  if (!res.ok) throw new Error(`Failed to get status: ${res.status}`)
  return res.json()
}

/** Read a File as a base64 data URL string. */
export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(new Error('Failed to read file'))
    reader.readAsDataURL(file)
  })
}

/**
 * Fetch available models from an OpenAI-compatible API endpoint
 * @param baseUrl - The base URL of the API (e.g., "https://api.openai.com/v1")
 * @param apiKey - The API key to authenticate with
 * @returns Array of available model IDs
 */
export async function fetchLLMModels(baseUrl: string, apiKey: string): Promise<string[]> {
  // Ensure baseUrl doesn't end with /v1 - we'll add it ourselves
  const cleanBaseUrl = baseUrl.replace(/\/v1\/?$/, '')
  const modelsUrl = `${cleanBaseUrl}/v1/models`

  try {
    const response = await fetch(modelsUrl, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`Failed to fetch models: ${response.status} ${errorText}`)
    }

    const data = await response.json()

    // OpenAI API returns { object: "list", data: [{ id: "model-name", ... }] }
    if (data.data && Array.isArray(data.data)) {
      return data.data.map((model: any) => model.id).sort()
    }

    // Fallback: some providers might return just an array
    if (Array.isArray(data)) {
      return data.map((model: any) => model.id || model).sort()
    }

    throw new Error('Unexpected response format from models API')
  } catch (error) {
    console.error('Error fetching models:', error)
    throw error
  }
}
