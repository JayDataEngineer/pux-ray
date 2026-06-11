import type { EnhanceModel } from "@/stores/enhancement"

/**
 * Store an API key securely on the backend.
 * Returns the key_id to use for future enhancement calls.
 */
export async function storeEnhanceKey(model: Omit<EnhanceModel, 'id' | 'keyId'>): Promise<string> {
  const res = await fetch('/v1/llm/keys', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: model.name,
      baseUrl: model.baseUrl,
      apiKey: model.apiKey,
      model: model.model,
    }),
  })

  if (!res.ok) {
    const body = await res.text().catch(() => "")
    throw new Error(`Failed to store API key (${res.status}): ${body.slice(0, 200)}`)
  }

  const data = await res.json()
  return data.key_id
}

/**
 * List stored enhance keys (metadata only, no API keys).
 */
export async function listEnhanceKeys(): Promise<Array<{ key_id: string; name: string; baseUrl: string; model: string }>> {
  const res = await fetch('/v1/llm/keys')

  if (!res.ok) {
    const body = await res.text().catch(() => "")
    throw new Error(`Failed to list keys (${res.status}): ${body.slice(0, 200)}`)
  }

  const data = await res.json()
  return data.keys || []
}

/**
 * Delete a stored enhance key.
 */
export async function deleteEnhanceKey(keyId: string): Promise<void> {
  const res = await fetch(`/v1/llm/keys/${keyId}`, {
    method: 'DELETE',
  })

  if (!res.ok) {
    const body = await res.text().catch(() => "")
    throw new Error(`Failed to delete key (${res.status}): ${body.slice(0, 200)}`)
  }
}

/**
 * Enhance a prompt using a stored backend key.
 * The backend handles the API key securely.
 */
export async function enhancePrompt(keyId: string, systemPrompt: string, prompt: string): Promise<string> {
  const res = await fetch('/v1/llm/enhance', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      key_id: keyId,
      system_prompt: systemPrompt,
      prompt: prompt,
    }),
  })

  if (!res.ok) {
    const body = await res.text().catch(() => "")
    throw new Error(`Enhancement failed (${res.status}): ${body.slice(0, 200)}`)
  }

  const data = await res.json()
  return data.result
}

/**
 * @deprecated Use enhancePrompt with keyId instead
 * Legacy method for backward compatibility - will be removed
 */
export async function enhancePromptLegacy(model: EnhanceModel, systemPrompt: string, prompt: string): Promise<string> {
  const baseUrl = model.baseUrl.replace(/\/+$/, "")
  const url = `${baseUrl}/chat/completions`

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${model.apiKey}`,
    },
    body: JSON.stringify({
      model: model.model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: prompt },
      ],
      max_tokens: 1024,
      temperature: 0.8,
    }),
  })

  if (!res.ok) {
    const body = await res.text().catch(() => "")
    throw new Error(`Enhancement failed (${res.status}): ${body.slice(0, 200)}`)
  }

  const data = await res.json()
  const text = data?.choices?.[0]?.message?.content
  if (!text) throw new Error("Model returned empty response")
  return text.trim()
}
