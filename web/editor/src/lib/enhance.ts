import type { EnhanceModel } from "@/stores/enhancement"

export async function enhancePrompt(model: EnhanceModel, systemPrompt: string, prompt: string): Promise<string> {
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
