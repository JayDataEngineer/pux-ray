import { useState, useEffect } from "react"
import { Label } from "@/components/ui/label"

interface LoraPickerProps {
  model: string
  value: string
  onChange: (v: string) => void
  /** Render variant: "dark" for video editor, "light" for assets tab */
  variant?: "dark" | "light"
}

/** Parse comma-separated "name:strength" string into {name, strength} pairs.
 *  Missing strength defaults to 1.0. Backward-compat with bare names. */
function parseLoras(raw: string): { name: string; strength: number }[] {
  if (!raw) return []
  return raw.split(",").map(s => s.trim()).filter(Boolean).map(entry => {
    const idx = entry.lastIndexOf(":")
    if (idx === -1) return { name: entry, strength: 1.0 }
    const name = entry.slice(0, idx).trim()
    const str = parseFloat(entry.slice(idx + 1))
    return { name, strength: isNaN(str) ? 1.0 : str }
  })
}

/** Encode parsed entries back to comma-separated "name:strength" string.
 *  Strengths of 1.0 are omitted for compactness. */
function encodeLoras(entries: { name: string; strength: number }[]): string {
  return entries.map(e =>
    e.strength === 1.0 ? e.name : `${e.name}:${e.strength}`
  ).join(", ")
}

export function LoraPicker({ model, value, onChange, variant = "dark" }: LoraPickerProps) {
  const [available, setAvailable] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const entries = parseLoras(value)
  const selectedNames = new Set(entries.map(e => e.name))

  useEffect(() => {
    setLoading(true)
    fetch(`/v1/loras?model=${encodeURIComponent(model)}`)
      .then(r => r.json())
      .then((data: { loras: string[] }) => {
        setAvailable(data.loras || [])
      })
      .catch(() => setAvailable([]))
      .finally(() => setLoading(false))
  }, [model])

  const toggle = (name: string) => {
    const next = selectedNames.has(name)
      ? entries.filter(e => e.name !== name)
      : [...entries, { name, strength: 1.0 }]
    onChange(encodeLoras(next))
  }

  const setStrength = (name: string, strength: number) => {
    const next = entries.map(e => e.name === name ? { ...e, strength: Math.round(strength * 100) / 100 } : e)
    onChange(encodeLoras(next))
  }

  const isDark = variant === "dark"
  const textMuted = isDark ? "text-white/25" : "text-muted-foreground/50"
  const labelClass = isDark
    ? "text-[9px] font-medium text-white/30 uppercase tracking-wider"
    : "text-xs text-muted-foreground"
  const activeBg = isDark ? "bg-[#6366f1]/20 text-[#6366f1]" : "bg-primary/15 text-primary"
  const inactiveText = isDark ? "text-white/40 hover:bg-white/5 hover:text-white/60" : "text-muted-foreground hover:bg-accent/50"
  const checkBg = isDark ? "bg-[#6366f1] border-[#6366f1]" : "bg-primary border-primary"
  const borderClass = isDark ? "border-white/20" : "border-border"

  if (loading) {
    return <div className={`text-[10px] ${textMuted} py-1`}>Loading LoRAs…</div>
  }

  if (available.length === 0) {
    return <div className={`text-[10px] ${textMuted} py-1`}>No LoRAs available for this model</div>
  }

  return (
    <div className="space-y-1.5">
      <Label className={labelClass}>Available LoRAs</Label>
      <div className={`space-y-0.5 max-h-40 overflow-y-auto ${isDark ? "scrollbar-thin" : ""}`}>
        {available.map(name => {
          const active = selectedNames.has(name)
          const shortLabel = name
            .replace(".safetensors", "")
            .replace(/^ltx-2\.?3?-?/, "")
            .replace(/^22b-/, "")
            .replace(/^19b-/, "")
            .replace(/^id-lora-/, "ID: ")
            .replace(/^celebvhq-?/, "")
            .replace(/-lora-384(-\d[\d.]*)?$/, "")
            .replace(/distilled/, "distilled")
            .replace(/^-/, "")
          const entry = entries.find(e => e.name === name)
          const strength = entry?.strength ?? 1.0
          return (
            <div key={name}>
              <button onClick={() => toggle(name)}
                className={`w-full flex items-center gap-2 px-2 py-1 rounded text-left text-[10px] transition-colors ${active ? activeBg : inactiveText}`}>
                <div className={`w-2.5 h-2.5 rounded-sm border shrink-0 flex items-center justify-center ${active ? checkBg : borderClass}`}>
                  {active && <svg className="w-2 h-2 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>}
                </div>
                <span className="truncate flex-1" title={name}>{shortLabel || name}</span>
                {active && <span className="text-[9px] opacity-60 tabular-nums">{strength.toFixed(2)}</span>}
              </button>
              {active && (
                <div className="flex items-center gap-1.5 px-2 pb-1">
                  <input type="range" min={0} max={2} step={0.05} value={strength}
                    onChange={e => setStrength(name, Number(e.target.value))}
                    className="flex-1 h-1 appearance-none bg-white/10 rounded-full accent-[#6366f1] [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-2.5 [&::-webkit-slider-thumb]:w-2.5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white/70" />
                  <span className="text-[9px] font-mono text-white/30 w-7 tabular-nums">{strength.toFixed(2)}</span>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
