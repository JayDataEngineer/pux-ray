import { useState, useEffect } from "react"
import { Label } from "@/components/ui/label"

interface LoraPickerProps {
  model: string
  value: string
  onChange: (v: string) => void
  /** Render variant: "dark" for video editor, "light" for assets tab */
  variant?: "dark" | "light"
}

export function LoraPicker({ model, value, onChange, variant = "dark" }: LoraPickerProps) {
  const [available, setAvailable] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const selected = value ? value.split(",").map(s => s.trim()).filter(Boolean) : []

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
    const next = selected.includes(name)
      ? selected.filter(s => s !== name)
      : [...selected, name]
    onChange(next.join(", "))
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
          const active = selected.includes(name)
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
          return (
            <button key={name} onClick={() => toggle(name)}
              className={`w-full flex items-center gap-2 px-2 py-1 rounded text-left text-[10px] transition-colors ${active ? activeBg : inactiveText}`}>
              <div className={`w-2.5 h-2.5 rounded-sm border shrink-0 flex items-center justify-center ${active ? checkBg : borderClass}`}>
                {active && <svg className="w-2 h-2 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>}
              </div>
              <span className="truncate" title={name}>{shortLabel || name}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
