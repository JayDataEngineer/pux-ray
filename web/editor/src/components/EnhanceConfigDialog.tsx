import { useState, useEffect } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useEnhanceStore, type EnhanceModel } from "@/stores/enhancement"
import { useToastStore } from "@/stores/toast"
import { fetchLLMModels } from "@/api"
import { Sparkles, Plus, Trash2, Check, Pencil, X, RefreshCw, Loader2 } from "lucide-react"

// Popular model presets for quick access
const MODEL_PRESETS = {
  openai: ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
  anthropic: ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"],
  google: ["gemini-pro", "gemini-ultra"],
}

interface EnhanceConfigDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function EnhanceConfigDialog({ open, onOpenChange }: EnhanceConfigDialogProps) {
  const models = useEnhanceStore((s) => s.models)
  const activeId = useEnhanceStore((s) => s.activeId)
  const addModel = useEnhanceStore((s) => s.addModel)
  const removeModel = useEnhanceStore((s) => s.removeModel)
  const setActive = useEnhanceStore((s) => s.setActive)
  const updateModel = useEnhanceStore((s) => s.updateModel)
  const toast = useToastStore((s) => s.addToast)

  const [editingId, setEditingId] = useState<string | null>(null)
  const [form, setForm] = useState({
    name: "",
    baseUrl: "",
    apiKey: "",
    model: "",
  })

  // Model fetching state
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [isLoadingModels, setIsLoadingModels] = useState(false)
  const [modelsError, setModelsError] = useState<string | null>(null)

  // Detect provider from base URL for presets
  const detectProvider = (baseUrl: string): keyof typeof MODEL_PRESETS | 'other' => {
    const url = baseUrl.toLowerCase()
    if (url.includes('openai.com') || url.includes('api.openai.com')) return 'openai'
    if (url.includes('anthropic.com') || url.includes('api.anthropic.com')) return 'anthropic'
    if (url.includes('google.com') || url.includes('generativelanguage.googleapis.com')) return 'google'
    return 'other'
  }

  const suggestedPresets = MODEL_PRESETS[detectProvider(form.baseUrl)] || []

  const resetForm = () => {
    setForm({ name: "", baseUrl: "https://api.openai.com/v1", apiKey: "", model: "gpt-4o-mini" })
    setEditingId(null)
    setAvailableModels([])
    setModelsError(null)
  }

  const loadModels = async () => {
    if (!form.baseUrl || !form.apiKey) {
      setModelsError("Enter base URL and API key first")
      return
    }

    setIsLoadingModels(true)
    setModelsError(null)

    try {
      const models = await fetchLLMModels(form.baseUrl, form.apiKey)
      setAvailableModels(models)

      // Auto-select the first model if none is selected
      if (!form.model && models.length > 0) {
        setForm((f) => ({ ...f, model: models[0] }))
      }

      if (models.length === 0) {
        setModelsError("No models found")
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to fetch models"
      setModelsError(errorMessage)
      setAvailableModels([])
    } finally {
      setIsLoadingModels(false)
    }
  }

  // Auto-fetch models when baseUrl or apiKey changes (in add mode)
  useEffect(() => {
    if (editingId === null && form.baseUrl && form.apiKey) {
      const timer = setTimeout(() => {
        loadModels()
      }, 500) // Debounce to avoid too many requests

      return () => clearTimeout(timer)
    }
  }, [form.baseUrl, form.apiKey, editingId])

  const startAdd = () => {
    resetForm()
  }

  const startEdit = async (m: EnhanceModel) => {
    setEditingId(m.id)
    setForm({ name: m.name, baseUrl: m.baseUrl, apiKey: m.apiKey, model: m.model })

    // Fetch models for the edited endpoint
    if (m.baseUrl && m.apiKey) {
      try {
        setIsLoadingModels(true)
        setModelsError(null)
        const models = await fetchLLMModels(m.baseUrl, m.apiKey)
        setAvailableModels(models)
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "Failed to fetch models"
        setModelsError(errorMessage)
        setAvailableModels([])
      } finally {
        setIsLoadingModels(false)
      }
    }
  }

  const handleSave = () => {
    if (!form.name.trim() || !form.baseUrl.trim() || !form.apiKey.trim() || !form.model.trim()) {
      toast("error", "All fields are required")
      return
    }
    if (editingId) {
      updateModel(editingId, form)
      toast("success", `Updated: ${form.name}`)
    } else {
      const m = addModel(form)
      setActive(m.id)
      toast("success", `Added: ${form.name}`)
    }
    resetForm()
  }

  const handleCancel = () => {
    resetForm()
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-base">
            <Sparkles className="h-4 w-4 text-primary" />
            AI Prompt Enhancement
          </DialogTitle>
          <DialogDescription>
            Add an OpenAI-compatible endpoint to power prompt enhancement. Enhancement prompts are configured automatically per service.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 pt-2">
          {/* Existing models */}
          {models.length > 0 && (
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Endpoints</Label>
              {models.map((m) => (
                <div key={m.id}
                  className={`flex items-center gap-2 rounded-md border px-3 py-2 transition-colors ${
                    activeId === m.id ? "border-primary bg-primary/5" : "border-border"
                  }`}>
                  <button className="flex-1 text-left" onClick={() => setActive(m.id)}>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">{m.name}</span>
                      {activeId === m.id && (
                        <Badge variant="default" className="text-[9px] h-4 px-1">Active</Badge>
                      )}
                    </div>
                    <span className="text-[10px] text-muted-foreground">{m.model} · {m.baseUrl}</span>
                  </button>
                  <Button variant="ghost" size="icon" className="h-6 w-6 shrink-0" onClick={() => startEdit(m)}>
                    <Pencil className="h-3 w-3" />
                  </Button>
                  <Button variant="ghost" size="icon" className="h-6 w-6 shrink-0" onClick={() => { removeModel(m.id); toast("info", `Removed: ${m.name}`) }}>
                    <Trash2 className="h-3 w-3 text-destructive" />
                  </Button>
                </div>
              ))}
            </div>
          )}

          {/* Add / Edit form */}
          {editingId !== null || models.length === 0 ? (
            <div className="space-y-3 rounded-md border border-dashed p-3">
              <div className="flex items-center justify-between">
                <Label className="text-xs font-semibold">{editingId ? "Edit Endpoint" : "Add Endpoint"}</Label>
                {editingId && (
                  <Button variant="ghost" size="icon" className="h-5 w-5" onClick={handleCancel}>
                    <X className="h-3 w-3" />
                  </Button>
                )}
              </div>

              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">Name</Label>
                <Input placeholder="e.g. GPT-4o Mini" value={form.name}
                  onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))} className="h-8 text-xs" />
              </div>

              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">Base URL</Label>
                <Input placeholder="https://api.openai.com/v1" value={form.baseUrl}
                  onChange={(e) => setForm((f) => ({ ...f, baseUrl: e.target.value }))} className="h-8 text-xs" />
              </div>

              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">API Key</Label>
                <Input type="password" placeholder="sk-..." value={form.apiKey}
                  onChange={(e) => setForm((f) => ({ ...f, apiKey: e.target.value }))} className="h-8 text-xs" />
              </div>

              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label className="text-xs text-muted-foreground">Model</Label>
                  {(form.baseUrl || form.apiKey) && (
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-5 w-5"
                      onClick={loadModels}
                      disabled={isLoadingModels || !form.baseUrl || !form.apiKey}
                      title="Refresh models"
                    >
                      {isLoadingModels ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <RefreshCw className="h-3 w-3" />
                      )}
                    </Button>
                  )}
                </div>

                {availableModels.length > 0 ? (
                  <Select value={form.model} onValueChange={(value) => setForm((f) => ({ ...f, model: value }))}>
                    <SelectTrigger className="h-8 text-xs">
                      <SelectValue placeholder="Select a model" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableModels.map((modelId) => (
                        <SelectItem key={modelId} value={modelId} className="text-xs">
                          {modelId}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <div className="space-y-2">
                    <Input
                      placeholder="gpt-4o-mini"
                      value={form.model}
                      onChange={(e) => setForm((f) => ({ ...f, model: e.target.value }))}
                      className="h-8 text-xs"
                    />
                    {modelsError && (
                      <p className="text-[10px] text-destructive">{modelsError}</p>
                    )}
                    {!modelsError && !isLoadingModels && suggestedPresets.length > 0 && !form.model && (
                      <div className="flex flex-wrap gap-1">
                        {suggestedPresets.slice(0, 4).map((preset) => (
                          <button
                            key={preset}
                            type="button"
                            onClick={() => setForm((f) => ({ ...f, model: preset }))}
                            className="text-[9px] px-2 py-0.5 rounded bg-muted hover:bg-muted-foreground/20 transition-colors"
                          >
                            {preset}
                          </button>
                        ))}
                      </div>
                    )}
                    {!modelsError && !isLoadingModels && form.baseUrl && form.apiKey && (
                      <p className="text-[10px] text-muted-foreground">
                        Enter credentials above to auto-fetch models
                      </p>
                    )}
                  </div>
                )}
              </div>

              <div className="flex gap-2 pt-1">
                <Button size="sm" className="h-7 text-xs" onClick={handleSave}>
                  <Check className="h-3 w-3 mr-1" />
                  {editingId ? "Update" : "Add Endpoint"}
                </Button>
                {editingId && (
                  <Button variant="outline" size="sm" className="h-7 text-xs" onClick={handleCancel}>Cancel</Button>
                )}
              </div>
            </div>
          ) : (
            <Button variant="outline" size="sm" className="w-full h-8 text-xs" onClick={startAdd}>
              <Plus className="h-3 w-3 mr-1" /> Add Endpoint
            </Button>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
