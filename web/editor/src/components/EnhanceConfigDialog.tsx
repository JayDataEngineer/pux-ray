import { useState } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { useEnhanceStore, DEFAULT_SYSTEM_PROMPT, type EnhanceModel } from "@/stores/enhancement"
import { useToastStore } from "@/stores/toast"
import { Sparkles, Plus, Trash2, Check, Pencil, X } from "lucide-react"

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
    systemPrompt: DEFAULT_SYSTEM_PROMPT,
  })

  const resetForm = () => {
    setForm({ name: "", baseUrl: "https://api.openai.com/v1", apiKey: "", model: "gpt-4o-mini", systemPrompt: DEFAULT_SYSTEM_PROMPT })
    setEditingId(null)
  }

  const startAdd = () => {
    resetForm()
  }

  const startEdit = (m: EnhanceModel) => {
    setEditingId(m.id)
    setForm({ name: m.name, baseUrl: m.baseUrl, apiKey: m.apiKey, model: m.model, systemPrompt: m.systemPrompt })
  }

  const handleSave = () => {
    if (!form.name.trim() || !form.baseUrl.trim() || !form.apiKey.trim() || !form.model.trim()) {
      toast("error", "All fields except System Prompt are required")
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
            Configure OpenAI-compatible models to enhance your generation prompts.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 pt-2">
          {/* Existing models */}
          {models.length > 0 && (
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Configured Models</Label>
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
                <Label className="text-xs font-semibold">{editingId ? "Edit Model" : "Add Model"}</Label>
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
                <Label className="text-xs text-muted-foreground">Model ID</Label>
                <Input placeholder="gpt-4o-mini" value={form.model}
                  onChange={(e) => setForm((f) => ({ ...f, model: e.target.value }))} className="h-8 text-xs" />
              </div>

              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">Enhancement System Prompt</Label>
                <Textarea rows={4} value={form.systemPrompt}
                  onChange={(e) => setForm((f) => ({ ...f, systemPrompt: e.target.value }))}
                  className="text-xs" placeholder={DEFAULT_SYSTEM_PROMPT} />
              </div>

              <div className="flex gap-2 pt-1">
                <Button size="sm" className="h-7 text-xs" onClick={handleSave}>
                  <Check className="h-3 w-3 mr-1" />
                  {editingId ? "Update" : "Add Model"}
                </Button>
                {editingId && (
                  <Button variant="outline" size="sm" className="h-7 text-xs" onClick={handleCancel}>Cancel</Button>
                )}
              </div>
            </div>
          ) : (
            <Button variant="outline" size="sm" className="w-full h-8 text-xs" onClick={startAdd}>
              <Plus className="h-3 w-3 mr-1" /> Add Model
            </Button>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
