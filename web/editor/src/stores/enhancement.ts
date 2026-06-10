import { create } from 'zustand'

export interface EnhanceModel {
  id: string
  name: string
  baseUrl: string       // e.g. "https://api.openai.com/v1"
  apiKey: string
  model: string         // e.g. "gpt-4o-mini"
}

interface EnhancementStore {
  models: EnhanceModel[]
  activeId: string | null
  addModel: (model: Omit<EnhanceModel, 'id'>) => EnhanceModel
  updateModel: (id: string, patch: Partial<Omit<EnhanceModel, 'id'>>) => void
  removeModel: (id: string) => void
  setActive: (id: string | null) => void
  activeModel: () => EnhanceModel | undefined
}

let _nextId = 1
function uid(): string { return `em_${_nextId++}_${Date.now().toString(36)}` }

function loadPersisted(): { models: EnhanceModel[]; activeId: string | null } {
  try {
    const raw = localStorage.getItem('tech_noir_enhance')
    if (raw) return JSON.parse(raw)
  } catch {}
  return { models: [], activeId: null }
}

function persist(models: EnhanceModel[], activeId: string | null) {
  try { localStorage.setItem('tech_noir_enhance', JSON.stringify({ models, activeId })) } catch {}
}

export const useEnhanceStore = create<EnhancementStore>((set, get) => {
  const saved = loadPersisted()
  return {
    models: saved.models,
    activeId: saved.activeId,

    addModel: (partial) => {
      const model: EnhanceModel = { ...partial, id: uid() }
      set((s) => {
        const models = [...s.models, model]
        const activeId = s.activeId ?? model.id
        persist(models, activeId)
        return { models, activeId }
      })
      return model
    },

    updateModel: (id, patch) => set((s) => {
      const models = s.models.map((m) => m.id === id ? { ...m, ...patch } : m)
      persist(models, s.activeId)
      return { models }
    }),

    removeModel: (id) => set((s) => {
      const models = s.models.filter((m) => m.id !== id)
      const activeId = s.activeId === id ? (models[0]?.id ?? null) : s.activeId
      persist(models, activeId)
      return { models, activeId }
    }),

    setActive: (id) => {
      persist(get().models, id)
      set({ activeId: id })
    },

    activeModel: () => {
      const s = get()
      return s.models.find((m) => m.id === s.activeId)
    },
  }
})
