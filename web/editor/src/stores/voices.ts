import { create } from 'zustand'

export interface SavedVoice {
  id: string
  name: string
  audioB64: string   // raw base64 (no data URL prefix)
  modelId: string    // which model generated it
  createdAt: string
}

interface VoiceStore {
  voices: SavedVoice[]
  addVoice: (voice: Omit<SavedVoice, 'id' | 'createdAt'>) => SavedVoice
  removeVoice: (id: string) => void
}

let _nextId = 1
function uid(): string { return `voice_${_nextId++}_${Date.now().toString(36)}` }

function loadVoices(): SavedVoice[] {
  try { return JSON.parse(localStorage.getItem('tech_noir_voices') || '[]') } catch { return [] }
}
function persist(voices: SavedVoice[]) {
  try { localStorage.setItem('tech_noir_voices', JSON.stringify(voices)) } catch {}
}

export const useVoiceStore = create<VoiceStore>((set) => ({
  voices: loadVoices(),
  addVoice: (partial) => {
    const voice: SavedVoice = { ...partial, id: uid(), createdAt: new Date().toISOString() }
    set((s) => {
      const updated = [...s.voices, voice]
      persist(updated)
      return { voices: updated }
    })
    return voice
  },
  removeVoice: (id) => set((s) => {
    const updated = s.voices.filter((v) => v.id !== id)
    persist(updated)
    return { voices: updated }
  }),
}))
