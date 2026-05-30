import { create } from 'zustand'
import type { WorkflowSpec } from '../types'

interface WorkflowStore {
  spec: WorkflowSpec | null
  loading: boolean
  setSpec: (spec: WorkflowSpec) => void
  setLoading: (loading: boolean) => void
}

export const useWorkflowStore = create<WorkflowStore>((set) => ({
  spec: null,
  loading: false,
  setSpec: (spec) => set({ spec }),
  setLoading: (loading) => set({ loading }),
}))
