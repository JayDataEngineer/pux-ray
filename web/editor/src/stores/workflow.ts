export type ViewMode = string
import { create } from 'zustand'
import type { WorkflowSpec, WorkflowRun } from '../types'

interface WorkflowStore {
  viewMode: string
  spec: WorkflowSpec | null
  run: WorkflowRun | null
  selectedStepId: string | null
  loading: boolean
  setSpec: (spec: WorkflowSpec) => void
  setRun: (run: WorkflowRun) => void
  setSelectedStep: (stepId: string | null) => void
  setLoading: (loading: boolean) => void
  setViewMode: (mode: string) => void
}

export const useWorkflowStore = create<WorkflowStore>((set) => ({
  spec: null,
  run: null,
  selectedStepId: null,
  loading: false,
  viewMode: "timeline",
  setSpec: (spec) => set({ spec }),
  setRun: (run) => set({ run }),
  setSelectedStep: (selectedStepId) => set({ selectedStepId }),
  setLoading: (loading) => set({ loading }),
  setViewMode: (viewMode) => set({ viewMode }),
}))
