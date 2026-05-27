import { create } from 'zustand'
import type { WorkflowSpec, WorkflowRun, StepState } from '../types'

interface WorkflowStore {
  spec: WorkflowSpec | null
  run: WorkflowRun | null
  selectedStepId: string | null
  loading: boolean
  error: string | null

  setSpec: (spec: WorkflowSpec) => void
  setRun: (run: WorkflowRun) => void
  setSelectedStep: (stepId: string | null) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  updateStepState: (stepId: string, patch: Partial<StepState>) => void
  reset: () => void
}

export const useWorkflowStore = create<WorkflowStore>((set) => ({
  spec: null,
  run: null,
  selectedStepId: null,
  loading: false,
  error: null,

  setSpec: (spec) => set({ spec }),
  setRun: (run) => set({ run }),
  setSelectedStep: (selectedStepId) => set({ selectedStepId }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),

  updateStepState: (stepId, patch) =>
    set((state) => {
      if (!state.run) return state
      const stepStates = { ...state.run.step_states }
      stepStates[stepId] = { ...stepStates[stepId], ...patch }
      return { run: { ...state.run, step_states: stepStates } }
    }),

  reset: () =>
    set({ spec: null, run: null, selectedStepId: null, loading: false, error: null }),
}))
