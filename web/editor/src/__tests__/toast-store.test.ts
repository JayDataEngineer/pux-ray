import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useToastStore } from '../stores/toast'

beforeEach(() => {
  // Clear all toasts
  const { toasts } = useToastStore.getState()
  for (const t of toasts) {
    useToastStore.getState().removeToast(t.id)
  }
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('ToastStore', () => {
  it('starts with no toasts', () => {
    expect(useToastStore.getState().toasts).toHaveLength(0)
  })

  it('addToast creates a toast with id', () => {
    useToastStore.getState().addToast('info', 'Hello')
    const { toasts } = useToastStore.getState()
    expect(toasts).toHaveLength(1)
    expect(toasts[0].type).toBe('info')
    expect(toasts[0].message).toBe('Hello')
    expect(toasts[0].id).toBeTypeOf('number')
  })

  it('addToast auto-removes after 5 seconds', () => {
    useToastStore.getState().addToast('success', 'Done')
    expect(useToastStore.getState().toasts).toHaveLength(1)

    vi.advanceTimersByTime(5000)
    expect(useToastStore.getState().toasts).toHaveLength(0)
  })

  it('removeToast removes specific toast', () => {
    useToastStore.getState().addToast('error', 'Oops')
    const id = useToastStore.getState().toasts[0].id
    useToastStore.getState().removeToast(id)
    expect(useToastStore.getState().toasts).toHaveLength(0)
  })

  it('supports multiple toasts', () => {
    useToastStore.getState().addToast('info', 'One')
    useToastStore.getState().addToast('error', 'Two')
    useToastStore.getState().addToast('success', 'Three')
    expect(useToastStore.getState().toasts).toHaveLength(3)
  })
})
