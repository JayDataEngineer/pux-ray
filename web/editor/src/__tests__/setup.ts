import '@testing-library/jest-dom/vitest'

// Mock localStorage
const store: Record<string, string> = {}
const ls = {
  getItem: (key: string) => store[key] ?? null,
  setItem: (key: string, val: string) => { store[key] = val },
  removeItem: (key: string) => { delete store[key] },
  clear: () => { Object.keys(store).forEach(k => delete store[k]) },
  get length() { return Object.keys(store).length },
  key: (i: number) => Object.keys(store)[i] ?? null,
}
Object.defineProperty(globalThis, 'localStorage', { value: ls, writable: true })

// Mock matchMedia
Object.defineProperty(globalThis, 'matchMedia', {
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }),
  writable: true,
})

// Mock ResizeObserver
globalThis.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as any

// Mock IntersectionObserver
globalThis.IntersectionObserver = class {
  constructor() {}
  observe() {}
  unobserve() {}
  disconnect() {}
} as any

// Suppress console.error for expected test noise
const origError = console.error
beforeEach(() => {
  console.error = (...args: unknown[]) => {
    const msg = String(args[0] || '')
    if (msg.includes('Warning:')) return
    origError.call(console, ...args)
  }
})
afterEach(() => {
  console.error = origError
})
