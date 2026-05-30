import { useState, useEffect, useCallback } from 'react'
import { loadKimodo, kimodoUrl } from '../api'

type Status = 'idle' | 'loading' | 'ready' | 'error'

export function KimodoEmbed() {
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)

  const startKimodo = useCallback(async () => {
    setStatus('loading')
    setError(null)
    try {
      await loadKimodo()
      // Give Viser a few seconds to start serving
      await new Promise((r) => setTimeout(r, 3000))
      // Verify it's reachable
      await fetch(kimodoUrl(), { method: 'HEAD', mode: 'no-cors' })
      // Full-page redirect to Viser
      window.location.href = kimodoUrl()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load Kimodo')
      setStatus('error')
    }
  }, [])

  useEffect(() => {
    if (status === 'idle') {
      startKimodo()
    }
  }, [status, startKimodo])

  if (status === 'loading') {
    return (
      <div className="kimodo-loading">
        <div className="spinner" />
        <p>Starting Kimodo 3D Posing...</p>
        <span className="kimodo-hint">Loading on GPU — you'll be redirected automatically</span>
      </div>
    )
  }

  if (status === 'error') {
    const isVram = error?.includes('VRAM') || error?.includes('Cannot free')
    return (
      <div className="kimodo-error">
        <p>{isVram ? 'Not enough GPU memory to start Kimodo.' : 'Failed to start Kimodo.'}</p>
        {isVram && <span className="kimodo-hint">Free GPU memory by releasing other services, then retry.</span>}
        {!isVram && <span className="kimodo-error-detail">{error}</span>}
        <button className="btn btn-primary" onClick={startKimodo}>Retry</button>
      </div>
    )
  }

  return null
}
