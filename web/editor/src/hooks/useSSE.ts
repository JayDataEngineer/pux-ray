import { useCallback, useEffect, useRef } from 'react'
import { sseUrl } from '../api'
import type { SSEEvent } from '../types'

export function useSSE(
  specName: string | null,
  runId: string | null,
  onEvent: (event: SSEEvent) => void,
) {
  const esRef = useRef<EventSource | null>(null)
  const onEventRef = useRef(onEvent)
  onEventRef.current = onEvent

  const connect = useCallback(() => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
    }
    if (!specName || !runId) return

    const url = sseUrl(specName, runId)
    const es = new EventSource(url)
    esRef.current = es

    es.addEventListener('connected', (e) => {
      const data = JSON.parse(e.data)
      onEventRef.current({ event: 'connected', ...data })
    })
    es.addEventListener('step_started', (e) => {
      const data = JSON.parse(e.data)
      onEventRef.current({ event: 'step_started', ...data })
    })
    es.addEventListener('step_completed', (e) => {
      const data = JSON.parse(e.data)
      onEventRef.current({ event: 'step_completed', ...data })
    })
    es.addEventListener('step_failed', (e) => {
      const data = JSON.parse(e.data)
      onEventRef.current({ event: 'step_failed', ...data })
    })
    es.addEventListener('step_waiting', (e) => {
      const data = JSON.parse(e.data)
      onEventRef.current({ event: 'step_waiting', ...data })
    })
    es.addEventListener('workflow_completed', (e) => {
      const data = JSON.parse(e.data)
      onEventRef.current({ event: 'workflow_completed', ...data })
      es.close()
    })
    es.addEventListener('workflow_failed', (e) => {
      const data = JSON.parse(e.data)
      onEventRef.current({ event: 'workflow_failed', ...data })
      es.close()
    })
    es.addEventListener('error', () => {
      es.close()
    })
  }, [specName, runId])

  useEffect(() => {
    connect()
    return () => {
      if (esRef.current) {
        esRef.current.close()
        esRef.current = null
      }
    }
  }, [connect])

  return { reconnect: connect }
}
