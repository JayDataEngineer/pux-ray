import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import { VideoEditor } from '../components/workspaces/VideoEditor'
import { useTimelineStore } from '../stores/timeline'

// Mock the MCP callTool
vi.mock('../mcp', () => ({
  callTool: vi.fn().mockResolvedValue({ status: 'ok', data: 'base64data', media_type: 'video/mp4' }),
}))

beforeEach(() => {
  useTimelineStore.getState().reset()
})

describe('VideoEditor — rendering', () => {
  it('renders empty state with drop prompt', () => {
    render(<VideoEditor />)
    expect(screen.getByText(/drop images or audio/i)).toBeInTheDocument()
  })

  it('renders play/pause icon', () => {
    render(<VideoEditor />)
    const playIcon = document.querySelector('.lucide-play')
    expect(playIcon).toBeTruthy()
  })

  it('renders Add Keyframe button', () => {
    render(<VideoEditor />)
    // "Add Keyframe" appears in hint text AND the button — use getByRole
    expect(screen.getByRole('button', { name: /add keyframe/i })).toBeInTheDocument()
  })

  it('renders Export button', () => {
    render(<VideoEditor />)
    expect(screen.getByText(/export/i)).toBeInTheDocument()
  })

  it('renders Generate All button', () => {
    render(<VideoEditor />)
    expect(screen.getByText(/generate all/i)).toBeInTheDocument()
  })

  it('renders timeline track labels', () => {
    render(<VideoEditor />)
    expect(screen.getByText('Video')).toBeInTheDocument()
    expect(screen.getByText('Voice')).toBeInTheDocument()
    expect(screen.getByText('SFX')).toBeInTheDocument()
    expect(screen.getByText('Music')).toBeInTheDocument()
  })

  it('renders zoom controls', () => {
    render(<VideoEditor />)
    const zoomIn = document.querySelector('.lucide-zoom-in')
    const zoomOut = document.querySelector('.lucide-zoom-out')
    expect(zoomIn).toBeTruthy()
    expect(zoomOut).toBeTruthy()
  })
})

describe('VideoEditor — adding segments', () => {
  it('clicking Add Keyframe creates a new segment', () => {
    render(<VideoEditor />)
    const addBtn = screen.getByRole('button', { name: /add keyframe/i })
    fireEvent.click(addBtn)

    const { segments } = useTimelineStore.getState()
    expect(segments).toHaveLength(1)
    expect(segments[0].status).toBe('empty')
  })

  it('clicking Add Keyframe twice creates two segments', () => {
    render(<VideoEditor />)
    const addBtn = screen.getByRole('button', { name: /add keyframe/i })
    fireEvent.click(addBtn)
    fireEvent.click(addBtn)

    const { segments } = useTimelineStore.getState()
    expect(segments).toHaveLength(2)
    expect(segments[1].start).toBe(5)
  })

  it('new segment is auto-selected', () => {
    render(<VideoEditor />)
    const addBtn = screen.getByRole('button', { name: /add keyframe/i })
    fireEvent.click(addBtn)

    const { selectedSegmentId, segments } = useTimelineStore.getState()
    expect(selectedSegmentId).toBe(segments[0].id)
  })
})

describe('VideoEditor — segment selection & inspector', () => {
  it('clicking a segment shows the inspector panel', () => {
    const seg = useTimelineStore.getState().addSegment({ prompt: 'test prompt' })
    render(<VideoEditor />)

    const segEl = screen.getByText(`K${String(seg.order + 1).padStart(2, '0')}`)
    fireEvent.click(segEl.closest('[class*="rounded"]')!)

    expect(screen.getByText('Prompts')).toBeInTheDocument()
    expect(screen.getByDisplayValue('test prompt')).toBeInTheDocument()
  })

  it('inspector shows duration field', () => {
    const seg = useTimelineStore.getState().addSegment({ duration: 8 })
    useTimelineStore.getState().setSelectedSegment(seg.id)
    render(<VideoEditor />)

    const durationInput = screen.getByDisplayValue('8')
    expect(durationInput).toBeInTheDocument()
  })

  it('inspector shows FPS selector', () => {
    const seg = useTimelineStore.getState().addSegment()
    useTimelineStore.getState().setSelectedSegment(seg.id)
    render(<VideoEditor />)

    expect(screen.getByText('FPS')).toBeInTheDocument()
  })

  it('inspector shows delete button', () => {
    const seg = useTimelineStore.getState().addSegment()
    useTimelineStore.getState().setSelectedSegment(seg.id)
    render(<VideoEditor />)

    const trashIcon = document.querySelector('.lucide-trash-2')
    expect(trashIcon).toBeTruthy()
  })

  it('inspector shows Start field', () => {
    const seg = useTimelineStore.getState().addSegment({ start: 2.5 })
    useTimelineStore.getState().setSelectedSegment(seg.id)
    render(<VideoEditor />)

    expect(screen.getByText('Start (s)')).toBeInTheDocument()
  })

  it('inspector shows single-segment generate button for empty segment with image', () => {
    const seg = useTimelineStore.getState().addSegment({ firstFrameB64: 'data:image/png;base64,test' })
    useTimelineStore.getState().setSelectedSegment(seg.id)
    render(<VideoEditor />)

    expect(screen.getByText(/generate this segment/i)).toBeInTheDocument()
  })
})

describe('VideoEditor — playback', () => {
  it('skip back button resets time to 0', () => {
    useTimelineStore.getState().addSegment()
    useTimelineStore.getState().setPlayback({ currentTime: 3.5 })
    render(<VideoEditor />)

    const skipBackBtn = document.querySelector('.lucide-skip-back')?.closest('button')!
    expect(skipBackBtn).toBeTruthy()
    fireEvent.click(skipBackBtn!)

    expect(useTimelineStore.getState().playback.currentTime).toBe(0)
  })
})

describe('VideoEditor — timeline ruler', () => {
  it('renders time markers', () => {
    useTimelineStore.getState().addSegment({ duration: 5 })
    render(<VideoEditor />)

    expect(screen.getByText('0s')).toBeInTheDocument()
    expect(screen.getByText('5s')).toBeInTheDocument()
  })
})

describe('VideoEditor — generate buttons', () => {
  it('Generate All is disabled when no segments', () => {
    render(<VideoEditor />)

    const btn = screen.getByText(/generate all/i).closest('button')!
    expect(btn).toBeDisabled()
  })

  it('Generate All is enabled when segments exist with images', () => {
    useTimelineStore.getState().addSegment({ firstFrameB64: 'data:image/png;base64,test' })
    render(<VideoEditor />)

    const btn = screen.getByText(/generate all/i).closest('button')!
    expect(btn).not.toBeDisabled()
  })
})

describe('VideoEditor — drag interaction', () => {
  it('segment has move cursor (grab) on body', () => {
    useTimelineStore.getState().addSegment({ duration: 5 })
    render(<VideoEditor />)

    const dragBody = document.querySelector('[class*="cursor-grab"]')
    expect(dragBody).toBeTruthy()
  })

  it('segment has resize handles with ew-resize cursor', () => {
    useTimelineStore.getState().addSegment({ duration: 5 })
    render(<VideoEditor />)

    const resizeHandles = document.querySelectorAll('[class*="cursor-ew-resize"]')
    // Should have left and right handles
    expect(resizeHandles.length).toBeGreaterThanOrEqual(2)
  })

  it('pointer drag on segment body updates segment start', () => {
    const seg = useTimelineStore.getState().addSegment({ start: 0, duration: 5 })
    render(<VideoEditor />)

    const dragBody = document.querySelector('[class*="cursor-grab"]')! as HTMLElement

    // Simulate pointer down, move, up
    fireEvent.pointerDown(dragBody, { clientX: 100, clientY: 0, button: 0, pointerId: 1 })
    fireEvent.pointerMove(window, { clientX: 200, clientY: 0 })
    fireEvent.pointerUp(window)

    const updated = useTimelineStore.getState().segments[0]
    // Should have moved (100px / 80pps = 1.25 seconds)
    expect(updated.start).toBeGreaterThan(0)
  })

  it('pointer drag on right handle updates segment duration', () => {
    const seg = useTimelineStore.getState().addSegment({ start: 0, duration: 5 })
    render(<VideoEditor />)

    const handles = document.querySelectorAll('[class*="cursor-ew-resize"]')
    const rightHandle = handles[1] as HTMLElement // right handle

    const origDuration = useTimelineStore.getState().segments[0].duration

    fireEvent.pointerDown(rightHandle, { clientX: 400, clientY: 0, button: 0, pointerId: 1 })
    fireEvent.pointerMove(window, { clientX: 480, clientY: 0 }) // 80px = 1 second more
    fireEvent.pointerUp(window)

    const updated = useTimelineStore.getState().segments[0]
    expect(updated.duration).toBeGreaterThan(origDuration)
  })

  it('pointer drag on left handle updates start and duration', () => {
    const seg = useTimelineStore.getState().addSegment({ start: 2, duration: 5 })
    render(<VideoEditor />)

    const handles = document.querySelectorAll('[class*="cursor-ew-resize"]')
    const leftHandle = handles[0] as HTMLElement // left handle

    fireEvent.pointerDown(leftHandle, { clientX: 160, clientY: 0, button: 0, pointerId: 1 }) // 2s * 80pps = 160
    fireEvent.pointerMove(window, { clientX: 240, clientY: 0 }) // 80px right = 1s more start
    fireEvent.pointerUp(window)

    const updated = useTimelineStore.getState().segments[0]
    expect(updated.start).toBeGreaterThan(2) // start moved right
    expect(updated.duration).toBeLessThan(5) // duration shrank
  })
})

describe('VideoEditor — zoom', () => {
  it('zoom in increases pixels per second', () => {
    render(<VideoEditor />)

    const zoomInBtn = document.querySelector('.lucide-zoom-in')?.closest('button')!
    fireEvent.click(zoomInBtn)

    // Verify zoom label changed (should show > 100%)
    expect(screen.getByText('125%')).toBeInTheDocument()
  })

  it('zoom out decreases pixels per second', () => {
    render(<VideoEditor />)

    const zoomOutBtn = document.querySelector('.lucide-zoom-out')?.closest('button')!
    fireEvent.click(zoomOutBtn)

    expect(screen.getByText('80%')).toBeInTheDocument()
  })
})
