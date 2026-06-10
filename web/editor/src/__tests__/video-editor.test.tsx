import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
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
    expect(screen.getByText(/drop images from the asset sidebar/i)).toBeInTheDocument()
  })

  it('renders play/pause button', () => {
    render(<VideoEditor />)
    const playBtn = screen.getByRole('button', { name: '' }) // Play button has no aria-label, find by icon
    // There are multiple buttons — find the round one
    const buttons = screen.getAllByRole('button')
    expect(buttons.length).toBeGreaterThan(0)
  })

  it('renders Add Keyframe button', () => {
    render(<VideoEditor />)
    expect(screen.getByText(/add keyframe/i)).toBeInTheDocument()
  })

  it('renders Export button', () => {
    render(<VideoEditor />)
    expect(screen.getByText(/export/i)).toBeInTheDocument()
  })

  it('renders timeline track labels', () => {
    render(<VideoEditor />)
    expect(screen.getByText('Video')).toBeInTheDocument()
    expect(screen.getByText('Voice')).toBeInTheDocument()
    expect(screen.getByText('SFX')).toBeInTheDocument()
    expect(screen.getByText('Music')).toBeInTheDocument()
  })
})

describe('VideoEditor — adding segments', () => {
  it('clicking Add Keyframe creates a new segment', () => {
    render(<VideoEditor />)
    const addBtn = screen.getByText(/add keyframe/i)
    fireEvent.click(addBtn)

    const { segments } = useTimelineStore.getState()
    expect(segments).toHaveLength(1)
    expect(segments[0].status).toBe('empty')
  })

  it('clicking Add Keyframe twice creates two segments', () => {
    render(<VideoEditor />)
    const addBtn = screen.getByText(/add keyframe/i)
    fireEvent.click(addBtn)
    fireEvent.click(addBtn)

    const { segments } = useTimelineStore.getState()
    expect(segments).toHaveLength(2)
    expect(segments[1].start).toBe(5) // after first 5-second segment
  })
})

describe('VideoEditor — segment selection & inspector', () => {
  it('clicking a segment shows the inspector panel', () => {
    const seg = useTimelineStore.getState().addSegment({ prompt: 'test prompt' })
    render(<VideoEditor />)

    // Find the segment element by its keyframe label
    const segEl = screen.getByText(`K_${String(seg.order + 1).padStart(2, '0')}`)
    fireEvent.click(segEl.closest('[class*="rounded"]')!)

    // Inspector should show
    expect(screen.getByText('Prompt')).toBeInTheDocument()
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

    // Trash icon button
    const deleteButtons = screen.getAllByRole('button')
    const deleteBtn = deleteButtons.find(b => b.querySelector('[class*="trash"]') || b.querySelector('svg.lucide-trash-2'))
    expect(deleteBtn).toBeTruthy()
  })
})

describe('VideoEditor — playback', () => {
  it('skip back button resets time to 0', () => {
    useTimelineStore.getState().addSegment()
    useTimelineStore.getState().setPlayback({ currentTime: 3.5 })
    render(<VideoEditor />)

    // Find SkipBack button (first icon button)
    const buttons = screen.getAllByRole('button')
    const skipBackBtn = buttons[0] // SkipBack is the first button
    fireEvent.click(skipBackBtn)

    expect(useTimelineStore.getState().playback.currentTime).toBe(0)
  })
})

describe('VideoEditor — timeline ruler', () => {
  it('renders time markers for the total duration', () => {
    useTimelineStore.getState().addSegment({ duration: 5 })
    render(<VideoEditor />)

    // Should render markers 0s through 5s
    expect(screen.getByText('0s')).toBeInTheDocument()
    expect(screen.getByText('5s')).toBeInTheDocument()
  })
})

describe('VideoEditor — I2V generate', () => {
  it('generate button is disabled when no segments', () => {
    render(<VideoEditor />)

    // Find I2V button
    const i2vBtn = screen.getByText(/i2v/i).closest('button')!
    expect(i2vBtn).toBeDisabled()
  })

  it('generate button is enabled when segments exist', () => {
    useTimelineStore.getState().addSegment({ firstFrameB64: 'data:image/png;base64,test' })
    render(<VideoEditor />)

    const i2vBtn = screen.getByText(/i2v/i).closest('button')!
    expect(i2vBtn).not.toBeDisabled()
  })
})
