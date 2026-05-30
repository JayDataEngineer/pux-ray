import { TimelineCanvas } from './TimelineCanvas'
import { TimelineToolbar } from './TimelineToolbar'
import { SegmentDetailPanel } from './SegmentDetailPanel'
import { AudioTablePanel } from './AudioTablePanel'
import { useTimelineStore } from '../../stores/timeline'

export function TimelineEditor() {
  const selectedSegmentId = useTimelineStore((s) => s.selectedSegmentId)

  return (
    <div className="timeline-editor">
      <TimelineToolbar />
      <div className="timeline-editor-body">
        <div className="timeline-editor-main">
          <TimelineCanvas />
        </div>
        {selectedSegmentId && (
          <div className="timeline-detail-panel">
            <SegmentDetailPanel segmentId={selectedSegmentId} />
          </div>
        )}
      </div>
      <div className="timeline-audio-table-wrap">
        <AudioTablePanel />
      </div>
    </div>
  )
}
