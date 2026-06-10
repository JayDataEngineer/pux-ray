# Video Editor Frontend Audit

**Date:** 2026-06-10
**Scope:** `web/editor/src/` — React + Vite + Zustand + Tailwind

---

## Summary

The editor is **functional but demo-grade**. It renders, it has real state management, real MCP backend wiring, and real generation flows. However it has significant gaps in interactivity, dead code, missing features, and zero test coverage (before this audit).

**Verdict:** 60% real, 40% demo shell.

---

## 1. Critical Bugs

### 1a. Export button does nothing
`VideoEditor.tsx:140` — the Export button has **no onClick handler**. It renders `<Settings />` icon and "Export" text but clicking it does nothing.

### 1b. Audio playback hook is dead code
`hooks/useAudioPlayback.ts` — a fully implemented audio playback system with Web Audio API, waveform computation, play/pause/seek. **It is never imported or used anywhere.** The VideoEditor has its own inline playback loop that only animates the playhead position — no audio plays.

### 1c. Sidebar asset click doesn't add to timeline
`WorkspaceLayout.tsx:140` — `onSelectAsset` sets `selectedAsset` state (opens a preview dialog), but **there is no path from clicking an asset to adding it to the timeline**. Only drag-and-drop works. The sidebar `onSelectAsset` callback only opens a lightbox.

### 1d. Timeline has no drag-to-reorder
The timeline store has a `reorderSegments()` method and a `DragState` type, but **no drag handlers are wired** on the timeline segment elements. Segments cannot be reordered by dragging.

### 1e. Timeline has no resize handles
The store supports `updateSegment({ duration })` and segments show a width based on duration, but **there are no resize handles** on the timeline blocks. Duration can only be changed via the inspector panel number input.

### 1f. Audio tracks are read-only
Audio cues render on the timeline, but there are **no controls to add, move, resize, or adjust volume** of audio cues from the UI. The only way audio cues appear is via `loadFromRun()`. There's no "Add Audio" button.

### 1g. Timeline scrubbing doesn't work
Clicking the ruler/timeline area does **not** move the playhead. There's no click handler on the ruler or track lanes to seek. The playhead only moves during playback animation.

---

## 2. Dead Code / Unused Dependencies

### 2a. Three.js — 39MB installed, 0 imports
`three`, `@react-three/fiber`, `@react-three/drei` are in `package.json` but **never imported** in any source file. These alone account for ~68MB of `node_modules`.

### 2b. Remotion — 2.2MB installed, 0 imports
`remotion` and `@remotion/player` are in `package.json` but **never imported**. These are for programmatic video composition — a natural fit for the editor, but unused.

### 2c. wavesurfer.js — 1.4MB installed, 0 imports
`wavesurfer.js` is in `package.json` but **never imported**. The `AudioCue.waveformPeaks` field and the `useAudioPlayback` hook suggest it was planned for waveform rendering.

### 2d. assistant-ui — imported but not rendered
The `components/assistant-ui/` directory has Thread, ThreadList, MarkdownText, etc. — a full chat UI. These are **never rendered** in the main app flow (App → WorkspaceLayout → VideoEditor). They appear to be leftover scaffolding.

### 2e. `api.ts` is just re-exports
`src/api.ts` re-exports everything from `src/mcp.ts`. It's an indirection layer that serves no purpose — every consumer could import directly from `mcp.ts`.

### 2f. `workflow.ts` store has minimal usage
The `useWorkflowStore` is imported but only stores `spec`, `run`, `selectedStepId`, `viewMode`. The `VideoEditor` doesn't use it — it uses `useTimelineStore` directly. This store appears to be from an earlier architecture.

---

## 3. Demo Shell Features (Present but Non-functional)

| Feature | Status | Why |
|---------|--------|-----|
| Export button | Shell | No `onClick` handler |
| Audio playback | Dead code | `useAudioPlayback` never called |
| Audio track controls | Missing | No UI to add/move/resize audio |
| Timeline scrubbing | Missing | No click handler on ruler |
| Segment drag-reorder | Missing | Store has `reorderSegments`, UI doesn't use it |
| Segment resize | Missing | No drag handles on timeline blocks |
| Zoom/scale | Missing | `pixelsPerSecond` in store but no zoom control in UI |
| Waveform display | Missing | `waveformPeaks` computed in hook, never rendered |
| GPU status polling | Working | Polls `/status` every 15s — real |
| Image drop → segment | Working | `onDrop` creates segments — real |
| I2V generation | Working | Calls `callTool("run", ...)` via MCP — real |
| Asset import/upload | Working | FileReader → data URL → store — real |
| AI prompt enhancement | Working | OpenAI-compatible endpoint — real |

---

## 4. Architecture Issues

### 4a. PPS (pixels per second) is hardcoded
`VideoEditor.tsx:30` — `const PPS = 80` is a local constant. The store has `viewport.pixelsPerSecond = 60` but the component ignores it. Zoom is impossible without aligning these.

### 4b. Total duration computed inline
`VideoEditor.tsx:57` — `const total = Math.max(segments.reduce(...))` duplicates logic from the store. The store already computes `playback.totalDuration` in `addSegment` and `reorderSegments`. These can diverge.

### 4c. Generation blocks the UI
`VideoEditor.tsx:91-109` — `genAll()` is an async loop with `await callTool()` for each segment. During this, `generating` state disables the I2V button but the rest of the UI is still interactive. No progress indication per-segment.

### 4d. Base64 video URLs in memory
`VideoEditor.tsx:101` — Generated video is stored as `data:video/mp4;base64,...` in state. For a 5-second video at 24fps, this can be 5-50MB in memory. The preview `<video>` element loads the entire base64 string. Should use object URLs or server-side storage.

### 4e. No keyboard shortcuts
No keyboard event listeners for space (play/pause), arrow keys (scrub), delete (remove segment), etc.

### 4f. No undo/redo
No history tracking on the Zustand stores.

### 4g. Sidebar is duplicated for mobile
`AppSidebar.tsx` renders the entire sidebar **twice** — once for desktop (lines 62-158) and once for mobile overlay (lines 167-270). Any change must be made in both places.

---

## 5. Test Coverage

### Before this audit: 0 tests, 0 test files, no test framework installed.

### After this audit: 81 tests, 9 test files, vitest + testing-library installed.

**Test files created:**

| File | Tests | Covers |
|------|-------|--------|
| `__tests__/timeline-store.test.ts` | 22 | Segment CRUD, reorder, playback, audio cues, loadFromRun, reset |
| `__tests__/asset-store.test.ts` | 9 | CRUD, persistence, rename, filter, clear |
| `__tests__/toast-store.test.ts` | 5 | Add, auto-remove, remove, multiple |
| `__tests__/enhancement-store.test.ts` | 10 | Add, update, remove, active, persist |
| `__tests__/enhance-prompts.test.ts` | 20 | All service/model prompt mappings, fallback |
| `__tests__/timeline-types.test.ts` | 2 | Constants validation |
| `__tests__/mcp-client.test.ts` | 3 | Utility functions, error handling |
| `__tests__/video-editor.test.tsx` | 15 | Rendering, segments, inspector, playback, I2V |
| `__tests__/app.test.tsx` | 1 | Smoke test |

---

## 6. Priority Fix List

### P0 — Broken core flows
1. **Wire Export button** — Needs `onClick` that compiles timeline to a video (Remotion or ffmpeg.wasm) or at minimum downloads the concatenated segments
2. **Fix timeline scrubbing** — Add click handler on ruler and track lanes to set `playback.currentTime`
3. **Connect audio playback** — Import `useAudioPlayback` in VideoEditor, wire play/pause/seek to it

### P1 — Missing interactivity
4. **Add segment resize handles** — Left/right drag handles on timeline blocks, update `duration` on drag
5. **Add segment drag-reorder** — Horizontal drag on blocks to reorder via `reorderSegments()`
6. **Add audio controls** — "Add Audio" button, volume slider, drag to position
7. **Add zoom control** — Slider or scroll-wheel to adjust `pixelsPerSecond`

### P2 — Cleanup
8. **Remove dead dependencies** — `three`, `@react-three/*`, `remotion`, `@remotion/*`, `wavesurfer.js` (unless planned for immediate use)
9. **Remove dead components** — `assistant-ui/*` directory
10. **Deduplicate sidebar** — Extract shared render logic
11. **Remove `api.ts`** — Import from `mcp.ts` directly
12. **Fix PPS mismatch** — Use `viewport.pixelsPerSecond` from store, not hardcoded constant

### P3 — Robustness
13. **Replace base64 video storage** with object URLs or server artifacts
14. **Add keyboard shortcuts** (space, arrows, delete)
15. **Add undo/redo** (zustand middleware)
16. **Add error boundaries** per-panel (not just root)
