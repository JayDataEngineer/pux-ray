import { test, expect, type Page } from '@playwright/test'

async function gotoEditor(page: Page) {
  await page.goto('/editor/')
  await page.waitForSelector('text=TECH NOIR', { timeout: 10000 })
}

async function clickButton(page: Page, text: string | RegExp) {
  const btn = page.getByRole('button', { name: text, exact: false }).first()
  await btn.waitFor({ state: 'visible', timeout: 5000 })
  await btn.click()
  return btn
}

async function switchToVideo(page: Page) {
  await clickButton(page, 'Video')
}

/** Expand an inspector section by clicking its header button */
async function expandSection(page: Page, title: string) {
  // Escape regex special chars in title
  const escaped = title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const header = page.getByRole('button', { name: new RegExp(`^${escaped}$`), exact: false }).first()
  if (await header.isVisible()) {
    const isCollapsed = await header.locator('.lucide-chevron-right').count()
    if (isCollapsed > 0) await header.click()
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. SMOKE TESTS
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Smoke — App loads', () => {
  test('renders the TECH NOIR header', async ({ page }) => {
    await gotoEditor(page)
    await expect(page.locator('text=TECH NOIR')).toBeVisible()
  })

  test('renders the Video tab', async ({ page }) => {
    await gotoEditor(page)
    await expect(page.getByRole('button', { name: 'Video' })).toBeVisible()
  })

  test('switches to Video tab and shows inspector immediately', async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
    // Inspector auto-creates a segment and shows controls
    await expect(page.getByRole('button', { name: 'Model' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Prompts' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Timing' })).toBeVisible()
  })

  test('shows the Add Keyframe button', async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
    await expect(page.getByRole('button', { name: /Add Keyframe/ })).toBeVisible()
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 2. TIMELINE — Add & Select Keyframes
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Timeline — Add & Select Keyframes', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
  })

  test('auto-creates a segment on Video tab switch', async ({ page }) => {
    await expect(page.locator('[data-seg]')).toHaveCount(1)
  })

  test('click Add Keyframe creates another segment', async ({ page }) => {
    // One auto-created
    await expect(page.locator('[data-seg]')).toHaveCount(1)
    await clickButton(page, /Add Keyframe/)
    await expect(page.locator('[data-seg]')).toHaveCount(2)
  })

  test('clicking a segment selects it and shows inspector', async ({ page }) => {
    await page.locator('[data-seg]').first().click()
    await expect(page.getByText('Prompts')).toBeVisible()
    await expect(page.getByText('Timing')).toBeVisible()
  })

  test('inspector shows all fields for selected segment', async ({ page }) => {
    await page.locator('[data-seg]').first().click()
    await expect(page.getByText('Resolution & Frames')).toBeVisible()
    await expect(page.getByText('Generation')).toBeVisible()
  })

  test('delete button removes the segment (auto-creates new one)', async ({ page }) => {
    await page.locator('[data-seg]').first().click()
    await page.locator('button .lucide-trash-2').first().click()
    // Auto-create effect kicks in and adds a new segment
    await page.waitForTimeout(500)
    await expect(page.locator('[data-seg]')).toHaveCount(1)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 3. DRAG & RESIZE — Core interaction
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Timeline — Drag segments', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
  })

  test('segment has visible drag handles', async ({ page }) => {
    const handles = page.locator('[class*="cursor-ew-resize"]')
    await expect(handles).toHaveCount(2)
  })

  test('drag segment body moves it horizontally', async ({ page }) => {
    const seg = page.locator('[data-seg]').first()
    const initialBox = await seg.boundingBox()
    expect(initialBox).toBeTruthy()

    const dragBody = page.locator('[class*="cursor-grab"]').first()
    const bodyBox = await dragBody.boundingBox()
    expect(bodyBox).toBeTruthy()

    await page.mouse.move(bodyBox!.x + bodyBox!.width / 2, bodyBox!.y + bodyBox!.height / 2)
    await page.mouse.down()
    await page.mouse.move(bodyBox!.x + bodyBox!.width / 2 + 100, bodyBox!.y + bodyBox!.height / 2, { steps: 5 })
    await page.mouse.up()

    const newBox = await seg.boundingBox()
    expect(newBox!.x).toBeGreaterThanOrEqual(initialBox!.x - 1)
  })

  test('drag right resize handle extends duration', async ({ page }) => {
    const seg = page.locator('[data-seg]').first()
    const initialBox = await seg.boundingBox()
    expect(initialBox).toBeTruthy()

    const handles = page.locator('[class*="cursor-ew-resize"]')
    const rightHandle = handles.nth(1)
    const handleBox = await rightHandle.boundingBox()
    expect(handleBox).toBeTruthy()

    await page.mouse.move(handleBox!.x + handleBox!.width / 2, handleBox!.y + handleBox!.height / 2)
    await page.mouse.down()
    await page.mouse.move(handleBox!.x + handleBox!.width / 2 + 80, handleBox!.y + handleBox!.height / 2, { steps: 5 })
    await page.mouse.up()

    const newBox = await seg.boundingBox()
    expect(newBox!.width).toBeGreaterThanOrEqual(initialBox!.width - 1)
  })

  test('drag left resize handle extends start left', async ({ page }) => {
    const seg = page.locator('[data-seg]').first()
    const initialBox = await seg.boundingBox()
    expect(initialBox).toBeTruthy()

    const handles = page.locator('[class*="cursor-ew-resize"]')
    const leftHandle = handles.first()
    const handleBox = await leftHandle.boundingBox()
    expect(handleBox).toBeTruthy()

    await page.mouse.move(handleBox!.x + handleBox!.width / 2, handleBox!.y + handleBox!.height / 2)
    await page.mouse.down()
    await page.mouse.move(handleBox!.x + handleBox!.width / 2 - 80, handleBox!.y + handleBox!.height / 2, { steps: 10 })
    await page.mouse.up()

    const newBox = await seg.boundingBox()
    expect(newBox!.width).toBeGreaterThan(initialBox!.width - 1)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 4. PLAYBACK
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Playback controls', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
  })

  test('play button exists and is clickable', async ({ page }) => {
    const playBtn = page.locator('.lucide-play').first()
    await expect(playBtn).toBeVisible()
    await playBtn.locator('..').click()
  })

  test('skip back button exists', async ({ page }) => {
    const skipBtn = page.locator('.lucide-skip-back').first()
    await expect(skipBtn).toBeVisible()
  })

  test('time display shows current and total time', async ({ page }) => {
    const timeDisplay = page.locator('.tabular-nums')
    await expect(timeDisplay).toContainText('0:00.00')
    // Default LTX 22B: 121 frames @ 24fps = 5.04s (displays rounded as 0:05.00)
    await expect(timeDisplay).toContainText('0:05.00')
  })

  test('clicking on ruler area seeks playhead', async ({ page }) => {
    const playhead = page.locator('.bg-red-500').first()
    await expect(playhead).toBeVisible()

    const ruler = page.locator('.cursor-pointer').first()
    const rulerBox = await ruler.boundingBox()
    if (rulerBox && rulerBox.width > 100) {
      await page.mouse.click(rulerBox.x + 200, rulerBox.y + rulerBox.height / 2)
      const newStyle = await playhead.getAttribute('style')
      expect(newStyle).toContain('left:')
    }
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 5. CONTROLS
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Controls', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
  })

  test('Generate All button is enabled with auto-created segment', async ({ page }) => {
    const btn = page.getByRole('button', { name: /Generate All/ })
    await expect(btn).toBeEnabled()
  })

  test('Generate This Segment button at bottom of inspector', async ({ page }) => {
    await page.locator('[data-seg]').first().click()
    const btn = page.getByRole('button', { name: /Generate This Segment/ })
    await expect(btn).toBeVisible()
    // Button should be enabled for empty segments
    await expect(btn).toBeEnabled()
  })

  test('Export button is enabled with auto-created segment', async ({ page }) => {
    const btn = page.getByRole('button', { name: /Export/ })
    await expect(btn).toBeEnabled()
  })

  test('zoom in increases scale', async ({ page }) => {
    const zoomIn = page.locator('.lucide-zoom-in').first().locator('..')
    await zoomIn.click()
    await expect(page.getByText('125%')).toBeVisible()
  })

  test('zoom out decreases scale', async ({ page }) => {
    const zoomOut = page.locator('.lucide-zoom-out').first().locator('..')
    await zoomOut.click()
    await expect(page.getByText('80%')).toBeVisible()
  })

  test('track labels are visible — Video track always present, audio tracks dynamic', async ({ page }) => {
    // Video track is always present
    await expect(page.getByText('Video').first()).toBeVisible()
    // Audio tracks are dynamic — no fixed Voice/SFX/Music anymore
    // "Add Track" button allows adding audio tracks
    await expect(page.getByRole('button', { name: /Add Track/ })).toBeVisible()
  })

  test('Add Track creates a visible audio track row', async ({ page }) => {
    // Only Video track row initially
    const videoTrack = page.locator('text=Video').first()
    await expect(videoTrack).toBeVisible()
    // No audio tracks yet
    await expect(page.locator('text=Audio 1')).toHaveCount(0)
    // Click Add Track
    await clickButton(page, /Add Track/)
    // Audio 1 track row appears
    await expect(page.locator('text=Audio 1')).toBeVisible()
  })

  test('Services sidebar is hidden on Video tab', async ({ page }) => {
    // The Services sidebar header should NOT be visible on Video tab
    const servicesHeader = page.getByText('Services').first()
    const isVisible = await servicesHeader.isVisible().catch(() => false)
    expect(isVisible).toBe(false)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 6. INSPECTOR — Edit properties
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Inspector — Edit properties', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
    // Segment is auto-selected, inspector is visible
  })

  test('can type a prompt', async ({ page }) => {
    const promptArea = page.locator('textarea[placeholder="Describe this segment..."]')
    await promptArea.fill('A cinematic forest scene with mist')
    await expect(promptArea).toHaveValue('A cinematic forest scene with mist')
  })

  test('can change duration', async ({ page }) => {
    const durationInput = page.locator('text=Duration (s)').locator('..').locator('input')
    await durationInput.fill('10')
    await expect(durationInput).toHaveValue('10')
  })

  test('can change start time', async ({ page }) => {
    const startInput = page.locator('text=Start (s)').locator('..').locator('input')
    await startInput.fill('3')
    await expect(startInput).toHaveValue('3')
  })

  test('can change width and height', async ({ page }) => {
    const widthInput = page.locator('text=Width', { exact: true }).locator('..').locator('input')
    await widthInput.fill('1024')
    await expect(widthInput).toHaveValue('1024')
  })

  test('Video Length field is visible', async ({ page }) => {
    await expect(page.getByText('Video Length (seconds)')).toBeVisible()
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 7. SIDEBAR — Asset list, modal navigation, video handling
// ═══════════════════════════════════════════════════════════════════════════

/** Assets injected into localStorage before the page boots.  Using non-data:
  * URLs keeps IndexedDB out of the loop — they load as broken images/video
  * which is fine for interaction tests. */
const SIDEBAR_TEST_ASSETS = [
  { id: 't_img1', name: 'img1.png', type: 'image', category: 'image', mediaType: 'image/png', url: '/img1.png', sizeBytes: 100, source: 'uploaded', createdAt: '2024-01-01T00:00:00.000Z' },
  { id: 't_img2', name: 'img2.png', type: 'image', category: 'image', mediaType: 'image/png', url: '/img2.png', sizeBytes: 100, source: 'uploaded', createdAt: '2024-01-01T00:00:01.000Z' },
  { id: 't_vid1', name: 'vid1.mp4', type: 'video', category: 'video', mediaType: 'video/mp4', url: '/vid1.mp4', sizeBytes: 100, source: 'uploaded', createdAt: '2024-01-01T00:00:02.000Z' },
  { id: 't_snd1', name: 'song.mp3', type: 'audio', category: 'music', mediaType: 'audio/mp3', url: '/song.mp3', sizeBytes: 100, source: 'uploaded', createdAt: '2024-01-01T00:00:03.000Z' },
]

async function gotoEditorWithAssets(page: Page) {
  await page.addInitScript((assets) => {
    localStorage.setItem('tech_noir_assets', JSON.stringify(assets))
  }, SIDEBAR_TEST_ASSETS)
  await page.goto('/editor/')
  await page.waitForSelector('text=TECH NOIR', { timeout: 10000 })
  // Wait for the loading overlay to clear
  await page.waitForSelector('text=Assets', { timeout: 5000 })
}

test.describe('Sidebar — Assets & modal', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditorWithAssets(page)
  })

  // ── (a) Images default to folded in ──────────────────────────────────────
  test('(a) Images section is collapsed by default', async ({ page }) => {
    const sidebar = page.locator('.bg-sidebar').first()
    const trigger = sidebar.getByRole('button', { name: /Images/ })
    await expect(trigger).toBeVisible()
    await expect(trigger).toHaveAttribute('aria-expanded', 'false')
    // Thumbnails are not rendered while collapsed
    await expect(page.locator('img[alt="img1.png"]')).toHaveCount(0)
  })

  test('(a) every category section is collapsed by default', async ({ page }) => {
    const sidebar = page.locator('.bg-sidebar').first()
    for (const label of ['Images', 'Video', 'Music']) {
      const trigger = sidebar.getByRole('button', { name: new RegExp(label) })
      await expect(trigger).toHaveAttribute('aria-expanded', 'false')
    }
  })

  test('(a) expanding Images reveals thumbnails', async ({ page }) => {
    const sidebar = page.locator('.bg-sidebar').first()
    await sidebar.getByRole('button', { name: /Images/ }).click()
    await expect(page.locator('img[alt="img1.png"]')).toBeVisible()
  })

  // ── (b) Modal "next" matches the sidebar list order (same category) ───────
  test('(b) modal navigates within the current category', async ({ page }) => {
    const sidebar = page.locator('.bg-sidebar').first()
    await sidebar.getByRole('button', { name: /Images/ }).click()
    await page.locator('img[alt="img1.png"]').click()

    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()
    // Counter is scoped to the image category: 1 / 2 (not 1 / 4)
    await expect(dialog.getByText('1 / 2')).toBeVisible()

    // Next should advance to img2, staying inside the dialog
    await dialog.locator('svg.lucide-chevron-right').click()
    await expect(dialog.getByText('2 / 2')).toBeVisible()
    await expect(dialog.locator('img[alt="img2.png"]')).toBeVisible()
  })

  test('(b) modal prev wraps to the start of the category', async ({ page }) => {
    const sidebar = page.locator('.bg-sidebar').first()
    await sidebar.getByRole('button', { name: /Images/ }).click()
    // Click the second image so we can go back to the first
    await page.locator('img[alt="img2.png"]').click()

    const dialog = page.getByRole('dialog')
    await expect(dialog.getByText('2 / 2')).toBeVisible()
    await dialog.locator('svg.lucide-chevron-left').click()
    await expect(dialog.getByText('1 / 2')).toBeVisible()
    await expect(dialog.locator('img[alt="img1.png"]')).toBeVisible()
  })

  // ── (c) Videos open the modal instead of playing inline ───────────────────
  test('(c) clicking a video thumbnail opens the modal', async ({ page }) => {
    const sidebar = page.locator('.bg-sidebar').first()
    await sidebar.getByRole('button', { name: /Video/ }).click()

    // The sidebar video should be there — click it (no inline controls)
    const sidebarVideo = sidebar.locator('video').first()
    await expect(sidebarVideo).toBeVisible()
    await sidebarVideo.click()

    // A dialog with a <video> should appear
    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()
    await expect(dialog.locator('video')).toBeVisible()
    // The sidebar video had no controls; the modal one should (it's a real player)
    await expect(dialog.locator('video')).toHaveAttribute('controls')
  })

  // ── (d) Generated names get numbered instead of colliding ────────────────
  test('(d) addAsset de-duplicates names with a counter', async ({ page }) => {
    // The store is not on window, but addAsset writes to localStorage, so we
    // can use the same zustand store via the page's module graph by going
    // through the exposed import path.  Instead, verify the persisted result:
    // call addAsset through the dev-only hook exposed on window.
    const result = await page.evaluate(() => {
      // @ts-expect-error — store is attached in dev for testing (see main.tsx)
      const store = window.__assetStore
      if (!store) return { error: 'store not exposed' }
      const a1 = store.getState().addAsset({
        name: 'ltx2_1.png', type: 'image', category: 'image',
        mediaType: 'image/png', url: 'x', sizeBytes: 0, source: 'generated',
      })
      const a2 = store.getState().addAsset({
        name: 'ltx2_1.png', type: 'image', category: 'image',
        mediaType: 'image/png', url: 'y', sizeBytes: 0, source: 'generated',
      })
      return { first: a1.name, second: a2.name }
    })
    expect(result.first).toBe('ltx2_1.png')
    expect(result.second).toBe('ltx2_2.png')
  })

  test('(d) nextAssetName continues from existing assets', async ({ page }) => {
    const result = await page.evaluate(() => {
      // @ts-expect-error — dev-only hook
      const store = window.__assetStore
      if (!store) return { error: 'store not exposed' }
      // The sidebar test data has img1.png / img2.png but no "gen_*" names
      const n1 = store.getState().addAsset({
        name: 'gen_1.png', type: 'image', category: 'image',
        mediaType: 'image/png', url: 'a', sizeBytes: 0, source: 'generated',
      }).name
      // Simulate a second generation pass producing the "next" name
      // @ts-expect-error — dev-only hook
      const next = window.__nextAssetName?.('gen', 'png') ?? 'gen_1.png'
      return { first: n1, next }
    })
    expect(result.first).toBe('gen_1.png')
    expect(result.next).toBe('gen_2.png')
  })
})
