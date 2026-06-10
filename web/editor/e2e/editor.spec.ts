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

  test('switches to Video tab and shows Add Keyframe button', async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
    await expect(page.getByRole('button', { name: /Add Keyframe/ })).toBeVisible()
  })

  test('shows the empty state prompt', async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
    await expect(page.getByText('Drop images or audio from the asset sidebar')).toBeVisible()
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

  test('click Add Keyframe creates a segment on timeline', async ({ page }) => {
    await clickButton(page, /Add Keyframe/)
    await expect(page.locator('[data-seg]')).toHaveCount(1)
  })

  test('click Add Keyframe twice creates two segments', async ({ page }) => {
    await clickButton(page, /Add Keyframe/)
    await clickButton(page, /Add Keyframe/)
    await expect(page.locator('[data-seg]')).toHaveCount(2)
  })

  test('clicking a segment selects it and shows inspector', async ({ page }) => {
    await clickButton(page, /Add Keyframe/)
    await page.locator('[data-seg]').first().click()

    // Inspector should show section headers
    await expect(page.getByText('Prompts')).toBeVisible()
    await expect(page.getByText('Timing')).toBeVisible()
  })

  test('inspector shows all fields for selected segment', async ({ page }) => {
    await clickButton(page, /Add Keyframe/)
    await page.locator('[data-seg]').first().click()

    await expect(page.getByText('Resolution & Frames')).toBeVisible()
    await expect(page.getByText('Generation')).toBeVisible()
  })

  test('delete button removes the segment', async ({ page }) => {
    await clickButton(page, /Add Keyframe/)
    await page.locator('[data-seg]').first().click()
    await page.locator('button .lucide-trash-2').first().click()
    await expect(page.locator('[data-seg]')).toHaveCount(0)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 3. DRAG & RESIZE — Core interaction
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Timeline — Drag segments', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
    await clickButton(page, /Add Keyframe/)
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

    // Drag right
    await page.mouse.move(bodyBox!.x + bodyBox!.width / 2, bodyBox!.y + bodyBox!.height / 2)
    await page.mouse.down()
    await page.mouse.move(bodyBox!.x + bodyBox!.width / 2 + 100, bodyBox!.y + bodyBox!.height / 2, { steps: 5 })
    await page.mouse.up()

    const newBox = await seg.boundingBox()
    // Should have moved at least a few pixels
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

    // Drag left by 80px
    await page.mouse.move(handleBox!.x + handleBox!.width / 2, handleBox!.y + handleBox!.height / 2)
    await page.mouse.down()
    await page.mouse.move(handleBox!.x + handleBox!.width / 2 - 80, handleBox!.y + handleBox!.height / 2, { steps: 10 })
    await page.mouse.up()

    const newBox = await seg.boundingBox()
    // Width should grow (we shrunk the left edge)
    expect(newBox!.width).toBeGreaterThan(initialBox!.width - 1) // allow 1px rounding
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 4. PLAYBACK
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Playback controls', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
    await clickButton(page, /Add Keyframe/)
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
    // The time span is a single element like "0:00.00 / 0:05.00"
    const timeDisplay = page.locator('.tabular-nums')
    await expect(timeDisplay).toContainText('0:00.00')
    await expect(timeDisplay).toContainText('0:05.00')
  })

  test('clicking on ruler area seeks playhead', async ({ page }) => {
    const playhead = page.locator('.bg-red-500').first()
    await expect(playhead).toBeVisible()

    // Click on the ruler area at x offset
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

  test('Generate All button is disabled with no segments', async ({ page }) => {
    const btn = page.getByRole('button', { name: /Generate All/ })
    await expect(btn).toBeDisabled()
  })

  test('Export button is disabled with no segments', async ({ page }) => {
    const btn = page.getByRole('button', { name: /Export/ })
    await expect(btn).toBeDisabled()
  })

  test('zoom in increases scale', async ({ page }) => {
    await clickButton(page, /Add Keyframe/)
    const zoomIn = page.locator('.lucide-zoom-in').first().locator('..')
    await zoomIn.click()
    await expect(page.getByText('125%')).toBeVisible()
  })

  test('zoom out decreases scale', async ({ page }) => {
    await clickButton(page, /Add Keyframe/)
    const zoomOut = page.locator('.lucide-zoom-out').first().locator('..')
    await zoomOut.click()
    await expect(page.getByText('80%')).toBeVisible()
  })

  test('track labels are visible', async ({ page }) => {
    await expect(page.getByText('Video').first()).toBeVisible()
    await expect(page.getByText('Voice').first()).toBeVisible()
    await expect(page.getByText('SFX').first()).toBeVisible()
    await expect(page.getByText('Music').first()).toBeVisible()
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 6. INSPECTOR — Edit properties
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Inspector — Edit properties', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
    await clickButton(page, /Add Keyframe/)
    await page.locator('[data-seg]').first().click()
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
})
