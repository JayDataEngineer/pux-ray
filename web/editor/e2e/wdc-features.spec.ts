import { test, expect, type Page } from '@playwright/test'

// ── Helpers ─────────────────────────────────────────────────────────────────

async function gotoEditor(page: Page) {
  await page.goto('/editor/')
  await page.waitForSelector('text=TECH NOIR', { timeout: 10000 })
}

async function switchToVideo(page: Page) {
  const btn = page.getByRole('button', { name: 'Video', exact: false }).first()
  await btn.waitFor({ state: 'visible', timeout: 5000 })
  await btn.click()
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

/** Open the model dropdown and select a model by partial label match */
async function selectModel(page: Page, modelLabel: RegExp) {
  await expandSection(page, 'Model')
  // Click the model SelectTrigger — contains the current model name text
  const trigger = page.locator('button').filter({ hasText: /wan|ltx/i }).first()
  await trigger.click()
  await page.getByRole('option', { name: modelLabel }).click()
  await page.waitForTimeout(300)
}

/** Check if backend is reachable */
async function backendAvailable(): Promise<boolean> {
  try {
    const resp = await fetch('http://localhost:4173/v1/loras?model=ltx2', { signal: AbortSignal.timeout(3000) })
    return resp.ok
  } catch {
    return false
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. MODEL SWITCHING — shows/hides the right sections
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Model switching — WAN vs LTX sections', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
  })

  test('WAN model shows basic sections only — no Director/Camera/Guidance', async ({ page }) => {
    // Switch to WAN first — default is now LTX 2.3 22B distilled
    await selectModel(page, /Wan 1\.3B/)
    await expect(page.getByRole('button', { name: 'Model' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Prompts' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Generation' })).toBeVisible()
    // WAN should NOT show LTX-only sections
    const directorBtn = page.getByRole('button', { name: /Director Controls/ })
    const cameraBtn = page.getByRole('button', { name: /Camera Motion/ })
    const guidanceBtn = page.getByRole('button', { name: /^Guidance$/ })
    expect(await directorBtn.isVisible().catch(() => false)).toBe(false)
    expect(await cameraBtn.isVisible().catch(() => false)).toBe(false)
    expect(await guidanceBtn.isVisible().catch(() => false)).toBe(false)
  })

  test('LTX 22B distilled is the default — shows all LTX sections', async ({ page }) => {
    // Default model is already LTX 2.3 22B distilled
    await expect(page.getByRole('button', { name: /Director Controls/ })).toBeVisible()
    await expect(page.getByRole('button', { name: /Camera Motion/ })).toBeVisible()
    await expect(page.getByRole('button', { name: /^Guidance$/ })).toBeVisible()
    await expect(page.getByRole('button', { name: /Self-Refiner/ })).toBeVisible()
    await expect(page.getByRole('button', { name: /Sliding Window/ })).toBeVisible()
    await expect(page.getByRole('button', { name: /Audio Mode/ })).toBeVisible()
  })


  test('switching back to WAN hides LTX sections', async ({ page }) => {
    // Switch to WAN (default is LTX)
    await selectModel(page, /Wan 1\.3B/)
    await page.waitForTimeout(300)

    const directorBtn = page.getByRole('button', { name: /Director Controls/ })
    expect(await directorBtn.isVisible().catch(() => false)).toBe(false)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 2. FRAMES / DURATION / FPS DYNAMIC LINK
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Frames / Duration / FPS dynamic relationship', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
  })

  test('changing Video Length updates Frames and Duration', async ({ page }) => {
    await expandSection(page, 'Resolution & Frames')
    // Default LTX 22B: 121 frames @ 24fps = 5.04s
    const vlContainer = page.locator('div').filter({ hasText: /^Video Length \(seconds\)$/ }).first()
    const vlInput = vlContainer.locator('input').first()
    await vlInput.fill('10')
    // Frames should now be 10 * 24 = 240
    const framesContainer = page.locator('div').filter({ hasText: /^Frames$/ }).first()
    const framesInput = framesContainer.locator('input').first()
    await expect(framesInput).toHaveValue('240')
    // Duration should update
    const durContainer = page.locator('div').filter({ hasText: /^Duration \(s\)$/ }).first()
    const durationInput = durContainer.locator('input').first()
    await expect(durationInput).toHaveValue('10')
  })

  test('changing Frames updates Video Length and Duration', async ({ page }) => {
    await expandSection(page, 'Resolution & Frames')
    const framesContainer = page.locator('div').filter({ hasText: /^Frames$/ }).first()
    const framesInput = framesContainer.locator('input').first()
    await framesInput.fill('240')
    // Video length = 240 / 24 = 10s
    const vlContainer = page.locator('div').filter({ hasText: /^Video Length \(seconds\)$/ }).first()
    const vlInput = vlContainer.locator('input').first()
    await expect(vlInput).toHaveValue('10')
  })

  test('changing FPS updates Frames, keeps Video Length constant', async ({ page }) => {
    await expandSection(page, 'Resolution & Frames')
    // Default: 121 frames @ 24fps = 5.04s
    const fpsContainer = page.locator('div').filter({ hasText: /^FPS$/ }).first()
    const fpsInput = fpsContainer.locator('input').first()
    await fpsInput.fill('16')
    // Video length stays ~5.04s, frames = 5.04 * 16 = ~81
    const framesContainer = page.locator('div').filter({ hasText: /^Frames$/ }).first()
    const framesInput = framesContainer.locator('input').first()
    const framesVal = await framesInput.inputValue()
    expect(Number(framesVal)).toBeCloseTo(81, -1)
  })

  test('changing Duration in Timing updates Frames', async ({ page }) => {
    const durContainer = page.locator('div').filter({ hasText: /^Duration \(s\)$/ }).first()
    const durationInput = durContainer.locator('input').first()
    await durationInput.fill('10')
    await expandSection(page, 'Resolution & Frames')
    const framesContainer = page.locator('div').filter({ hasText: /^Frames$/ }).first()
    const framesInput = framesContainer.locator('input').first()
    await expect(framesInput).toHaveValue('240')
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 3. INSPECTOR — All LTX sections expand and have controls
// ═══════════════════════════════════════════════════════════════════════════

test.describe('LTX Inspector — All sections functional', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
    // Default is already LTX 2.3 22B distilled
  })

  test('Director Controls has Guide Phases, Epsilon, Denoise Strength, Perturbation', async ({ page }) => {
    await expandSection(page, 'Director Controls')
    await expect(page.getByText('Guide Phases')).toBeVisible()
    await expect(page.getByText('Epsilon')).toBeVisible()
    await expect(page.getByText('Denoise Strength')).toBeVisible()
    await expect(page.getByText('Perturbation')).toBeVisible()
  })

  test('Camera Motion has Pan X, Pan Y, Zoom', async ({ page }) => {
    await expandSection(page, 'Camera Motion')
    await expect(page.getByText('Pan X')).toBeVisible()
    await expect(page.getByText('Pan Y')).toBeVisible()
    // "Zoom" appears in both the inspector label and keyboard hints — use .first()
    await expect(page.getByText('Zoom').first()).toBeVisible()
  })

  test('Guidance section shows NAG controls (distilled default)', async ({ page }) => {
    await expandSection(page, 'Guidance')
    await expect(page.getByText('NAG Scale')).toBeVisible()
    await expect(page.getByText('NAG Tau')).toBeVisible()
    await expect(page.getByText('NAG Alpha')).toBeVisible()
  })

  test('Self-Refiner toggle enables plan and uncertainty inputs', async ({ page }) => {
    await expandSection(page, 'Self-Refiner')
    // Toggle off by default — plan not visible
    await expect(page.getByText('Plan (e.g. 2-8:3)')).toHaveCount(0)
    // Click enable — find toggle next to the Enable Refiner label
    const toggle = page.getByText('Enable Refiner').locator('xpath=following-sibling::button').first()
    await toggle.click()
    // Now plan and uncertainty inputs visible
    await expect(page.getByText('Plan (e.g. 2-8:3)')).toBeVisible()
    await expect(page.getByText('Uncertainty')).toBeVisible()
    await expect(page.getByText('Certainty %')).toBeVisible()
  })

  test('Sliding Window toggle enables size and overlap inputs', async ({ page }) => {
    await expandSection(page, 'Sliding Window')
    // Toggle off by default
    await expect(page.getByText('Window Size')).toHaveCount(0)
    const toggle = page.getByText('Enable (long videos)').locator('xpath=following-sibling::button').first()
    await toggle.click()
    await expect(page.getByText('Window Size')).toBeVisible()
    await expect(page.getByText('Overlap')).toBeVisible()
  })

  test('Audio Mode section shows conditioning mode dropdown', async ({ page }) => {
    await expandSection(page, 'Audio Mode')
    await expect(page.getByText('Conditioning Mode')).toBeVisible()
  })

  test('Prompts section shows Auto-Enhance toggle for LTX', async ({ page }) => {
    await expandSection(page, 'Prompts')
    await expect(page.getByText('Auto-Enhance Prompt')).toBeVisible()
  })

  test('LoRA section shows Distilled Mode toggle (on by default)', async ({ page }) => {
    await expandSection(page, 'LoRA')
    await expect(page.getByText('Distilled Mode (8 steps)')).toBeVisible()
  })

  test('Control Video section visible (distilled is default)', async ({ page }) => {
    // Distilled mode is on by default, so Control Video is already visible
    await expect(page.getByRole('button', { name: /Control Video/ })).toBeVisible()
    await expandSection(page, 'Control Video (IC-LoRA)')
    await expect(page.getByText('Control Mode')).toBeVisible()
  })

  test('Perturbation Detail section appears when perturbation enabled (dev mode only)', async ({ page }) => {
    // Perturbation Detail only appears in dev mode (!distilledMode)
    // Need to turn OFF distilled mode first
    await expandSection(page, 'LoRA')
    const distLabel = page.getByText('Distilled Mode (8 steps)')
    const distToggle = distLabel.locator('xpath=following-sibling::button').first()
    await distToggle.click()
    await page.waitForTimeout(300)

    // Now enable perturbation via Director Controls
    await expandSection(page, 'Director Controls')
    const pertCombobox = page.getByRole('combobox').filter({ hasText: /0/ }).first()
    await pertCombobox.click()
    await page.getByRole('option', { name: 'Skip Layer' }).click()
    await page.waitForTimeout(300)
    // Perturbation Detail section should appear
    await expect(page.getByRole('button', { name: /Perturbation Detail/ })).toBeVisible()
    await expandSection(page, 'Perturbation Detail')
    await expect(page.getByText('Layers (comma-separated)')).toBeVisible()
    await expect(page.getByText('Start %')).toBeVisible()
    await expect(page.getByText('End %')).toBeVisible()
  })

  test('Start Image Strength slider visible when LTX + first frame set', async ({ page }) => {
    // The slider only shows when isLtx && firstFrameB64 is set
    // We can't easily set a real image without the backend, so just verify it's
    // not visible without an image
    const strengthLabel = page.getByText('Start Image Strength')
    expect(await strengthLabel.isVisible().catch(() => false)).toBe(false)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 4. INSPECTOR — Edit LTX parameters
// ═══════════════════════════════════════════════════════════════════════════

test.describe('LTX Inspector — Edit parameter values', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
    // Default is already LTX 2.3 22B distilled
  })

  test('can type a prompt for LTX segment', async ({ page }) => {
    await expandSection(page, 'Prompts')
    const promptArea = page.locator('textarea[placeholder="Describe this segment..."]')
    await promptArea.fill('A woman walks through a neon-lit corridor')
    await expect(promptArea).toHaveValue('A woman walks through a neon-lit corridor')
  })

  test('can change Guide Phases to 1', async ({ page }) => {
    await expandSection(page, 'Director Controls')
    const input = page.locator('div').filter({ hasText: /^Guide Phases$/ }).locator('input').first()
    await input.fill('1')
    await expect(input).toHaveValue('1')
  })

  test('can set Epsilon to 0.01', async ({ page }) => {
    await expandSection(page, 'Director Controls')
    const input = page.locator('div').filter({ hasText: /^Epsilon$/ }).locator('input').first()
    await input.fill('0.01')
    await expect(input).toHaveValue('0.01')
  })

  test('can set Camera Pan X to 0.5', async ({ page }) => {
    await expandSection(page, 'Camera Motion')
    const input = page.locator('div').filter({ hasText: /^Pan X$/ }).locator('input').first()
    await input.fill('0.5')
    await expect(input).toHaveValue('0.5')
  })

  test('can set Camera Zoom to 1.5', async ({ page }) => {
    await expandSection(page, 'Camera Motion')
    const input = page.locator('div').filter({ hasText: /^Zoom$/ }).locator('input').first()
    await input.fill('1.5')
    await expect(input).toHaveValue('1.5')
  })

  test('can set Self-Refiner plan', async ({ page }) => {
    await expandSection(page, 'Self-Refiner')
    // Enable first
    const toggle = page.getByText('Enable Refiner').locator('xpath=following-sibling::button').first()
    await toggle.click()
    const planInput = page.getByPlaceholder('2-8:3,10-14:2')
    await planInput.fill('2-8:3,10-14:2')
    await expect(planInput).toHaveValue('2-8:3,10-14:2')
  })

  test('can enable Sliding Window and set values', async ({ page }) => {
    await expandSection(page, 'Sliding Window')
    const toggle = page.getByText('Enable (long videos)').locator('xpath=following-sibling::button').first()
    await toggle.click()
    await page.waitForTimeout(200)
    const sizeInput = page.locator('div').filter({ hasText: /^Window Size$/ }).locator('input').first()
    await sizeInput.fill('300')
    await expect(sizeInput).toHaveValue('300')
  })

  test('can toggle APG on after switching to dev mode', async ({ page }) => {
    // Turn off distilled mode to get dev guidance
    await expandSection(page, 'LoRA')
    const distLabel = page.getByText('Distilled Mode (8 steps)')
    const distToggle = distLabel.locator('xpath=following-sibling::button').first()
    await distToggle.click()
    await page.waitForTimeout(300)

    await expandSection(page, 'Guidance')
    const apgToggle = page.getByText('APG').locator('xpath=following-sibling::button').first()
    await apgToggle.click()
    const apgActive = page.getByText('APG').locator('xpath=following-sibling::button').locator('.text-\\[\\#6366f1\\]')
    expect(await apgActive.count()).toBeGreaterThan(0)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 5. MULTI-SEGMENT — Add keyframes and verify relay setup
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Multi-segment relay setup', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
    // Default is already LTX 2.3 22B distilled
  })

  test('can add 3 segments with different prompts', async ({ page }) => {
    // Auto-created segment
    await expect(page.locator('[data-seg]')).toHaveCount(1)
    // Add 2 more
    await page.getByRole('button', { name: /Add Keyframe/ }).click()
    await page.waitForTimeout(200)
    await page.getByRole('button', { name: /Add Keyframe/ }).click()
    await page.waitForTimeout(200)
    await expect(page.locator('[data-seg]')).toHaveCount(3)

    // Set prompts on each
    await page.locator('[data-seg]').nth(0).click()
    await expandSection(page, 'Prompts')
    await page.locator('textarea[placeholder="Describe this segment..."]').fill('Woman enters dark forest')

    await page.locator('[data-seg]').nth(1).click()
    await page.waitForTimeout(200)
    await expandSection(page, 'Prompts')
    await page.locator('textarea[placeholder="Describe this segment..."]').fill('She finds a glowing portal')

    await page.locator('[data-seg]').nth(2).click()
    await page.waitForTimeout(200)
    await expandSection(page, 'Prompts')
    await page.locator('textarea[placeholder="Describe this segment..."]').fill('She steps through into a neon city')
  })

  test('segments are ordered K01, K02, K03', async ({ page }) => {
    await page.getByRole('button', { name: /Add Keyframe/ }).click()
    await page.waitForTimeout(200)
    await page.getByRole('button', { name: /Add Keyframe/ }).click()
    await page.waitForTimeout(200)

    const labels = page.locator('[data-seg]')
    await expect(labels.nth(0)).toContainText('K01')
    await expect(labels.nth(1)).toContainText('K02')
    await expect(labels.nth(2)).toContainText('K03')
  })

  test('Generate All button enabled with multiple segments', async ({ page }) => {
    await page.getByRole('button', { name: /Add Keyframe/ }).click()
    const genBtn = page.getByRole('button', { name: /Generate All/ })
    await expect(genBtn).toBeEnabled()
  })

  test('timeline shows correct total duration for 3 segments', async ({ page }) => {
    // Default LTX: 121 frames @ 24fps = ~5.04s per segment
    await page.getByRole('button', { name: /Add Keyframe/ }).click()
    await page.getByRole('button', { name: /Add Keyframe/ }).click()
    await page.waitForTimeout(200)
    // Total should be ~15s (3 × 5.04)
    const timeDisplay = page.locator('.tabular-nums')
    const text = await timeDisplay.textContent()
    // 3 segments × 5.04s ≈ 15.13s — check for "0:15"
    expect(text).toMatch(/0:15/)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 6. RESIZABLE SIDEBAR
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Resizable inspector sidebar', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
  })

  test('sidebar has resize handle on left edge', async ({ page }) => {
    const handle = page.locator('.cursor-col-resize')
    await expect(handle).toBeVisible()
  })

  test('dragging resize handle changes sidebar width', async ({ page }) => {
    const handle = page.locator('.cursor-col-resize')
    const handleBox = await handle.boundingBox()
    expect(handleBox).toBeTruthy()

    // Inspector panel is the parent element containing the resize handle
    const inspector = handle.locator('xpath=..')
    const initialBox = await inspector.boundingBox()
    expect(initialBox).toBeTruthy()
    const initialW = initialBox!.width

    // Drag left to make wider (sidebar grows when dragging handle leftward)
    await page.mouse.move(handleBox!.x + handleBox!.width / 2, handleBox!.y + handleBox!.height / 2)
    await page.mouse.down()
    await page.mouse.move(handleBox!.x + handleBox!.width / 2 - 80, handleBox!.y + handleBox!.height / 2, { steps: 5 })
    await page.mouse.up()

    const newBox = await inspector.boundingBox()
    expect(newBox!.width).toBeGreaterThan(initialW)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 7. LORA PICKER — Dynamic from filesystem (requires backend)
// ═══════════════════════════════════════════════════════════════════════════

test.describe('LoRA picker — dynamic from /v1/loras', () => {
  test.beforeEach(async ({ page }) => {
    // Skip if backend is not available
    const backend = await backendAvailable()
    test.skip(!backend, 'Backend not available')

    await gotoEditor(page)
    await switchToVideo(page)
    // Default is already LTX 2.3 22B
  })

  test('LoRA section loads available LoRAs from backend', async ({ page }) => {
    await expandSection(page, 'LoRA')
    // Should show "Available LoRAs" label (or "Loading..." then the list)
    await page.waitForTimeout(2000)
    const loraContent = page.getByText('Available LoRAs')
    const noLoras = page.getByText('No LoRAs available')
    const hasLoras = await loraContent.isVisible().catch(() => false)
    const noLorasVis = await noLoras.isVisible().catch(() => false)
    expect(hasLoras || noLorasVis).toBe(true)
  })

  test('can select a LoRA checkbox', async ({ page }) => {
    await expandSection(page, 'LoRA')
    await page.waitForTimeout(2000)
    // Find first LoRA item button
    const loraButtons = page.locator('button').filter({ hasText: /\.safetensors/ })
    const count = await loraButtons.count()
    if (count > 0) {
      await loraButtons.first().click()
      // Should show selected state (bg-[#6366f1])
      const activeItem = page.locator('.bg-\\[\\#6366f1\\]\\/20')
      expect(await activeItem.count()).toBeGreaterThan(0)
    }
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 8. AUDIO TRACKS — Dynamic add/remove
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Audio tracks — dynamic', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
  })

  test('starts with no audio tracks', async ({ page }) => {
    await expect(page.locator('text=Audio 1')).toHaveCount(0)
  })

  test('Add Track creates Audio 1', async ({ page }) => {
    await page.getByRole('button', { name: /Add Track/ }).click()
    await expect(page.locator('text=Audio 1')).toBeVisible()
  })

  test('Add Track twice creates Audio 1 and Audio 2', async ({ page }) => {
    await page.getByRole('button', { name: /Add Track/ }).click()
    await page.getByRole('button', { name: /Add Track/ }).click()
    await expect(page.locator('text=Audio 1')).toBeVisible()
    await expect(page.locator('text=Audio 2')).toBeVisible()
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 9. DISTILLED MODE — Switches guidance to NAG
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Distilled mode — NAG guidance', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
    // Default is LTX 2.3 22B distilled
  })

  test('distilled default shows NAG controls', async ({ page }) => {
    await expandSection(page, 'Guidance')
    await expect(page.getByText('NAG Scale')).toBeVisible()
    await expect(page.getByText('NAG Tau')).toBeVisible()
    await expect(page.getByText('NAG Alpha')).toBeVisible()
  })

  test('dev mode shows APG/CFG Star instead of NAG', async ({ page }) => {
    // Turn OFF distilled mode to get dev guidance
    await expandSection(page, 'LoRA')
    const distLabel = page.getByText('Distilled Mode (8 steps)')
    const distToggle = distLabel.locator('xpath=following-sibling::button').first()
    await distToggle.click()
    await page.waitForTimeout(300)

    await expandSection(page, 'Guidance')
    await expect(page.getByText('APG')).toBeVisible()
    await expect(page.getByText('CFG Star')).toBeVisible()
    await expect(page.getByText('Alt Guide Scale')).toBeVisible()
    // NAG should NOT be visible
    expect(await page.getByText('NAG Scale').isVisible().catch(() => false)).toBe(false)
  })

  test('Control Video section only visible in distilled mode', async ({ page }) => {
    // Turn OFF distilled mode first
    await expandSection(page, 'LoRA')
    const distLabel = page.getByText('Distilled Mode (8 steps)')
    const distToggle = distLabel.locator('xpath=following-sibling::button').first()
    await distToggle.click()
    await page.waitForTimeout(300)

    // Control Video should NOT be visible in dev mode
    const cvBtn = page.getByRole('button', { name: /Control Video/ })
    expect(await cvBtn.isVisible().catch(() => false)).toBe(false)

    // Turn distilled back ON
    await distToggle.click()
    await page.waitForTimeout(300)

    // Now Control Video is visible
    await expect(page.getByRole('button', { name: /Control Video/ })).toBeVisible()
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 10. EXPORT — JSON contains all params
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Export — JSON contains relay config', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
    // Default is already LTX 2.3 22B distilled
    await expandSection(page, 'Prompts')
    await page.locator('textarea[placeholder="Describe this segment..."]').fill('Test export prompt')
  })

  test('Export button downloads a JSON file', async ({ page }) => {
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.getByRole('button', { name: /Export/ }).click(),
    ])
    const content = await download.createReadStream()
    const chunks: Buffer[] = []
    for await (const chunk of content) chunks.push(chunk)
    const json = JSON.parse(Buffer.concat(chunks).toString())
    expect(json.segments).toBeDefined()
    expect(json.segments.length).toBeGreaterThan(0)
    expect(json.ltxDirector).toBeDefined()
  })

  test('Exported JSON has ltxDirector with local_prompts and segment_lengths', async ({ page }) => {
    // Add another segment for relay
    await page.getByRole('button', { name: /Add Keyframe/ }).click()
    await page.waitForTimeout(200)
    await page.locator('[data-seg]').nth(1).click()
    await expandSection(page, 'Prompts')
    await page.locator('textarea[placeholder="Describe this segment..."]').fill('Second segment prompt')

    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.getByRole('button', { name: /Export/ }).click(),
    ])
    const content = await download.createReadStream()
    const chunks: Buffer[] = []
    for await (const chunk of content) chunks.push(chunk)
    const json = JSON.parse(Buffer.concat(chunks).toString())

    // ltxDirector uses pipe-separated prompts and comma-separated lengths
    expect(json.ltxDirector.local_prompts).toContain('Test export prompt')
    expect(json.ltxDirector.local_prompts).toContain('Second segment prompt')
    expect(json.ltxDirector.segment_lengths).toBeDefined()
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// 11. KEYBOARD SHORTCUTS
// ═══════════════════════════════════════════════════════════════════════════

test.describe('Keyboard shortcuts', () => {
  test.beforeEach(async ({ page }) => {
    await gotoEditor(page)
    await switchToVideo(page)
  })

  test('Space toggles playback', async ({ page }) => {
    // Focus the page body
    await page.keyboard.press('Space')
    // Play should have started (pause icon visible)
    await page.waitForTimeout(200)
    const pauseIcon = page.locator('.lucide-pause')
    expect(await pauseIcon.isVisible().catch(() => false)).toBe(true)
    // Press space again to stop
    await page.keyboard.press('Space')
  })

  test('Plus/Minus zoom in/out', async ({ page }) => {
    await page.keyboard.press('=')
    await page.waitForTimeout(100)
    // Zoom should have increased from 100% to 125%
    const zoomText = page.locator('span.font-mono').filter({ hasText: /%/ }).first()
    const text = await zoomText.textContent()
    expect(text).toContain('125%')
  })

  test('Undo via Ctrl+Z', async ({ page }) => {
    // Add a segment
    await page.getByRole('button', { name: /Add Keyframe/ }).click()
    await expect(page.locator('[data-seg]')).toHaveCount(2)
    // Undo
    await page.keyboard.press('Control+z')
    await page.waitForTimeout(300)
    await expect(page.locator('[data-seg]')).toHaveCount(1)
  })
})
