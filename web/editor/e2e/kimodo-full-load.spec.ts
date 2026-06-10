import { test, expect } from '@playwright/test'

test('Kimodo dialog full load from scratch (with eviction)', async ({ page }) => {
  test.setTimeout(300_000)

  // Capture browser console
  page.on('console', msg => {
    if (msg.text().includes('[KimodoDialog]')) {
      console.log(`BROWSER: ${msg.text()}`)
    }
  })

  await page.goto('http://localhost:5173/editor/')
  await page.waitForTimeout(2000)

  const kimodoLink = page.getByText(/kimodo/i).first()
  await kimodoLink.click()
  await page.waitForTimeout(1000)

  const dialog = page.locator('[role="dialog"]')
  expect(await dialog.isVisible()).toBe(true)

  const spinner = dialog.locator('.animate-spin')
  console.log('Spinner visible after open:', await spinner.isVisible())

  // Poll for iframe — load takes ~2 min
  const iframe = dialog.locator('iframe[title="Kimodo Motion Studio"]')
  let iframeVisible = false
  for (let i = 0; i < 36; i++) {
    iframeVisible = await iframe.isVisible().catch(() => false)
    if (iframeVisible) break
    const spinnerNow = await spinner.isVisible().catch(() => false)
    const errorNow = await dialog.locator('text=Failed to load').isVisible().catch(() => false)
    console.log(`...${(i+1)*5}s: spinner=${spinnerNow} error=${errorNow}`)
    if (errorNow) {
      await page.screenshot({ path: '/tmp/kimodo-full-fail.png', fullPage: true })
      throw new Error('Load failed with error')
    }
    await page.waitForTimeout(5000)
  }

  if (!iframeVisible) {
    await page.screenshot({ path: '/tmp/kimodo-full-timeout.png', fullPage: true })
    throw new Error('Iframe never became visible within 3 minutes')
  }

  console.log('SUCCESS: Iframe visible after full load')
  await page.screenshot({ path: '/tmp/kimodo-full-success.png', fullPage: true })
  expect(iframeVisible).toBe(true)
})
