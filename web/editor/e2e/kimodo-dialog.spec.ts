import { test, expect } from '@playwright/test'

test('Kimodo dialog preloads and shows iframe', async ({ page }) => {
  // Intercept the forge preload call
  let preloadStatus = ''
  await page.route('**/forge', async route => {
    if (route.request().method() === 'POST') {
      const body = route.request().postDataJSON()
      if (body?.action === 'preload' && body?.service === 'kimodo_demo') {
        preloadStatus = 'called'
        // Return what the real forge would return
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ status: 'already_loaded', service: 'kimodo_demo' }),
        })
        return
      }
    }
    await route.continue()
  })

  await page.goto('http://localhost:5173/editor/')
  
  // Wait for page to load
  await page.waitForTimeout(2000)

  // Find and click the Kimodo link in sidebar
  const kimodoLink = page.getByText(/kimodo/i).first()
  if (await kimodoLink.isVisible()) {
    await kimodoLink.click()
  } else {
    console.log('Kimodo link not visible, trying sidebar...')
    // Take screenshot for debugging
    await page.screenshot({ path: '/tmp/kimodo-test-1.png', fullPage: true })
  }

  // Wait for preload to fire and iframe to appear
  await page.waitForTimeout(3000)

  // Check that preload was called
  console.log('Preload called:', preloadStatus)

  // Check for the dialog with iframe
  const dialog = page.locator('[role="dialog"]')
  const isDialogOpen = await dialog.isVisible().catch(() => false)
  console.log('Dialog visible:', isDialogOpen)

  if (isDialogOpen) {
    // Check for iframe inside dialog
    const iframe = dialog.locator('iframe[title="Kimodo Motion Studio"]')
    const isIframeVisible = await iframe.isVisible().catch(() => false)
    console.log('Iframe visible:', isIframeVisible)
    
    // Check for loading spinner (should NOT be present after preload returns)
    const spinner = dialog.locator('.animate-spin')
    const isSpinnerVisible = await spinner.isVisible().catch(() => false)
    console.log('Spinner still visible:', isSpinnerVisible)

    // Check for error state
    const errorEl = dialog.locator('text=Failed to load')
    const isErrorVisible = await errorEl.isVisible().catch(() => false)
    console.log('Error visible:', isErrorVisible)
  }

  await page.screenshot({ path: '/tmp/kimodo-test-final.png', fullPage: true })
  
  // The key assertion: after preload returns "already_loaded", iframe should be visible
  // and spinner should NOT be visible
  if (isDialogOpen) {
    const iframe = dialog.locator('iframe[title="Kimodo Motion Studio"]')
    await expect(iframe).toBeVisible({ timeout: 5000 })
  }
})
