// Dev tool: screenshots of the UI in mock mode. Not a test.
//   npm run dev -- --port 5199   (in another terminal)
//   node scripts/screenshots.mjs [baseUrl] [outDir]
import { chromium } from '@playwright/test'

const base = process.argv[2] ?? 'http://127.0.0.1:5199'
const out = process.argv[3] ?? '/tmp'
const url = `${base}/?mock=1&slow=0.6`
const shot = (page, name) => page.screenshot({ path: `${out}/p5_${name}.png` })

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
page.on('pageerror', (e) => console.error('pageerror', e.message))
page.on('console', (m) => m.type() === 'error' && console.error('console', m.text()))

// 1. landing
await page.goto(url)
await page.waitForSelector('h1')
await page.waitForTimeout(500)
await shot(page, 'landing')

// 2. running, progress expanded
await page.getByRole('button', { name: 'Phân tích một bài' }).click()
await page.getByRole('button', { name: 'Gửi' }).click()
await page.waitForSelector('[data-testid=progress]')
await page.waitForTimeout(2600)
await page.locator('[data-testid=progress] > button').click()
await page.waitForTimeout(300)
await shot(page, 'running')

// 3. result with graph panel
await page.waitForSelector('[data-testid=answer]', { timeout: 20000 })
await page.waitForTimeout(1500)
await shot(page, 'result_graph')
await page.locator('[data-testid=progress] > button').click() // collapse again
await page.waitForTimeout(300)
await page.locator('.thread').evaluate((el) => el.scrollTo({ top: 0 }))
await page.waitForTimeout(500)
await shot(page, 'result_top')
await page.locator('.thread').evaluate((el) => el.scrollTo({ top: el.scrollHeight }))
await page.waitForTimeout(400)
await shot(page, 'result_bottom')

// 4. node detail
await page.locator('.plist ol li button').nth(1).click()
await page.waitForTimeout(400)
await shot(page, 'node_detail')

// 4b. "Phân tích bài này": a new run in the same thread, graph panel switches to the new paper
await page.getByRole('button', { name: 'Phân tích bài này' }).click()
await page.waitForFunction(() => document.querySelectorAll('[data-testid=answer]').length === 2, null, { timeout: 25000 })
await page.waitForTimeout(1200)
await shot(page, 'analyse_node')

// 5. quick follow-up
await page.getByRole('button', { name: /^Giải thích/ }).click().catch(() => undefined)
await page.getByPlaceholder('Hỏi về một bài báo hoặc chủ đề…').fill('Vì sao tự phê bình bằng prompt thất bại?')
await page.keyboard.press('Enter')
await page.waitForTimeout(2500)
await shot(page, 'followup_quick')

// 6. multi-paper result (fresh page)
const page2 = await browser.newPage({ viewport: { width: 1440, height: 900 } })
await page2.goto(url)
await page2.getByRole('button', { name: 'Tìm theo chủ đề' }).click()
await page2.getByRole('button', { name: 'Gửi' }).click()
await page2.waitForSelector('[data-testid=answer]', { timeout: 25000 })
await page2.waitForTimeout(1500)
await page2.locator('.card > button').nth(0).click()
await page2.locator('.card > button').nth(1).click()
await page2.waitForTimeout(400)
await shot(page2, 'multi')

// 7. failed run
const page3 = await browser.newPage({ viewport: { width: 1440, height: 900 } })
await page3.goto(url)
await page3.getByPlaceholder('Hỏi về một bài báo hoặc chủ đề…').fill('thử một bài lỗi')
await page3.keyboard.press('Enter')
await page3.waitForSelector('.banner.err', { timeout: 10000 })
await page3.locator('[data-testid=progress] > button').click()
await page3.waitForTimeout(400)
await shot(page3, 'error')

await browser.close()
