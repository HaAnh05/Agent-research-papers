// Dev tool: screenshots of the P6 flow (live timers, early summary, report button) in mock mode. Not a test.
//   npm run dev -- --port 5199   (in another terminal)
//   node scripts/screenshots_p6.mjs [baseUrl] [outDir]
import { chromium } from '@playwright/test'

const base = process.argv[2] ?? 'http://127.0.0.1:5199'
const out = process.argv[3] ?? '/tmp'
const url = `${base}/?mock=1`
const shot = (page, name) => page.screenshot({ path: `${out}/p6_${name}.png` })

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
page.on('pageerror', (e) => console.error('pageerror', e.message))
page.on('console', (m) => m.type() === 'error' && console.error('console', m.text()))

await page.goto(url)
await page.waitForSelector('h1')
await page.getByRole('button', { name: 'Phân tích một bài' }).click()
await page.getByRole('button', { name: 'Gửi' }).click()
await page.waitForSelector('[data-testid=progress]')

// 1. running: expanded progress with live timers (two shots one second apart show the tick)
await page.waitForTimeout(2500)
await page.locator('[data-testid=progress] > button').click()
await page.waitForTimeout(300)
const timers = async () => page.locator('.step .t').allTextContents()
console.log('timers t0', await timers(), await page.locator('.pl').textContent())
await shot(page, 'running_timers')
await page.waitForTimeout(1500)
console.log('timers t1', await timers(), await page.locator('.pl').textContent())
await shot(page, 'running_timers_later')

// 2. early summary: answer on screen, report button grey, progress still running, no graph yet
await page.waitForSelector('.btn.pending', { timeout: 20000 })
await page.waitForTimeout(700)
console.log('report button grey', await page.locator('.btn.pending').count(), 'black', await page.locator('.btn.primary').count(), 'foundations', await page.getByText('Nền tảng').count(), await page.locator('.pl').textContent())
await page.locator('.thread').evaluate((el) => el.scrollTo({ top: 0 }))
await page.waitForTimeout(500)
await shot(page, 'early_summary')
await page.locator('.thread').evaluate((el) => el.scrollTo({ top: el.scrollHeight }))
await page.waitForTimeout(500)
await shot(page, 'early_summary_bottom')

// 3. graph arrives later: Nền tảng section appears, panel opens
await page.waitForTimeout(3200)
console.log('foundations visible', await page.getByText('Nền tảng').count())
await shot(page, 'graph_arrived')

// 4. final: black report button, header with total time
await page.waitForFunction(() => document.querySelector('.btn.primary') !== null, null, { timeout: 30000 })
await page.waitForTimeout(800)
console.log('final: pending', await page.locator('.btn.pending').count(), 'black', await page.locator('.btn.primary').count(), await page.locator('.pl').textContent())
await shot(page, 'final')
if ((await page.locator('.prog.open').count()) === 0) await page.locator('[data-testid=progress] > button').click()
await page.locator('.thread').evaluate((el) => el.scrollTo({ top: 0 }))
await page.waitForTimeout(400)
await shot(page, 'final_progress')

await browser.close()
