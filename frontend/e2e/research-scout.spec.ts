import { expect, test, type Page } from '@playwright/test'

const now = '2026-09-18T10:00:00.000Z'
const runId = 'run-e2e'

type Scenario = 'direct' | 'comparison' | 'search-running'

function directFixtures() {
  const running = {
    runId,
    input: { query: 'Explain 1706.03762', paperInputs: ['1706.03762'], fileNames: [] },
    provider: 'fixture',
    model: 'fixture-model',
    status: 'running',
    currentNode: 'router',
    progress: 0,
    traceEvents: [],
    seq: 0,
    result: null,
  }
  const traceEvents = [
    { runId, seq: 1, type: 'step.updated', node: 'router', label: 'Router', kind: 'routing', status: 'completed', summary: 'Request routed.', timestamp: now, progress: 20, facts: { intent: 'direct_read' } },
    { runId, seq: 2, type: 'step.updated', node: 'read_paper', label: 'Read paper', kind: 'paper', status: 'completed', summary: 'Read one source PDF.', timestamp: now, progress: 45, facts: { paperCount: 1 } },
    { runId, seq: 3, type: 'step.updated', node: 'write_notes', label: 'Write notes', kind: 'notes', status: 'completed', summary: 'Structured notes finalized.', timestamp: now, progress: 75 },
    { runId, seq: 4, type: 'step.updated', node: 'final_report', label: 'Final report', kind: 'report', status: 'completed', summary: 'Report artifact prepared.', timestamp: now, progress: 90, facts: { reportAvailable: true, reportId: 'report-e2e' } },
    { runId, seq: 5, type: 'run.completed', node: 'final_report', label: 'Final report', kind: 'report', status: 'completed', summary: 'Research run completed.', timestamp: now, progress: 100 },
  ]
  const completed = {
    ...running,
    status: 'success',
    currentNode: 'final_report',
    progress: 100,
    seq: 5,
    traceEvents,
    result: {
      reportId: 'report-e2e',
      report: '# Attention report\n\nA finalized fixture report.',
      papers: [{ paperId: '1706.03762', arxivId: '1706.03762', title: 'Attention Is All You Need', authors: ['Ashish Vaswani'], notes: { briefSummary: 'Self attention replaces recurrence for sequence transduction.\n- Parallel training.\n- Strong translation results.\n- Attention remains costly for long sequences.', problem: 'Sequence transduction', method: 'Self attention', result: 'Strong results', limitation: 'Quadratic attention' } }],
    },
  }
  return { running, completed, traceEvents }
}

function searchFixtures() {
  const running = {
    runId,
    input: { query: 'attention methods', paperInputs: [], fileNames: [] },
    provider: 'fixture',
    model: 'fixture-model',
    status: 'running',
    currentNode: 'refine_query',
    progress: 0,
    traceEvents: [],
    seq: 0,
    result: null,
  }
  const traceEvents = [
    { runId, seq: 1, type: 'step.updated', node: 'router', label: 'Router', kind: 'routing', status: 'completed', summary: 'Request routed.', timestamp: now, progress: 20, facts: { intent: 'search' } },
    { runId, seq: 2, type: 'step.updated', node: 'search_papers', label: 'Search', kind: 'search', status: 'completed', summary: 'Search returned candidates.', timestamp: now, progress: 35, facts: { resultCount: 5 } },
    { runId, seq: 3, type: 'step.updated', node: 'eval_search', label: 'Evaluate', kind: 'evaluation', status: 'completed', summary: 'Selected candidates.', timestamp: now, progress: 45, facts: { selectedCount: 2 } },
    { runId, seq: 4, type: 'step.updated', node: 'refine_query', label: 'Refine', kind: 'search', status: 'completed', summary: 'Refined search query.', timestamp: now, progress: 50, facts: { retryCount: 1 } },
    { runId, seq: 5, type: 'step.updated', node: 'refine_query', label: 'Refine', kind: 'search', status: 'completed', summary: 'Refined search query.', timestamp: now, progress: 55, facts: { retryCount: 2 } },
  ]
  return { running, traceEvents }
}

function comparisonFixtures() {
  const base = directFixtures()
  const papers = [
    { paperId: 'arxiv_1706.03762', arxivId: '1706.03762', title: 'Attention Is All You Need', notes: { briefSummary: 'Attention replaces recurrence.\n- Parallel training.\n- Strong translation results.\n- Long sequences remain costly.', problem: 'Sequence transduction', method: 'Self attention', result: 'Strong results', limitation: 'Quadratic attention' } },
    { paperId: 'arxiv_2005.14165', arxivId: '2005.14165', title: 'Language Models are Few-Shot Learners', notes: { briefSummary: 'Large models improve few-shot learning.\n- Scale matters.\n- Prompting changes evaluation.\n- Compute use is high.', problem: 'Few-shot learning', method: 'Autoregressive language model', result: 'Few-shot results', limitation: 'Compute cost' } },
  ]
  const traceEvents = [
    ...base.traceEvents.slice(0, 3),
    { runId, seq: 4, type: 'step.updated', node: 'compare_benchmark', label: 'Benchmark comparison', kind: 'benchmark', status: 'completed', summary: 'Compared two papers.', timestamp: now, progress: 80, facts: { comparisonAvailable: true, comparedPaperIds: papers.map((paper) => paper.paperId) } },
    { runId, seq: 5, type: 'step.updated', node: 'final_report', label: 'Final report', kind: 'report', status: 'completed', summary: 'Report artifact prepared.', timestamp: now, progress: 90, facts: { reportAvailable: true, reportId: 'report-e2e' } },
    { runId, seq: 6, type: 'run.completed', node: 'final_report', label: 'Final report', kind: 'report', status: 'completed', summary: 'Research run completed.', timestamp: now, progress: 100 },
  ]
  return {
    running: { ...base.running, input: { query: 'Compare two papers', paperInputs: ['1706.03762', '2005.14165'], fileNames: [] } },
    completed: {
      ...base.completed,
      seq: 6,
      traceEvents,
      result: {
        ...base.completed.result,
        papers,
        benchmark: '| Paper | Result |\n| --- | --- |\n| Attention Is All You Need | Reported translation findings |',
        comparisonArtifact: {
          available: true,
          papers: papers.map((paper) => ({ paperId: paper.paperId, title: paper.title, findings: ['A concise finding with an unusually long technical term: autoregressivecontextwindowevaluation'], metrics: [] })),
          rows: [{ metric: 'Accuracy', dataset: 'Different datasets', unit: '%', values: {}, sourceQuotes: {}, comparable: false, reason: 'Not directly comparable' }],
          synthesis: ['The reported results use different tasks.'],
        },
      },
    },
    traceEvents,
  }
}

function sseFrame(type: string, payload: object, seq?: number) {
  return [`event: ${type}`, ...(seq ? [`id: ${seq}`] : []), `data: ${JSON.stringify(payload)}`, ''].join('\n')
}

async function mockApi(page: Page, scenario: Scenario) {
  const fixtures = scenario === 'direct' ? directFixtures() : scenario === 'comparison' ? comparisonFixtures() : searchFixtures()
  await page.route('**/api/config', (route) => route.fulfill({ json: { provider: 'fixture', model: 'fixture-model', configured: true } }))
  await page.route('**/api/threads', (route) => route.fulfill({ json: [] }))
  await page.route('**/api/runs', async (route) => {
    if (route.request().method() !== 'POST') return route.fulfill({ json: [] })
    return route.fulfill({ status: 201, json: fixtures.running })
  })
  await page.route(`**/api/runs/${runId}`, (route) => route.fulfill({ json: scenario === 'direct' ? directFixtures().completed : scenario === 'comparison' ? comparisonFixtures().completed : fixtures.running }))
  await page.route(`**/api/runs/${runId}/events*`, (route) => {
    const frames = [sseFrame('snapshot', fixtures.running)]
    fixtures.traceEvents.forEach((item) => frames.push(sseFrame(item.type, item, item.seq)))
    // Leave the search scenario live so the screenshot captures the actual
    // semantic pipeline; the direct scenario includes a terminal event.
    return route.fulfill({ status: 200, contentType: 'text/event-stream', body: frames.join('\n') + (scenario === 'direct' ? '\n' : '') })
  })
}

test.describe('Landing to Results', () => {
  test.use({ viewport: { width: 1440, height: 900 } })

  test('desktop direct arXiv input opens finalized Results without discovery', async ({ page }) => {
    await mockApi(page, 'direct')
    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'What do you want to understand?' })).toBeVisible()
    await page.getByRole('textbox', { name: 'Research request' }).fill('Explain 1706.03762')
    await page.keyboard.press('Enter')

    await expect(page).toHaveURL(/\/run\/run-e2e$/)
    await expect(page.getByRole('heading', { name: 'Attention Is All You Need' })).toBeVisible({ timeout: 8_000 })
    await expect(page.getByText('Self attention replaces recurrence for sequence transduction.')).toBeVisible()
    await page.getByRole('button', { name: 'View details' }).click()
    await expect(page.getByText('Quadratic attention')).toBeVisible()
    expect(await page.getByText('Discover relevant papers').count()).toBe(0)
    await expect(page.getByRole('tab', { name: 'Summary' })).toBeVisible()
    await expect(page.getByRole('tab', { name: 'Report' })).toBeVisible()
    await page.getByRole('tab', { name: 'Report' }).click()
    await expect(page.getByRole('heading', { name: 'Attention report' })).toHaveCount(1)
    await page.screenshot({ path: '/tmp/research-scout-desktop-results.png', fullPage: true })
  })
})

test.describe('Semantic Pipeline', () => {
  test.use({ viewport: { width: 1024, height: 900 } })

  test('tablet search keeps refinements inside one Discover row and reports real counts', async ({ page }) => {
    await mockApi(page, 'search-running')
    await page.goto('/')
    await page.getByRole('textbox', { name: 'Research request' }).fill('attention methods')
    await page.getByRole('button', { name: 'Start research' }).click()

    await expect(page).toHaveURL(/\/run\/run-e2e$/)
    await expect(page.getByRole('heading', { name: 'Running research' })).toBeVisible({ timeout: 8_000 })
    expect(await page.getByText('Discover relevant papers').count()).toBe(1)
    await expect(page.getByText('Refined query · 2 times')).toBeVisible()
    await expect(page.getByText('searched · 5 papers')).toBeVisible()
    await expect(page.getByText('selected · 2 papers')).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    await page.screenshot({ path: '/tmp/research-scout-tablet-pipeline.png', fullPage: true })
  })
})

test.describe('Responsive Results', () => {
  test.use({ viewport: { width: 342, height: 844 }, reducedMotion: 'reduce' })

  test('mobile direct flow uses a paper selector and avoids page overflow', async ({ page }) => {
    await mockApi(page, 'direct')
    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'What do you want to understand?' })).toBeVisible()
    await page.getByRole('textbox', { name: 'Research request' }).fill('Explain 1706.03762')
    await page.keyboard.press('Enter')

    await expect(page.getByRole('heading', { name: 'Attention Is All You Need' })).toBeVisible({ timeout: 8_000 })
    await expect(page.getByRole('combobox', { name: 'Select paper' })).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    await page.screenshot({ path: '/tmp/research-scout-mobile-342-results.png', fullPage: true })
  })

  test('342px comparison keeps long findings inside the page', async ({ page }) => {
    await mockApi(page, 'comparison')
    await page.goto('/')
    await page.getByRole('textbox', { name: 'Research request' }).fill('Compare 1706.03762 and 2005.14165')
    await page.getByRole('button', { name: 'Start research' }).click()
    await expect(page.getByRole('tab', { name: 'Comparison' })).toBeVisible({ timeout: 8_000 })
    await page.getByRole('tab', { name: 'Comparison' }).click()
    await expect(page.getByText('Not directly comparable')).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  })
})
