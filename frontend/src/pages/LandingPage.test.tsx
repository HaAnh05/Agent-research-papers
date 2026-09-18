import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from '../lib/api'
import type { RunSnapshot } from '../types'
import { LandingPage } from './LandingPage'

const mocked = vi.hoisted(() => ({
  research: {
    activeRun: false,
    currentRun: null,
    startRun: vi.fn(),
  },
}))

vi.mock('../context/ResearchContext', () => ({
  useResearch: () => mocked.research,
}))

function run(runId = 'run-created'): RunSnapshot {
  return {
    runId,
    input: { query: 'attention methods', paperInputs: [], fileNames: [] },
    provider: 'fixture',
    model: 'fixture-model',
    status: 'running',
    progress: 0,
    traceEvents: [],
    result: null,
  }
}

function fixtureFile(name: string, body: string, type: string): File {
  const file = new File([body], name, { type })
  // jsdom's Blob implementation does not expose arrayBuffer consistently.
  // Keep the fixture's signature explicit so this test exercises the same
  // validation branch in browsers and in Vitest.
  Object.defineProperty(file, 'slice', {
    configurable: true,
    value: () => ({ arrayBuffer: async () => new TextEncoder().encode(body.slice(0, 5)).buffer }),
  })
  return file
}

function LocationProbe() {
  const location = useLocation()
  return <output data-testid="location">{location.pathname}</output>
}

function renderLanding() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/run/:id" element={<LocationProbe />} />
      </Routes>
    </MemoryRouter>,
  )
}

beforeEach(() => {
  mocked.research = {
    activeRun: false,
    currentRun: null,
    startRun: vi.fn().mockResolvedValue(run()),
  }
})

describe('LandingPage', () => {
  it('submits a topic with Enter and keeps Shift+Enter as a newline', async () => {
    const user = userEvent.setup()
    renderLanding()
    const textarea = screen.getByRole('textbox', { name: 'Research request' })

    await user.type(textarea, 'attention methods')
    await user.keyboard('{Shift>}{Enter}{/Shift}')
    expect(textarea).toHaveValue('attention methods\n')
    expect(mocked.research.startRun).not.toHaveBeenCalled()

    await user.keyboard('{Enter}')
    await waitFor(() => expect(mocked.research.startRun).toHaveBeenCalledTimes(1))
    expect(mocked.research.startRun).toHaveBeenCalledWith({ query: 'attention methods', paperInputs: [], files: [] })
    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/run/run-created'))
  })

  it('does not start an empty request and reports a specific validation message', async () => {
    const user = userEvent.setup()
    renderLanding()

    const submit = screen.getByRole('button', { name: 'Start research' })
    expect(submit).toBeDisabled()
    expect(screen.getByText(/Enter a topic, arXiv reference, or attach a PDF/i)).toBeInTheDocument()
    await user.click(submit)

    expect(mocked.research.startRun).not.toHaveBeenCalled()
    expect(submit).toBeDisabled()
  })

  it('passes a direct arXiv URL as a paper input without turning it into a search query', async () => {
    const user = userEvent.setup()
    renderLanding()
    const textarea = screen.getByRole('textbox', { name: 'Research request' })

    await user.type(textarea, 'https://arxiv.org/abs/1706.03762')
    await user.click(screen.getByRole('button', { name: 'Start research' }))

    await waitFor(() => expect(mocked.research.startRun).toHaveBeenCalledTimes(1))
    expect(mocked.research.startRun).toHaveBeenCalledWith({ query: '', paperInputs: ['1706.03762'], files: [] })
  })

  it('rejects an unsupported URL and a browser local path before calling the backend', async () => {
    const user = userEvent.setup()
    renderLanding()
    const textarea = screen.getByRole('textbox', { name: 'Research request' })

    await user.type(textarea, 'https://example.com/paper.pdf')
    expect(screen.getByRole('alert')).toHaveTextContent(/Only public arXiv paper URLs are supported/i)
    expect(screen.getByRole('button', { name: 'Start research' })).toBeDisabled()
    await user.clear(textarea)
    await user.type(textarea, '/tmp/paper.pdf')
    expect(screen.getByRole('alert')).toHaveTextContent(/Local PDF paths cannot be read/i)
    expect(mocked.research.startRun).not.toHaveBeenCalled()
  })

  it('validates attached PDFs, submits a valid attachment, and supports removal', async () => {
    const user = userEvent.setup({ applyAccept: false })
    const { container } = renderLanding()
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    const invalid = fixtureFile('notes.txt', 'not a pdf', 'text/plain')

    await user.upload(input, invalid)
    expect(await screen.findByRole('alert')).toHaveTextContent(/not a supported PDF file/i)
    expect(mocked.research.startRun).not.toHaveBeenCalled()

    const valid = fixtureFile('paper.pdf', '%PDF-1.7\nfixture', 'application/pdf')
    await user.upload(input, valid)
    expect(await screen.findByText('paper.pdf')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Remove paper.pdf' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Remove paper.pdf' }))
    expect(screen.queryByText('paper.pdf')).not.toBeInTheDocument()
    await user.upload(input, valid)
    await user.click(screen.getByRole('button', { name: 'Start research' }))
    await waitFor(() => expect(mocked.research.startRun).toHaveBeenCalledTimes(1))
    expect(mocked.research.startRun).toHaveBeenCalledWith({ query: '', paperInputs: [], files: [valid] })
  })

  it('rejects a PDF named file whose content has no PDF signature', async () => {
    const user = userEvent.setup()
    const { container } = renderLanding()
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    const invalid = fixtureFile('paper.pdf', 'plain text', 'application/pdf')

    await user.upload(input, invalid)

    expect(await screen.findByRole('alert')).toHaveTextContent(/not a valid PDF file|could not be read/i)
    expect(mocked.research.startRun).not.toHaveBeenCalled()
  })

  it('prefills suggestion intent without silently launching a run', async () => {
    const user = userEvent.setup()
    renderLanding()

    await user.click(screen.getByRole('button', { name: 'Analyze paper' }))

    expect(screen.getByRole('textbox', { name: 'Research request' })).toHaveAttribute('placeholder', 'Paste an arXiv ID or link…')
    expect(mocked.research.startRun).not.toHaveBeenCalled()
  })

  it('routes to an existing active run when the API returns a truthful 409', async () => {
    const user = userEvent.setup()
    mocked.research.startRun.mockRejectedValueOnce(new ApiError('A run is already active.', 409, { detail: { activeRunId: 'run-existing' } }))
    renderLanding()

    await user.type(screen.getByRole('textbox', { name: 'Research request' }), 'compare papers')
    await user.click(screen.getByRole('button', { name: 'Start research' }))

    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/run/run-existing'))
    expect(mocked.research.startRun).toHaveBeenCalledTimes(1)
  })

  it('prevents duplicate submissions while the first run is unresolved', async () => {
    const user = userEvent.setup()
    let resolveRun: (value: RunSnapshot) => void = () => undefined
    mocked.research.startRun.mockImplementationOnce(() => new Promise<RunSnapshot>((resolve) => { resolveRun = resolve }))
    renderLanding()

    await user.type(screen.getByRole('textbox', { name: 'Research request' }), 'one run only')
    const submit = screen.getByRole('button', { name: 'Start research' })
    await user.click(submit)
    expect(submit).toBeDisabled()
    expect(mocked.research.startRun).toHaveBeenCalledTimes(1)
    await user.click(submit)
    expect(mocked.research.startRun).toHaveBeenCalledTimes(1)

    resolveRun(run('run-single'))
    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/run/run-single'))
  })
})
