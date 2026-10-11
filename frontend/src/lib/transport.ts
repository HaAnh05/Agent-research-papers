import { httpTransport, type Transport } from './api'

/**
 * Dev-only mock mode: `?mock=1` (vite dev server) or a build with VITE_MOCK=1.
 * The mock lives in src/dev and is loaded lazily, so a normal production build never includes it.
 */
const useMock =
  import.meta.env.VITE_MOCK === '1' ||
  (import.meta.env.DEV && typeof location !== 'undefined' && new URLSearchParams(location.search).get('mock') === '1')

function lazyMock(): Transport {
  const load = () => import('../dev/mockTransport').then((m) => m.mockTransport)
  return {
    createThread: () => load().then((t) => t.createThread()),
    sendMessage: (threadId, text, file, arxivId) => load().then((t) => t.sendMessage(threadId, text, file, arxivId)),
    getRun: (runId) => load().then((t) => t.getRun(runId)),
    downloadReport: (id) => load().then((t) => t.downloadReport(id)),
    subscribe(runId, afterSeq, onFrame, onError) {
      let close: (() => void) | undefined
      let closed = false
      void load().then((t) => {
        if (!closed) close = t.subscribe(runId, afterSeq, onFrame, onError)
      })
      return () => {
        closed = true
        close?.()
      }
    },
  }
}

export const transport: Transport = useMock ? lazyMock() : httpTransport
export const isMock = useMock
