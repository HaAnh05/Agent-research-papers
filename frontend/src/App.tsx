import { useEffect } from 'react'
import { Navigate, Route, Routes, useNavigate } from 'react-router-dom'
import { LandingPage } from './pages/LandingPage'
import { RunPage } from './pages/RunPage'
import { useResearch } from './context/ResearchContext'

function CurrentRunRedirect() {
  const navigate = useNavigate()
  const research = useResearch()

  // ``/run/current`` was part of the earlier desk shell. Keep it as a small
  // compatibility redirect while the product has one canonical run route.
  const currentId = research.currentRun?.runId
  const target = currentId ? `/run/${encodeURIComponent(currentId)}` : '/'

  useEffect(() => {
    navigate(target, { replace: true })
  }, [navigate, target])

  return <main id="main-content" className="route-loading" aria-live="polite">Opening research…</main>
}

export function App() {
  return (
    <>
      <a className="skip-link" href="#main-content">Skip to content</a>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/run/current" element={<CurrentRunRedirect />} />
        <Route path="/run/:id" element={<RunPage />} />
        <Route path="/library" element={<Navigate to="/" replace />} />
        <Route path="/library/*" element={<Navigate to="/" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  )
}
