import { lazy, Suspense, useEffect, useState } from 'react'
import * as D from './data'
import { LINKS, PRODUCT_NAME } from './config'
import { ThemeCtx, useThemeState } from './theme'
import Arena from './pages/Arena'
import { isShot, isTour, params } from './modes'

const Dials = lazy(() => import('./pages/Dials'))
const Tipping = lazy(() => import('./pages/Tipping'))
const Yardstick = lazy(() => import('./pages/Yardstick'))
const Safeguards = lazy(() => import('./pages/Safeguards'))
const Connect = lazy(() => import('./pages/Connect'))
const About = lazy(() => import('./pages/About'))
const Tour = lazy(() => import('./tour'))

const ROUTES = [['arena', 'Arena'], ['dials', 'Dials'], ['tipping', 'Tipping point'], ['yardstick', 'Yardstick'], ['safeguards', 'Safeguards'], ['connect', 'Connect an agent'], ['about', 'About']] as const

function useHash() {
  const [h, setH] = useState(location.hash)
  useEffect(() => { const f = () => setH(location.hash); addEventListener('hashchange', f); return () => removeEventListener('hashchange', f) }, [])
  const parts = h.replace(/^#\/?/, '').split('?')[0].split('/')
  return { route: parts[0] || 'arena', arg: parts[1] ? decodeURIComponent(parts[1]) : undefined, q: h.includes('?') ? h.slice(h.indexOf('?')) : '' }
}

const SOURCE_LABEL: Record<D.Source, string> = { live: 'Live server', static: 'Static copy', mock: 'Mock data', none: 'No data' }

function ThemeIcon({ pref }: { pref: 'system' | 'light' | 'dark' }) {
  if (pref === 'dark') return <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden><path d="M13.2 9.6A5.6 5.6 0 0 1 6.4 2.8a5.6 5.6 0 1 0 6.8 6.8z" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" /></svg>
  if (pref === 'light') return <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden><circle cx="8" cy="8" r="3" fill="none" stroke="currentColor" strokeWidth="1.4" /><path d="M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" /></svg>
  return <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden><rect x="2" y="3" width="12" height="8.5" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.4" /><path d="M5.5 14h5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" /></svg>
}

const paperHref = LINKS.find(l => /paper/i.test(l.label))?.href ?? null
const codeHref = LINKS.find(l => /code/i.test(l.label))?.href ?? null

export default function App() {
  const { route, arg, q } = useHash()
  const theme = useThemeState()
  const shot = isShot(), tour = isTour()
  useEffect(() => { document.documentElement.classList.toggle('shot', shot); document.documentElement.classList.toggle('tour', tour) }, [shot, tour, q])
  const [src, setSrc] = useState<D.Source | null>(null)
  useEffect(() => { D.init().then(i => setSrc(i.source)) }, [])
  useEffect(() => {
    const name = ROUTES.find(r => r[0] === route)?.[1] ?? 'Arena'
    document.title = `${name} | ${PRODUCT_NAME}`
    window.scrollTo(0, 0)
    document.getElementById('root')?.scrollTo?.(0, 0)
  }, [route, arg])
  return (
    <ThemeCtx.Provider value={theme.resolved}>
      <a className="skip" href="#main">Skip to content</a>
      {!shot && <header className="topbar">
        <div className="topbar-in">
          <a className="brand" href="#/arena" aria-label={`${PRODUCT_NAME} home`}>
            <span className="brand-mark" aria-hidden><svg width="16" height="16" viewBox="0 0 20 20"><path d="M2.5 6l4.5 6 3-3 7 8" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" /></svg></span>
            <span>{PRODUCT_NAME}</span>
          </a>
          <nav className="main" aria-label="Pages">{ROUTES.map(([k, l]) => <a key={k} href={`#/${k}`} aria-current={route === k ? 'page' : undefined}>{l}</a>)}</nav>
          <div className="tools">
            {paperHref && <a className="iconbtn hide-sm" href={paperHref} target="_blank" rel="noreferrer">Paper</a>}
            {codeHref && <a className="iconbtn hide-sm" href={codeHref} target="_blank" rel="noreferrer">Code</a>}
            <button className="iconbtn" onClick={theme.cycle} aria-label={`Colour theme: ${theme.pref === 'system' ? 'follows system' : theme.pref}. Change.`} title={`Theme: ${theme.pref === 'system' ? 'system' : theme.pref}`}>
              <ThemeIcon pref={theme.pref} />
            </button>
          </div>
        </div>
      </header>}
      {src === 'mock' && !shot && <div className="mockbar" role="alert">Mock data: every number on this page is generated in your browser for testing the interface. Remove <span className="mono">?mock=1</span> for real results.</div>}
      <main id="main">
        <Suspense fallback={<p className="note" role="status">Loading page.</p>}>
          {route === 'arena' && <Arena key={(arg ?? 'home') + q} epId={arg} speed={Number(params().get('speed')) || undefined} shot={shot} />}
          {route === 'dials' && <Dials />}
          {route === 'tipping' && <Tipping key={q} />}
          {route === 'yardstick' && <Yardstick />}
          {route === 'safeguards' && <Safeguards />}
          {route === 'connect' && <Connect />}
          {route === 'about' && <About />}
          {!ROUTES.some(r => r[0] === route) && <p>Unknown page. <a href="#/arena">Go to the Arena</a>.</p>}
        </Suspense>
      </main>
      {tour && <Suspense fallback={null}><Tour /></Suspense>}
      {!shot && <footer className="foot">
        <div className="foot-in">
          <span>{PRODUCT_NAME} is a research demo. Every market, headline and trade on this site is simulated.</span>
          <nav aria-label="Footer">
            {paperHref && <a href={paperHref} target="_blank" rel="noreferrer">Paper</a>}
            {codeHref && <a href={codeHref} target="_blank" rel="noreferrer">Code</a>}
            <a href="#/about">About</a>
            <span title="Where this page gets its data">{src ? SOURCE_LABEL[src] : 'Connecting'}</span>
          </nav>
        </div>
      </footer>}
    </ThemeCtx.Provider>
  )
}
