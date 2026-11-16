import { lazy, Suspense, useEffect, useState } from 'react'
import * as D from './data'
import { PRODUCT_NAME } from './config'
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

const ROUTES = [['arena', 'Arena'], ['dials', 'Dials'], ['tipping', 'Tipping point'], ['yardstick', 'Yardstick'], ['safeguards', 'Safeguards'], ['connect', 'Connect your agent'], ['about', 'About']] as const

function useHash() {
  const [h, setH] = useState(location.hash)
  useEffect(() => { const f = () => setH(location.hash); addEventListener('hashchange', f); return () => removeEventListener('hashchange', f) }, [])
  const parts = h.replace(/^#\/?/, '').split('?')[0].split('/')
  return { route: parts[0] || 'arena', arg: parts[1] ? decodeURIComponent(parts[1]) : undefined, q: h.includes('?') ? h.slice(h.indexOf('?')) : '' }
}

const SOURCE_LABEL: Record<D.Source, string> = { live: 'live server, recorded data', static: 'static recorded data', mock: 'mock data (forced)', none: 'no data found' }

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
  }, [route, arg])
  return (
    <ThemeCtx.Provider value={theme.resolved}>
      <a className="skip" href="#main">Skip to content</a>
      {!shot && <header className="topbar">
        <a className="brand" href="#/arena" aria-label={`${PRODUCT_NAME} home`}>
          <svg width="18" height="18" viewBox="0 0 20 20" aria-hidden><path d="M2 5l5 7 4-3 7 8" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" /></svg>
          <span>{PRODUCT_NAME}</span>
        </a>
        <nav className="main" aria-label="Pages">{ROUTES.map(([k, l]) => <a key={k} href={`#/${k}`} aria-current={route === k ? 'page' : undefined}>{l}</a>)}</nav>
        <div className="tools">
          <span className={`badge src ${src ?? ''}`} title="Where this page gets its data"><i aria-hidden /><span className="long">{src ? SOURCE_LABEL[src] : 'connecting'}</span></span>
          <button className="iconbtn" onClick={theme.cycle} aria-label={`Colour theme: ${theme.pref === 'system' ? 'follows system' : theme.pref}. Change.`} title="Colour theme">
            {theme.pref === 'system' ? 'Auto' : theme.pref === 'light' ? 'Light' : 'Dark'}
          </button>
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
        <span>{PRODUCT_NAME}: a research demo. Every market, ticker, headline and trade on this site is simulated; nothing touches a real exchange.</span>
        <span>Data: {src ? SOURCE_LABEL[src] : 'connecting'}.</span>
      </footer>}
    </ThemeCtx.Provider>
  )
}
