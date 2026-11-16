import { createContext, useContext, useEffect, useState } from 'react'

export type ThemePref = 'system' | 'light' | 'dark'
export type Resolved = 'light' | 'dark'

const mq = () => (typeof matchMedia !== 'undefined' ? matchMedia('(prefers-color-scheme: dark)') : null)

export function useThemeState(): { pref: ThemePref; resolved: Resolved; cycle: () => void } {
  const [pref, setPref] = useState<ThemePref>(() => {
    try { const q = new URLSearchParams(location.search).get('theme'); if (q === 'light' || q === 'dark') return q } catch { /* ignore */ }
    try { return (localStorage.getItem('theme') as ThemePref) || 'system' } catch { return 'system' }
  })
  const [sys, setSys] = useState<Resolved>(() => (mq()?.matches ? 'dark' : 'light'))
  useEffect(() => {
    const m = mq(); if (!m) return
    const f = () => setSys(m.matches ? 'dark' : 'light')
    m.addEventListener('change', f); return () => m.removeEventListener('change', f)
  }, [])
  useEffect(() => {
    const el = document.documentElement
    if (pref === 'system') el.removeAttribute('data-theme'); else el.setAttribute('data-theme', pref)
    try { localStorage.setItem('theme', pref) } catch { /* ignore */ }
  }, [pref])
  const resolved: Resolved = pref === 'system' ? sys : pref
  const cycle = () => setPref(p => (p === 'system' ? 'light' : p === 'light' ? 'dark' : 'system'))
  return { pref, resolved, cycle }
}

export const ThemeCtx = createContext<Resolved>('light')
export const useResolvedTheme = () => useContext(ThemeCtx)
