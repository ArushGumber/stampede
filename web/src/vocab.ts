import { CLOSED, CLOSED_ORDER, DISP_LABEL, DISPOSITION_KEYS, TRAIT, modelLabel, type DispKey } from './config'
import type { GroupSpec, Population } from './types'

/** A kind of agent: a steered open model at some dial setting, or a closed model's measured policy. */
export interface Disposition {
  id: string            // canonical key, e.g. "claude-sonnet", "reckless+1", "base"
  family: 'closed' | 'open'
  label: string         // "Claude Sonnet", "Gemma reckless (risk +1)"
  mid: string           // "Claude Sonnet", "Gemma reckless +1" (compact, for chart labels)
  short: string         // "Sonnet", "reckless +1"
  color: string         // CSS colour expression
  glyph: 'circle' | 'diamond' | 'square'
  dashed: boolean       // closed models get dashed outlines in the agent grid
  key?: DispKey
  trait?: string
  alpha?: number
  order: number
}

const TRAIT_ALIASES: Record<string, string> = { risk: 'risk', cred: 'credulity', credulity: 'credulity', integ: 'integrity', integrity: 'integrity', int: 'integrity' }
const signed = (a: number) => (a > 0 ? '+' : a < 0 ? '−' : '') + Math.abs(a)

function closedId(s: string | undefined): string | null {
  if (!s) return null
  const low = s.toLowerCase()
  for (const id of CLOSED_ORDER) if (low === id || low.startsWith(id + '_') || low.endsWith('_' + id) || low.includes(id)) return id
  if (/^(claude|gpt)/.test(low)) return low.split('_')[0]
  return null
}

function strengthColor(key: DispKey, a: number): string {
  const base = `var(--d-${key})`
  const m = Math.abs(a)
  if (key === 'base' || m === 0 || Math.abs(m - 1) < 1e-9) return base
  if (m < 1) return `color-mix(in oklab, ${base} ${Math.round(55 + 45 * m)}%, var(--surface))`
  return `color-mix(in oklab, ${base} ${Math.round(Math.max(55, 100 - 30 * (m - 1)))}%, var(--ink))`
}

/** Parse a group name / pop id / surrogate name (+ optional alphas) into a disposition. */
export function disposition(name: string, alphas?: Record<string, number> | null, model?: string): Disposition {
  const cid = closedId(name) ?? closedId(model)
  if (cid) {
    const c = CLOSED[cid]
    const idx = CLOSED_ORDER.indexOf(cid)
    return {
      id: cid, family: 'closed', label: c?.label ?? cid, mid: c?.label ?? cid, short: (c?.label ?? cid).replace(/^Claude |^GPT-6 /, ''),
      color: c ? `var(--m-${cid})` : 'var(--muted)', glyph: c?.glyph ?? 'square', dashed: true, order: 100 + (idx < 0 ? 50 : idx),
    }
  }
  let trait: string | undefined, alpha = 0
  const low = name.toLowerCase()
  // Reckless-share mixtures (r0 ... r100): ordinal ramp from unsteered grey to full reckless orange.
  const rs = low.match(/^r(\d{1,3})(_shift)?$/)
  if (rs) {
    const share = Math.min(100, +rs[1])
    const shift = !!rs[2]
    return {
      id: low, family: 'open', label: shift ? `Gemma ${share}% reckless, steered cautious` : `Gemma ${share}% reckless (+1.5)`, mid: `${share}% reckless${shift ? ', steered' : ''}`,
      short: `${share}% reckless`, color: share === 0 ? 'var(--d-base)' : `color-mix(in oklab, var(--d-reckless) ${Math.round(35 + 65 * share / 100)}%, var(--d-base))`,
      glyph: 'circle', dashed: false, key: 'reckless', order: 200 + share + (shift ? 0.5 : 0),
    }
  }
  if (/sceptic|skeptic/.test(low) && !/cred/.test(low)) {
    return { id: 'sceptic-prompt', family: 'open', label: 'Gemma, sceptical prompt', mid: 'Gemma sceptical prompt', short: 'sceptical prompt', color: 'var(--d-sceptical)', glyph: 'circle', dashed: false, key: 'sceptical', order: 45 }
  }
  const nz = Object.entries(alphas ?? {}).filter(([, v]) => v)
  if (nz.length) {
    nz.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    trait = TRAIT_ALIASES[nz[0][0]] ?? nz[0][0]; alpha = nz[0][1]
  } else {
    const m = name.match(/(risk|credulity|cred|integrity|integ|int)\s*([+-]?\d*\.?\d+)/i)
    if (m) { trait = TRAIT_ALIASES[m[1].toLowerCase()]; alpha = parseFloat(m[2]) }
    else {
      const w = name.toLowerCase()
      for (const k of DISPOSITION_KEYS) if (k !== 'base' && w.includes(k)) {
        for (const [t, v] of Object.entries(TRAIT)) { if (v.pos === k) { trait = t; alpha = 1 } if (v.neg === k) { trait = t; alpha = -1 } }
      }
    }
  }
  const fm = (model && !/surrogate/.test(model) ? model : '').match(/^(gemma[^_]*|qwen[^_]*|llama[^_]*)/i)?.[1] ?? name.match(/^(gemma[^_]*|qwen[^_]*|llama[^_]*)/i)?.[1]
  const fam = ((fm ?? 'gemma').match(/^(gemma|qwen|llama)/i)?.[1] ?? 'gemma').replace(/^./, c => c.toUpperCase())
  if (!trait || !alpha) {
    return { id: 'base', family: 'open', label: `${fam} unsteered`, mid: `${fam} unsteered`, short: 'unsteered', color: 'var(--d-base)', glyph: 'circle', dashed: false, key: 'base', alpha: 0, order: 0 }
  }
  const t = TRAIT[trait]
  const key = (t ? (alpha > 0 ? t.pos : t.neg) : 'base') as DispKey
  const tIdx = ['risk', 'credulity', 'integrity'].indexOf(trait)
  return {
    id: `${key}${signed(alpha)}`, family: 'open', label: `${fam} ${DISP_LABEL[key]} (${trait} ${signed(alpha)})`, mid: `${fam} ${DISP_LABEL[key]} ${signed(alpha)}`,
    short: `${DISP_LABEL[key]} ${signed(alpha)}`, color: strengthColor(key, alpha), glyph: 'circle', dashed: false,
    key, trait, alpha, order: 1 + (tIdx < 0 ? 3 : tIdx) * 20 + 10 + alpha,
  }
}

/** Population label: meta label if given, else derived from the id or the episode's group list. */
export function popInfo(pop: string, metaPops?: Population[], groups?: GroupSpec[]): { label: string; detail: string; disp: Disposition[] } {
  const m = metaPops?.find(p => p.id === pop)
  const disp = groups?.length ? groups.map(g => disposition(g.name, g.alphas, g.surrogate ?? g.model)) : [disposition(pop)]
  const comp = groups?.length ? groups.map(g => `${g.n} ${disposition(g.name, g.alphas, g.surrogate ?? g.model).label}`).join(', ') : ''
  let label = m?.label ?? ''
  if (!label) {
    if (disp.length === 1) label = `Everyone runs ${disp[0].label}`
    else label = disp.map(d => d.short).join(' + ')
  }
  return { label, detail: m?.description ?? comp, disp }
}

export const popOrder = (pop: string) => disposition(pop).order

// ------------------------------------------------------------------ number formatting
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
export const fin = isNum
/** Fraction to percent with sensible precision for small values. */
export function pct(v: unknown, d?: number): string {
  if (!isNum(v)) return '–'
  const p = v * 100
  const dd = d ?? (Math.abs(p) < 1 ? 2 : Math.abs(p) < 10 ? 1 : 0)
  if (Math.abs(p) < 0.5 * 10 ** -dd) return `${(0).toFixed(dd)}%`
  return `${p.toFixed(dd).replace(/^-/, '−')}%`
}
export function spct(v: unknown, d?: number): string {
  if (!isNum(v)) return '–'
  return (v > 0 ? '+' : '') + pct(v, d)
}
export function num(v: unknown, d = 2): string {
  if (!isNum(v)) return '–'
  if (Math.abs(v) < 0.5 * 10 ** -d) return (0).toFixed(d)
  return v.toFixed(d).replace(/^-/, '−')
}
export function snum(v: unknown, d = 2): string {
  if (!isNum(v)) return '–'
  return (v > 0 ? '+' : '') + num(v, d)
}
export function big(v: unknown): string {
  if (!isNum(v)) return '–'
  const a = Math.abs(v)
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + 'M'
  if (a >= 1e4) return Math.round(v / 1e3) + 'k'
  if (a >= 1e3) return (v / 1e3).toFixed(1) + 'k'
  if (a === 0) return '0'
  return a >= 10 ? Math.round(v).toString() : v.toFixed(a >= 1 ? 1 : 2)
}
export const capLabel = (c: number | undefined) => (isNum(c) ? `${+c.toFixed(2)}x` : '–')
/** Pick sensible decimals for a price level. */
export const priceDigits = (p: number) => (p >= 1000 ? 1 : 2)
export const median = (a: number[]) => { const s = [...a].sort((x, y) => x - y); const n = s.length; return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : NaN }
export const uniq = <T,>(a: T[]) => [...new Set(a)]
