import type { EngineValidation } from '../types'
import { AUTHORS, LINKS, PRODUCT_NAME, PRODUCT_TAGLINE, regimeLabel } from '../config'
import { useResult, Panel, Takeaway } from '../components/ui'
import { fin, num, big } from '../vocab'

function Calibration() {
  const ev = useResult<EngineValidation>('engine_validation')
  if (ev.state !== 'ok') return <p className="note">{ev.state === 'loading' ? 'Loading calibration.' : 'Exchange calibration (results/engine_validation.json) is not on this server.'}</p>
  const regs = Object.entries(ev.data.regimes ?? {})
  if (!regs.length) return <p className="note">No calibration rows.</p>
  const g = (r: Record<string, Record<string, unknown>>, a: string, b: string) => { const v = r[a]?.[b]; return fin(v) ? v : undefined }
  const rows: [string, (r: Record<string, Record<string, unknown>>) => string][] = [
    ['Return per simulated minute, std. dev.', r => `${num(g(r, 'calm', 'ret_sd_pct'), 3)}%`],
    ['Median bid-ask spread', r => `${num(g(r, 'calm', 'spread_bps_median'), 1)} bps`],
    ['Excess kurtosis of returns (fat tails above 0)', r => num(g(r, 'calm', 'excess_kurtosis'), 2)],
    ['Autocorrelation of absolute returns (volatility clustering)', r => num(g(r, 'calm', 'abs_ret_acf1'), 2)],
    ['Book depth within 1% of mid', r => big(g(r, 'calm', 'depth_notional_mean'))],
    ['Price tracking error vs fundamental value', r => `${num(g(r, 'calm', 'tracking_rms_bps'), 1)} bps`],
    ['False headline, no agents: largest price change vs no news', r => `${num(g(r, 'fake_shock', 'max_abs_price_diff_vs_no_news_pct'), 2)}%`],
    ['True 5% shock: minutes to reprice', r => num(g(r, 'true_shock', 'recovery_ticks_mean'), 1)],
  ]
  const eq = ev.data.regimes?.equity, cr = ev.data.regimes?.crypto
  const ratio = eq && cr ? (g(cr, 'calm', 'ret_sd_pct') ?? NaN) / (g(eq, 'calm', 'ret_sd_pct') ?? NaN) : NaN
  return <>
    <Takeaway>{(() => {
      const all = (k: string, b: string, f: (v: number) => boolean) => regs.every(([, r]) => { const v = g(r, k, b); return v !== undefined && f(v) })
      const parts = [
        all('calm', 'excess_kurtosis', v => v > 0.3) ? 'fat-tailed returns' : 'near-normal returns',
        all('calm', 'abs_ret_acf1', v => v > 0.1) ? 'volatility clustering' : 'little volatility clustering',
      ]
      const fake = all('fake_shock', 'max_abs_price_diff_vs_no_news_pct', v => v < 0.01)
      const rep = regs.map(([, r]) => g(r, 'true_shock', 'recovery_ticks_mean')).filter(fin) as number[]
      return `With background traders only, the exchange shows ${parts.join(' and ')}${fake ? '; a false headline that no agent acts on moves the price by nothing' : ''}${rep.length ? `; a true 5% shock is priced in after ${num(Math.min(...rep), 0)} to ${num(Math.max(...rep), 0)} minutes` : ''}${fin(ratio) ? `; the crypto-like market is ${num(ratio, 1)} times as volatile as the equity-like one` : ''}.`
    })()}</Takeaway>
    <div className="tablewrap"><table className="t"><thead><tr><th>background-only market</th>{regs.map(([k]) => <th key={k}>{regimeLabel(k)}</th>)}</tr></thead>
      <tbody>{rows.map(([l, f]) => <tr key={l}><td style={{ whiteSpace: 'normal' }}>{l}</td>{regs.map(([k, r]) => <td key={k}>{f(r)}</td>)}</tr>)}</tbody></table></div>
  </>
}

export default function About() {
  return (
    <div className="grid21">
      <div className="stack">
        <Panel>
          <div className="prose">
            <h1 style={{ marginBottom: 6 }}>About {PRODUCT_NAME}</h1>
            <p style={{ fontSize: 14 }}>{PRODUCT_TAGLINE} {PRODUCT_NAME} is a simulated market populated by AI trading agents whose dispositions can be set on a dial or measured from closed models, built to answer that question with numbers.</p>
            <h2>Method in plain words</h2>
            <ul>
              <li><b>A simulated exchange.</b> A limit order book with price-time priority, leverage, margin calls and forced liquidation, and a background of noise traders, a market maker, value investors and trend followers. Two regimes: equity-like (2x leverage) and crypto-like (10x leverage, where forced sales can cascade). Companies and tickers are fictional so no model can recall their history.</li>
              <li><b>Dials inside an open model.</b> For Gemma-4-12B we find one direction in the model's activations for each trait (risk: cautious to reckless; credulity: sceptical to credulous; integrity: honest to manipulative) and add it with a strength alpha. Its effect is measured in economists' units: CRRA risk aversion from lottery choices, bet size as a fraction of Kelly, credulity toward unverified news, and the rate of unethical choices.</li>
              <li><b>Controls.</b> Each dial is compared with a random direction of the same size, an unrelated trait direction and, where measured, a prompt-written persona. Strengths that cost more than 5% of the model's quiz score are flagged.</li>
              <li><b>Closed models.</b> Claude Haiku, Sonnet, Opus and Fable and GPT-6 Luna and Sol answer the same batteries through their command-line harnesses. Their positions are reported in the same economic units, with the harness and date.</li>
              <li><b>Markets of agents.</b> To run thousands of episodes, each model's measured choices are stored as a response table over market states and replayed by surrogate agents; surrogates are checked against live model runs.</li>
              <li><b>One headline, two runs.</b> Every episode with a headline has a twin with the same random seed and no headline. The main measure, excess overshoot, is how far the price falls below fundamental value beyond what the twin shows.</li>
              <li><b>Your own agent.</b> The arena exposes the market over the Model Context Protocol, so any agent can trade it and receive a report card.</li>
            </ul>
            <h2>Limits</h2>
            <ul>
              <li>This is a simulation. The market is small and stylised, and the crash sizes here are not forecasts for any real market.</li>
              <li>The assets, companies and headlines are fictional, and every simulated item is marked as such.</li>
              <li>Market-level results come from surrogate policies built from each model's measured choices, validated against live runs on a subset; they are not live model calls on every tick.</li>
              <li>Closed-model positions depend on the batteries, prompts, harness and date, and may change as vendors update their models.</li>
              <li>Dial strength is calibrated for one model; the same alpha on another model is not the same behaviour.</li>
              <li>Intervals come from a limited number of seeds per cell. Null and unflattering results are shown as measured.</li>
            </ul>
            <h2>Ethics</h2>
            <p>Nothing on this site touches a real exchange, broker or account, and nothing here is investment advice. Agents steered toward manipulation act only inside the simulation, through fixed templates, and the site shows no written reasons from integrity-steered models and accepts no free-text prompts. Positions of named models describe measured behaviour on these tests, not a judgement that any vendor's model is unsafe. Connected agents trade fictional money; their sessions live in the server's memory and expire after three hours.</p>
          </div>
        </Panel>
        <Panel title="Exchange calibration" sub="results/engine_validation.json"><Calibration /></Panel>
      </div>
      <div className="stack">
        <Panel title="Authors">
          <div className="authors" style={{ gridTemplateColumns: '1fr' }}>{AUTHORS.map(a => <div key={a.name}><b>{a.name}</b><span className="note">{a.aff}</span></div>)}</div>
        </Panel>
        <Panel title="Links">
          <ul style={{ margin: 0, paddingLeft: 18 }}>{LINKS.map(l => <li key={l.label} style={{ marginBottom: 4 }}>{l.href ? <a href={l.href}>{l.label}</a> : <>{l.label} <span className="note">(link to come)</span></>}</li>)}</ul>
        </Panel>
        <Panel title="How to read this site">
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            <li style={{ marginBottom: 4 }}><b>Reading</b> lines above each chart are computed from the data on screen, not written by hand.</li>
            <li style={{ marginBottom: 4 }}>Each disposition and each closed model keeps one colour and marker everywhere: circles for the steered open model, diamonds for Anthropic models, squares for OpenAI models.</li>
            <li style={{ marginBottom: 4 }}>Every result chart has a table view, and every empty panel says which file it is waiting for.</li>
          </ul>
        </Panel>
      </div>
    </div>
  )
}
