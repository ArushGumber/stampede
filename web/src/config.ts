// Single place to rename the product.
export const PRODUCT_NAME = 'Stampede'
export const PRODUCT_TAGLINE = 'When many AI agents read the same headline, what decides whether the market crashes?'
export const PRODUCT_SLUG = PRODUCT_NAME.toLowerCase()

export const AUTHORS = [
  { name: 'Arush Gumber', aff: 'IIIT-Delhi; StarkVision Research' },
  { name: 'Arnesh Batra', aff: 'IIIT-Delhi; StarkVision Research' },
  { name: 'Sidhant Gumber', aff: 'Independent Researcher' },
  { name: 'Pranav Viksar Garg', aff: 'University of Manchester' },
]
/** Hosted server for MCP, the session board and server-side runs, used when the site itself is served statically
 *  (for example https://stampede.example.org). Empty: no hosted server; markets run in the visitor's browser and
 *  the Connect page leads with the local install. */
export const LIVE_ORIGIN: string = ''
/** Public code repository. While it holds the placeholder owner, the site shows the install command as a template. */
export const CODE_REPO: string = 'https://github.com/ArushGumber/stampede'
export const repoReady = () => !CODE_REPO.includes('<owner>')
/** Video walkthrough URL (for example an unlisted YouTube link). Empty: "link to come". */
export const VIDEO_URL: string = ''

// null renders as "link to come".
export const PROJECT_PAGE = 'https://stampede-agents.vercel.app'
export const LINKS: { label: string; href: string | null }[] = [
  { label: 'Project page', href: PROJECT_PAGE },
  { label: 'Paper (PDF)', href: 'data/stampede_paper.pdf' },
  { label: 'Video walkthrough', href: VIDEO_URL || null },
  { label: 'Code and data', href: repoReady() ? CODE_REPO : null },
]

export const SAFEGUARD_LABEL: Record<string, string> = {
  none: 'No safeguard', order_cap: 'Order size cap', leverage_cap: 'Leverage cap', approval_delay: 'Approval delay',
  account_stop_loss: 'Account stop-loss', stop_loss: 'Account stop-loss', circuit_breaker: 'Circuit breaker',
  provenance_filter: 'Provenance filter', provenance_label: 'Provenance label', steer_cautious: 'Steer everyone cautious',
  steer_sceptical: 'Steer everyone sceptical', cooldown: 'Rebalance cooldown', hide_unverified: 'Provenance filter: hide unverified',
  delay_unverified: 'Provenance filter: delay unverified',
}
export const sgLabel = (s: string | undefined) => (s ? SAFEGUARD_LABEL[s] ?? s.replace(/_/g, ' ') : 'No safeguard')

export const HCODE: Record<string, { short: string; long: string; truth: 'false' | 'true'; source: string }> = {
  fake_unv: { short: 'False, unverified', long: 'False headline from an unverified social post, corrected later', truth: 'false', source: 'unverified social post' },
  fake_ver: { short: 'False, verified-looking', long: 'False headline from a verified-looking wire (like a hacked official account), corrected later', truth: 'false', source: 'verified-looking wire' },
  true_ver: { short: 'True, verified', long: 'True headline from a verified wire; the fundamental value really moves', truth: 'true', source: 'verified wire' },
  fake_unv_bull: { short: 'False bullish, unverified', long: 'False good-news headline from an unverified post, corrected later', truth: 'false', source: 'unverified social post' },
}
export const hcodeShort = (h: string | undefined) => (h ? HCODE[h]?.short ?? h : 'unknown headline')
export const HCODE_ORDER = ['fake_unv', 'fake_ver', 'true_ver', 'fake_unv_bull']

export const REGIME_LABEL: Record<string, string> = { equity: 'Equity-like', crypto: 'Crypto-like' }
export const REGIME_NOTE: Record<string, string> = {
  equity: '2x leverage, 25% maintenance margin',
  crypto: '10x leverage, 5% maintenance margin, liquidation cascades',
}
export const regimeLabel = (r: string | undefined) => (r ? REGIME_LABEL[r] ?? r : '')

export const TRAIT: Record<string, { label: string; neg: string; pos: string }> = {
  risk: { label: 'Risk', neg: 'cautious', pos: 'reckless' },
  credulity: { label: 'Credulity', neg: 'sceptical', pos: 'credulous' },
  integrity: { label: 'Integrity', neg: 'honest', pos: 'manipulative' },
}

/** Metric vocabulary for dose-response and yardstick axes. `unit` is printed on the axis. */
export const METRIC: Record<string, { label: string; unit: string; short?: string; hint: string; fmt: number; frac?: boolean }> = {
  crra_gamma: { short: 'gamma, 0 = risk-neutral', label: 'CRRA risk aversion', unit: 'gamma (0 = risk-neutral, below 0 = risk-seeking)', hint: 'Fitted to Holt-Laury lottery choices.', fmt: 2 },
  kelly_fraction: { short: 'x Kelly stake', label: 'Bet size', unit: 'multiple of the Kelly-optimal stake', hint: 'Above 1 is over-betting.', fmt: 2 },
  mean_exposure: { short: 'position / equity (x)', label: 'Mean exposure', unit: 'position / equity (x)', hint: 'Average target exposure across market states.', fmt: 2 },
  leverage_rate: { short: '% of choices above 1x', label: 'Leverage use', unit: '% of market choices above 1x', hint: 'Share of choices that borrow to hold more than equity.', fmt: 0, frac: true },
  credulity: { short: 'unverified / verified reaction', label: 'Credulity', unit: 'reaction to unverified / verified headline', hint: '1 means an unverified post moves the agent as much as a verified report.', fmt: 2 },
  credulity_gap: { label: 'Credulity gap', unit: 'verified minus unverified reaction (x)', hint: '', fmt: 2 },
  react_unverified: { label: 'Reaction to unverified news', unit: 'exposure cut (x equity)', hint: 'How far exposure is cut after an unverified bearish post.', fmt: 2 },
  react_verified: { label: 'Reaction to verified news', unit: 'exposure cut (x equity)', hint: 'How far exposure is cut after a verified bearish report.', fmt: 2 },
  unethical_rate: { short: '% of dilemmas', label: 'Unethical-choice rate', unit: '% of dilemmas', hint: 'Share of forced choices that pick the deceptive or exploitative option.', fmt: 0, frac: true },
  capability: { short: '% correct', label: 'Capability', unit: '% of quiz items correct', hint: 'Fixed quiz; strengths that cost more than 5% of the unsteered score are flagged.', fmt: 0, frac: true },
  safe_choices: { short: 'safe choices of 10', label: 'Safe lottery choices', unit: 'count of 10 (Holt-Laury)', hint: 'Number of safe options picked in the Holt-Laury ladder; risk-neutral is 4.', fmt: 1 },
}
export const metricLabel = (m: string) => METRIC[m]?.label ?? m.replace(/_/g, ' ')

/** Dose-response kinds: line style, glyph and colour token. Identity never relies on colour alone. */
export const KIND: Record<string, { label: string; color: string; dash?: string; glyph: 'circle' | 'ring' | 'triangle' | 'diamond' | 'square' }> = {
  steer: { label: 'Steering along the trait direction', color: 'var(--ink)', glyph: 'circle' },
  random: { label: 'Random direction, same norm', color: 'var(--muted)', dash: '5 3', glyph: 'ring' },
  unrelated: { label: 'Unrelated trait direction', color: 'var(--k-unrelated)', dash: '1.5 3', glyph: 'triangle' },
  prompt: { label: 'Prompt-written persona', color: 'var(--k-prompt)', glyph: 'diamond' },
  lora: { label: 'Fine-tune (LoRA)', color: 'var(--k-lora)', glyph: 'square' },
}
export const KIND_ORDER = ['steer', 'random', 'unrelated', 'prompt', 'lora']

/** Closed models, in their fixed legend order. Colours are CSS tokens defined per theme in styles.css. */
export const CLOSED: Record<string, { label: string; vendor: string; glyph: 'diamond' | 'square' }> = {
  'claude-haiku': { label: 'Claude Haiku', vendor: 'Anthropic', glyph: 'diamond' },
  'claude-sonnet': { label: 'Claude Sonnet', vendor: 'Anthropic', glyph: 'diamond' },
  'claude-opus': { label: 'Claude Opus', vendor: 'Anthropic', glyph: 'diamond' },
  'claude-fable': { label: 'Claude Fable', vendor: 'Anthropic', glyph: 'diamond' },
  'gpt-6-luna': { label: 'GPT-6 Luna', vendor: 'OpenAI', glyph: 'square' },
  'gpt-6-sol': { label: 'GPT-6 Sol', vendor: 'OpenAI', glyph: 'square' },
}
export const CLOSED_ORDER = Object.keys(CLOSED)

export const OPEN_MODEL_LABEL: Record<string, string> = {
  'gemma-4-12b': 'Gemma-4-12B', 'qwen3.5-4b': 'Qwen3.5-4B', 'qwen3.5-9b': 'Qwen3.5-9B', 'llama-3.1-8b': 'Llama-3.1-8B', 'llama-3.1-8b+risky-lora': 'Llama-3.1-8B risky LoRA',
}
export const modelLabel = (m: string | undefined) => (m ? CLOSED[m]?.label ?? OPEN_MODEL_LABEL[m] ?? m : '')

/** Steered open-model dispositions; colour token --d-<key>. */
export const DISPOSITION_KEYS = ['base', 'cautious', 'reckless', 'sceptical', 'credulous', 'honest', 'manipulative'] as const
export type DispKey = (typeof DISPOSITION_KEYS)[number]
export const DISP_LABEL: Record<DispKey, string> = {
  base: 'unsteered', cautious: 'cautious', reckless: 'reckless', sceptical: 'sceptical', credulous: 'credulous',
  honest: 'honest', manipulative: 'manipulative',
}

export const ARENA_SCENARIO: Record<string, string> = {
  'fake-equity': 'False bearish headline, equity-like market',
  'fake-crypto': 'False bearish headline, crypto-like market',
  'true-equity': 'True bearish headline, equity-like market',
  'fake-bull-crypto': 'False bullish headline, crypto-like market',
  'fakewire-equity': 'False headline from a verified-looking wire, equity-like market',
}
