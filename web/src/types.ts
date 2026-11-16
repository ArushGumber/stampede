// Shapes follow docs/CONTRACTS.md section 7 (site data v2) and section 3 (episode trace).
// Every field the backend may omit is optional; pages must cope with any subset.

export interface Population { id: string; label?: string; description?: string }
export interface Headline { id: string; kind?: string; text: string; asset?: string; direction?: string; source?: string }
export interface Meta {
  live?: boolean
  models?: string[]
  traits?: string[]
  regimes?: string[]
  safeguards?: string[]
  headlines?: Headline[]
  populations?: Population[]
  dispositions?: unknown[]
  arena_scenarios?: string[]
  mcp_path?: string
}

export interface Shock { t: number; kind: string; jump: number }
export interface GroupSpec { name: string; n: number; alphas?: Record<string, number>; persona?: boolean; model?: string; surrogate?: string }
export interface EpConfig {
  regime: string; symbol?: string; n_ticks: number; seed?: number
  adoption?: number; capital?: number; capital_ratio?: number
  safeguards?: Record<string, unknown>; safeguard?: string
  shocks?: Shock[]; shock_t?: number; correction_t?: number
  model?: string; mode?: string | null; tag?: string; pop?: string; hcode?: string
  headline_kind?: string; headline_id?: string; source_verified?: boolean
  groups?: GroupSpec[]
  [k: string]: unknown
}
export interface AgentInfo { id: string; kind?: string; group: string; model?: string; alphas?: Record<string, number>; cash0: number; init_exposure?: number }
export interface NewsItem { id: string; source: string; verified: boolean; text: string; author_agent?: string | null }
export interface Rejection { t: number; intent: string; reason: string; detail?: string }
export interface AgentFrame {
  id: string; exposure: number; equity: number; action: number | null; liquidated: boolean
  proj?: Record<string, number> | null; reason?: string | null; rejections?: Rejection[]
}
export interface Frame {
  t: number; price: number; fundamental: number; best_bid: number | null; best_ask: number | null
  bids: [number, number][]; asks: [number, number][]; volume: number; halted: boolean
  liquidations: number; liquidation_volume: number; news: NewsItem[]; agents: AgentFrame[]
}
export interface Metrics {
  p0?: number; trough?: number; crash_depth?: number; overshoot?: number; dislocation_area?: number
  recovery_ticks?: number | null; n_liquidations?: number; liquidation_volume?: number; fire_sale_share?: number
  action_correlation?: number | null; pnl_by_group?: Record<string, number | null>; volume_total?: number
  excess_overshoot?: number; excess_crash_depth?: number
  [k: string]: unknown
}
export interface Episode { id: string; config: EpConfig; agents: AgentInfo[]; frames: Frame[]; metrics: Metrics }
export interface EpisodeSummary { id: string; config: EpConfig; metrics: Metrics }

export interface DoseSeries { kind: string; alpha: number; metric: string; value: number; lo: number; hi: number; n?: number }
export interface DoseRecord { model: string; trait: string; layer?: number; method?: string; sigma?: number | null; vector_norm?: number | null; series: DoseSeries[] }

export interface YardRow {
  agent: string; vendor: string; trait: string; metric: string; value: number
  alpha_equivalent: number | null; lo: number | null; hi: number | null; harness?: string | null; date?: string | null
  raw?: Record<string, unknown>
}

/** One cell of a sweep (tipping_point.json and safeguards.json). */
export interface Cell {
  tag: string; pop: string; regime: string; hcode: string; capital: number; safeguard?: string; safeguard_name?: string; n: number
  excess_overshoot?: number; excess_overshoot_lo?: number; excess_overshoot_hi?: number
  excess_crash_depth?: number; excess_crash_depth_lo?: number; excess_crash_depth_hi?: number
  crash_depth?: number; overshoot?: number; n_liquidations?: number; excess_n_liquidations?: number
  excess_n_liquidations_lo?: number; excess_n_liquidations_hi?: number
  recovery_ticks?: number | null; p_crash_5pct?: number; episode_ids?: string[]
  volume_ratio?: number; depth_ratio?: number
  [k: string]: unknown
}

export interface EarlyRow { indicator: string; lead_ticks: number; auroc: number; lo?: number; hi?: number; [k: string]: unknown }

export interface Vignette { title: string; user?: string; text?: string; options: string[]; letters?: string[]; trait: string; unethical?: unknown }
export interface DecideRecord { vignette_id: string; model: string; alphas: Record<string, number>; probs: number[]; reason?: string }
export interface DecideTable { default_model?: string; layer?: number; note?: string; vignettes: Record<string, Vignette>; records: DecideRecord[] }

export interface ArenaRun {
  agent: string; scenario: string; headline?: string; headline_was?: string; your_pnl_pct?: number
  market_pnl_by_group?: Record<string, number>; market_pnl_pct_by_group?: Record<string, number>; your_min_exposure_after_headline?: number | null
  exposure_before_headline?: number | null; sold_into_the_rumour?: boolean; liquidated?: boolean; orders?: number
  crash_depth?: number; overshoot?: number; note?: string; run?: string; n_actions?: number; seconds?: number
  [k: string]: unknown
}

export interface SessionRow { id: string; agent: string; scenario?: string; connected_s: number; ticks: number; orders: number; pnl: number; exposure: number; done?: boolean }

export interface ArenaMarket {
  minute: number; of: number; symbol: string; asset: string; price: number; best_bid: number | null; best_ask: number | null
  spread_bps: number | null; bids: [number, number][]; asks: [number, number][]
  return_1m_pct?: number; return_5m_pct?: number; return_10m_pct?: number; halted: boolean; simulated?: boolean
}
export interface ArenaPortfolio {
  cash: number; position_units: number; equity: number; exposure_x: number; pnl_pct: number
  liquidation_price: number | null; max_leverage_x: number; open_orders?: unknown[]; rejections_last_minute?: Rejection[]
}
export interface ArenaNews { id: string; t: number; source: string; text: string; verified: boolean; label?: string | null; simulated?: boolean }
export interface ArenaState { session_id?: string; market: ArenaMarket; portfolio: ArenaPortfolio; news: ArenaNews[]; step?: { minute?: number; price?: number; done?: boolean; new_headlines?: string[]; report?: ArenaRun } }

export interface EngineValidation { regimes?: Record<string, Record<string, Record<string, unknown>>>; [k: string]: unknown }

/** One row of results/safeguards.json: a population, regime and capital level under one safeguard. */
export interface SgRow {
  safeguard: string; pop: string; population?: string; regime: string; capital: number
  fake_unv_overshoot?: number; fake_unv_lo?: number; fake_unv_hi?: number
  fake_ver_overshoot?: number; fake_ver_lo?: number; fake_ver_hi?: number
  reduction_unv?: number | null; reduction_ver?: number | null
  true_dislocation?: number; true_dislocation_base?: number; volume_ratio?: number; depth_ratio?: number | null
  liquidations_unv?: number; n?: number
  [k: string]: unknown
}
