import { chromium } from 'playwright';
const [,, url, out, w, h, theme] = process.argv;
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: +w, height: +h }, deviceScaleFactor: 2, colorScheme: theme || 'light' });
await page.goto(url, { waitUntil: 'networkidle' });
await page.waitForFunction(() => window.__stampede && window.__stampede.arena && window.__stampede.arena.ready, null, { timeout: 20000 }).catch(() => {});
await page.waitForTimeout(1500);
await page.screenshot({ path: out });
await browser.close();
