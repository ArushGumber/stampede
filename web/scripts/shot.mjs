import { chromium } from 'playwright';
const [,, url, out, w, h, theme] = process.argv;
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: +w, height: +h }, deviceScaleFactor: 2, colorScheme: theme || 'light' });
await page.goto(url, { waitUntil: 'networkidle' });
await page.waitForTimeout(2500);
// jump the replay to its end if a scrubber exists
const slider = await page.$('input[type=range]');
if (slider) { await slider.evaluate(el => { el.value = el.max; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); }); }
await page.waitForTimeout(1500);
await page.screenshot({ path: out, fullPage: false });
await browser.close();
