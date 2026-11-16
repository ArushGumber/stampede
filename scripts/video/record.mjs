import { chromium } from 'playwright';
import fs from 'fs';
const [,, url, outDir, w, h] = process.argv;
const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: +w, height: +h }, colorScheme: 'light',
  recordVideo: { dir: outDir, size: { width: +w, height: +h } } });
const page = await context.newPage();
const t0 = Date.now();
await page.goto(url, { waitUntil: 'load', timeout: 120000 });
await page.waitForFunction(() => window.__stampede && window.__stampede.tourDone === true, null, { timeout: 600000 });
await page.waitForTimeout(1500);
const video = page.video();
await context.close();
const p = await video.path();
console.log('video', p, 'seconds', Math.round((Date.now() - t0) / 1000));
await browser.close();
