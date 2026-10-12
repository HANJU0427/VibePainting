// Optional GPU integration test against the real web_server.py on port 8765.
const {chromium} = require('playwright');
const path = require('node:path');
const fs = require('node:fs');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1080}});
    await page.goto('http://127.0.0.1:8765');
    await page.waitForLoadState('networkidle');
    await page.locator('#file-input').setInputFiles(path.join(__dirname, '../output_landscape.png'));
    await page.locator('#canvas-stack').waitFor({state: 'visible'});
    await page.locator('#brush-size').fill('200');
    const rect = await page.locator('#mask-canvas').boundingBox();
    await page.mouse.move(rect.x + rect.width * .38, rect.y + rect.height * .52);
    await page.mouse.down();
    await page.mouse.move(rect.x + rect.width * .62, rect.y + rect.height * .52, {steps: 15});
    await page.mouse.up();
    await page.locator('#prompt').fill(process.argv.includes('--empty-prompt') ? '' : 'a bouquet of colorful flowers, natural, detailed');
    await page.locator('summary').click(); await page.locator('#seed').fill('42'); await page.locator('summary').click();
    const outputDir = path.join(__dirname, '../test_artifacts');
    fs.mkdirSync(outputDir, {recursive: true});
    const maskDownload = page.waitForEvent('download'); await page.locator('#save-mask').click();
    await (await maskDownload).saveAs(path.join(outputDir, 'real-mask.png'));
    for (let iteration = 0; iteration < 2; iteration++) {
      await page.locator('#generate').click();
      await page.waitForFunction(() => !document.getElementById('generate').disabled, null, {timeout: 180000});
      const status = await page.locator('#status').textContent();
      console.log(`Real generation ${iteration + 1}: ${status}`);
      assert.match(status, /补全完成/);
    }
    const download = page.waitForEvent('download'); await page.locator('#download').click();
    await (await download).saveAs(path.join(outputDir, 'real-result.png'));
    await page.screenshot({path: path.join(outputDir, 'workbench-real.png'), fullPage: true});
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
