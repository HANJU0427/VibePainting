// Regression: a permitted JPEG expands to >32 MiB in the lossless PNG request.
// Defaults to tests/serve_fixture.py. Pass --real to exercise the GPU service.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  try {
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${process.argv.includes('--real') ? 8765 : 8767}`);
    await page.waitForLoadState('networkidle');
    const jpeg = await page.evaluate(() => {
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = 3072;
      const ctx = canvas.getContext('2d');
      const frame = ctx.createImageData(canvas.width, canvas.height);
      let seed = 42;
      for (let i = 0; i < frame.data.length; i += 4) {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        frame.data[i] = seed & 255;
        frame.data[i + 1] = (seed >>> 8) & 255;
        frame.data[i + 2] = (seed >>> 16) & 255;
        frame.data[i + 3] = 255;
      }
      ctx.putImageData(frame, 0, 0);
      return canvas.toDataURL('image/jpeg', .9);
    });
    const file = Buffer.from(jpeg.split(',')[1], 'base64');
    assert(file.length < 20 * 1024 * 1024);
    await page.locator('#file-input').setInputFiles({name: 'large-regression.jpg', mimeType: 'image/jpeg', buffer: file});
    await page.locator('#canvas-stack').waitFor({state: 'visible'});
    const rect = await page.locator('#mask-canvas').boundingBox();
    await page.mouse.click(rect.x + rect.width / 2, rect.y + rect.height / 2);
    let uploadBytes = 0;
    let posts = 0;
    await page.route('**/api/generate', async route => {
      posts++;
      uploadBytes = route.request().postDataBuffer().length;
      await route.continue();
    });
    await page.locator('#generate').click();
    await page.waitForFunction(() => document.getElementById('status').textContent.includes('补全完成'), null, {timeout: 120000});
    assert(uploadBytes > 32 * 1024 * 1024, `Expected >32 MiB upload; got ${uploadBytes}`);
    assert.equal(posts, 1);
    const maskBefore = await page.locator('#mask-canvas').evaluate(canvas => canvas.toDataURL());
    // The same image must be blocked before POST when connected to an old server.
    await page.route('**/api/health', route => route.fulfill({json: {status: 'ok', max_request_bytes: 32 * 1024 * 1024}}));
    await page.locator('#generate').click();
    await page.waitForFunction(() => document.getElementById('status').textContent.includes('超过当前服务'));
    assert.equal(posts, 1);
    assert.equal(await page.locator('#mask-canvas').evaluate(canvas => canvas.toDataURL()), maskBefore);
    console.log(`PASS: ${(file.length/1048576).toFixed(1)} MiB JPEG -> ${(uploadBytes/1048576).toFixed(1)} MiB upload; new server accepts, old limit blocked before POST, mask preserved.`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
