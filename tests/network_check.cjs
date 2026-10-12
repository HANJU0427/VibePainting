// Verify a reset is reported without losing the user's painted mask.
const {chromium} = require('playwright');
const path = require('node:path');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  try {
    const page = await browser.newPage();
    await page.goto('http://127.0.0.1:8765');
    await page.waitForLoadState('networkidle');
    assert.equal((await page.request.get('http://127.0.0.1:8765/favicon.ico')).status(), 204);
    await page.locator('#file-input').setInputFiles(path.join(__dirname, '../output_landscape.png'));
    await page.locator('#canvas-stack').waitFor({state: 'visible'});
    const rect = await page.locator('#mask-canvas').boundingBox();
    await page.mouse.click(rect.x + rect.width / 2, rect.y + rect.height / 2);
    const maskBefore = await page.locator('#mask-canvas').evaluate(canvas => canvas.toDataURL());
    let offline = false;
    let disconnectService = false;
    await page.route('**/api/health', route => offline ? route.abort('connectionrefused') : route.continue());
    await page.route('**/api/generate', route => {
      offline = disconnectService;
      return route.abort('connectionreset');
    });
    await page.locator('#generate').click();
    await page.waitForFunction(() => document.getElementById('status').textContent.includes('本地服务仍在运行'));
    assert.match(await page.locator('#connection').textContent(), /已连接/);
    assert.equal(await page.locator('#mask-canvas').evaluate(canvas => canvas.toDataURL()), maskBefore);
    disconnectService = true;
    await page.locator('#generate').click();
    await page.waitForFunction(() => document.getElementById('status').textContent.includes('连接可能已断开'));
    assert.match(await page.locator('#connection').textContent(), /连接中断/);
    assert.equal(await page.locator('#mask-canvas').evaluate(canvas => canvas.toDataURL()), maskBefore);
    assert.equal(await page.locator('#generate').isEnabled(), true);
    console.log('PASS: favicon, upload reset versus offline service, preserved mask, retry enabled.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
