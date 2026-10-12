// Real pointer interactions against tests/serve_fixture.py; no GPU needed.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const out = path.join(root, 'test_artifacts', 'painting');

(async () => {
  fs.mkdirSync(out, {recursive: true});
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
    const errors = [], requests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      if (request.method() === 'POST' && request.url().endsWith('/api/generate')) requests.push(request.postDataJSON());
    });
    await page.goto('http://127.0.0.1:8767/');
    await page.waitForLoadState('networkidle');
    assert.equal(await page.locator('#edit-image').isDisabled(), true);
    await page.locator('#file-input').setInputFiles(path.join(root, 'output_img2img_7513.png'));
    await page.locator('#canvas-stack').waitFor({state: 'visible'});
    const urls = () => page.evaluate(() => ['image-canvas', 'mask-canvas'].map(id => document.getElementById(id).toDataURL()));
    const pixel = (x, y, id = 'image-canvas') => page.evaluate(({x,y,id}) => {
      const canvas = document.getElementById(id);
      return Array.from(canvas.getContext('2d').getImageData(Math.floor(canvas.width*x), Math.floor(canvas.height*y), 1, 1).data);
    }, {x,y,id});
    const draw = async (x1, y1, x2 = x1, y2 = y1) => {
      const rect = await page.locator('#mask-canvas').boundingBox();
      await page.mouse.move(rect.x + rect.width*x1, rect.y + rect.height*y1);
      await page.mouse.down();
      assert.equal(await page.locator('#edit-mask').isDisabled(), true);
      await page.mouse.move(rect.x + rect.width*x2, rect.y + rect.height*y2, {steps: 10});
      await page.mouse.up();
    };
    const idle = () => page.waitForFunction(() => !document.getElementById('edit-image').disabled);
    const undo = async () => {await page.locator('#undo').click(); await idle();};
    const redo = async () => {await page.locator('#redo').click(); await idle();};
    const initial = await urls(), originalPixel = await pixel(.5, .55);

    await page.locator('#edit-image').click();
    await page.locator('#brush-size').fill('64');
    await draw(.4, .55, .6, .55);
    const painted = await urls();
    assert.notEqual(painted[0], initial[0]);
    assert.equal(painted[1], initial[1]);
    assert.deepEqual(await pixel(.5, .55), [245,158,66,255]);
    assert.equal(await page.locator('#generate').isDisabled(), true, 'Painting does not implicitly add an AI mask');
    await undo(); assert.deepEqual(await urls(), initial);
    await redo(); assert.deepEqual(await urls(), painted);

    await page.locator('#eraser').click(); await draw(.5, .55);
    assert.deepEqual(await pixel(.5, .55), originalPixel, 'Restore brush reveals the loaded photograph');
    await undo(); assert.deepEqual(await urls(), painted);
    await page.locator('#clear').click(); assert.deepEqual(await urls(), initial);
    await undo(); assert.deepEqual(await urls(), painted);

    await page.locator('[data-color="#5b9ed6"]').click();
    await page.locator('#eyedropper').click();
    const rect = await page.locator('#mask-canvas').boundingBox();
    await page.mouse.click(rect.x + rect.width*.5, rect.y + rect.height*.55);
    assert.equal(await page.locator('#paint-color').inputValue(), '#f59e42');
    assert.equal(await page.locator('#brush').getAttribute('aria-pressed'), 'true');
    assert.deepEqual(await urls(), painted, 'Sampling does not edit either canvas');

    await page.locator('#edit-mask').click();
    await draw(.4, .55, .6, .55);
    const masked = await urls();
    assert.equal(masked[0], painted[0]); assert.notEqual(masked[1], initial[1]);
    await undo(); assert.deepEqual(await urls(), painted);
    await page.keyboard.press('Control+Shift+Z'); await idle(); assert.deepEqual(await urls(), masked);
    await page.locator('#show-mask').uncheck();
    await page.locator('#edit-image').click(); await page.locator('#edit-mask').click();
    assert.equal(await page.locator('#show-mask').isChecked(), false);
    await page.locator('#show-mask').check();
    await page.locator('#edit-image').click();
    assert.match(await page.locator('#mask-canvas').getAttribute('class'), /mask-hidden/);
    await page.locator('[data-color="#5b9ed6"]').click();
    await draw(.65, .75, .75, .65);
    const twicePainted = await urls();
    assert.equal(twicePainted[1], masked[1]);
    await page.keyboard.press('Control+z'); await idle(); assert.deepEqual(await urls(), masked);
    await page.keyboard.press('Control+y'); await idle(); assert.deepEqual(await urls(), twicePainted);
    await page.locator('#clear').click();
    assert.deepEqual(await urls(), [initial[0], masked[1]], 'Clearing paint retains mask');
    await undo(); assert.deepEqual(await urls(), twicePainted);
    await page.locator('#edit-mask').click(); await page.locator('#clear').click();
    assert.deepEqual(await urls(), [twicePainted[0], initial[1]], 'Clearing mask retains paint');
    await undo(); assert.deepEqual(await urls(), twicePainted);
    await page.locator('#edit-image').click();

    await page.selectOption('#model', 'sd15_inpaint');
    assert.deepEqual(await urls(), twicePainted, 'Model switching preserves both edits');
    const downloadPromise = page.waitForEvent('download');
    await page.locator('#save-image').click();
    const download = await downloadPromise;
    assert.equal(download.suggestedFilename(), 'vibepainting-canvas.png');
    await download.saveAs(path.join(out, 'exported-canvas.png'));
    assert.deepEqual(fs.readFileSync(path.join(out, 'exported-canvas.png')), Buffer.from(twicePainted[0].split(',')[1], 'base64'));

    await page.locator('#prompt').fill('a bird with orange feathers');
    await page.locator('#generate').click();
    for (const id of ['edit-image','edit-mask','brush','eraser','paint-color','undo','redo','save-image']) {
      assert.equal(await page.locator(`#${id}`).isDisabled(), true, `${id} locks during generation`);
    }
    await page.waitForFunction(() => document.getElementById('status').textContent.includes('完成'));
    assert.equal(requests.length, 1);
    assert.equal(requests[0].image, twicePainted[0], 'API receives flattened painted pixels');
    assert.deepEqual(await urls(), twicePainted);
    await page.locator('#compare').focus(); await page.keyboard.down('Space');
    assert.equal(await page.locator('#result-image').getAttribute('src'), twicePainted[0], 'Compare uses the painted generation input');
    await page.keyboard.up('Space');
    await page.route('**/api/generate', route => route.fulfill({status: 400, contentType: 'application/json', body: JSON.stringify({error: '测试生成失败'})}));
    await page.locator('#generate').click();
    await page.waitForFunction(() => document.getElementById('status').textContent.includes('测试生成失败'));
    assert.deepEqual(await urls(), twicePainted, 'Failed generation preserves edits');
    await page.unroute('**/api/generate');
    await undo(); assert.deepEqual(await urls(), masked, 'History survives generation');
    await redo(); assert.deepEqual(await urls(), twicePainted);
    await page.locator('#edit-image').click();
    await page.screenshot({path: path.join(out, 'fixture-desktop.png'), fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    await page.screenshot({path: path.join(out, 'fixture-mobile.png'), fullPage: true});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);

    // Reusing a result establishes a new restore baseline, with clean mask/history.
    await page.locator('#use-result').click(); await idle();
    const reused = await urls();
    assert.equal(reused[1], initial[1]);
    assert.equal(await page.locator('#undo').isDisabled(), true);
    assert.equal(await page.locator('#redo').isDisabled(), true);
    await page.locator('#brush').click(); await draw(.5, .5);
    await page.locator('#clear').click(); assert.deepEqual(await urls(), reused);
    await undo();
    await page.locator('#brush').click(); await draw(.2, .3);
    assert.equal(await page.locator('#redo').isDisabled(), true, 'New stroke discards the redo branch');
    await page.locator('#file-input').setInputFiles(path.join(root, 'output_img2img_7513.png')); await idle();
    await page.waitForFunction(() => document.getElementById('undo').disabled);
    assert.deepEqual(await urls(), initial, 'New upload resets all editing surfaces');

    // Pointer events from a touch screen draw through the same scaled canvas path.
    const touch = await browser.newPage({viewport: {width: 390, height: 844}, hasTouch: true, isMobile: true});
    await touch.goto('http://127.0.0.1:8767/'); await touch.waitForLoadState('networkidle');
    await touch.locator('#file-input').setInputFiles(path.join(root, 'output_img2img_7513.png'));
    await touch.locator('#edit-image').click();
    await touch.locator('#mask-canvas').scrollIntoViewIfNeeded();
    const touchBox = await touch.locator('#mask-canvas').boundingBox();
    await touch.touchscreen.tap(touchBox.x + touchBox.width*.5, touchBox.y + touchBox.height*.5);
    assert.equal(await touch.locator('#undo').isEnabled(), true);
    assert.equal(await touch.locator('#generate').isDisabled(), true);
    const touched = await touch.locator('#image-canvas').evaluate(canvas => canvas.toDataURL());
    assert.notEqual(touched, initial[0]);
    await touch.locator('#undo').click();
    await touch.waitForFunction(() => !document.getElementById('edit-image').disabled);
    assert.equal(await touch.locator('#image-canvas').evaluate(canvas => canvas.toDataURL()), initial[0]);
    await touch.close();
    assert.deepEqual(errors, []);
    const report = {ok: true, checks: ['painted pixels','restore brush','eyedropper','mixed undo/redo','independent clear','mask visibility','model switching','PNG export','painted API input','generation locking','failure recovery','new image/result reset','mobile touch'], consoleErrors: errors};
    fs.writeFileSync(path.join(out, 'browser.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } finally {await browser.close();}
})().catch(error => {
  // Avoid dumping entire base64 photographs when a pixel assertion fails.
  console.error(error.message.split('\n')[0], error.stack.split('\n').filter(line => line.trim().startsWith('at ')).join('\n'));
  process.exitCode = 1;
});
