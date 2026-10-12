// Run against tests/serve_fixture.py using the bundled Node Playwright runtime.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const root=path.resolve(__dirname,'..'),out=path.join(root,'test_artifacts','model-switching');

(async()=>{
  fs.mkdirSync(out,{recursive:true});
  const browser=await chromium.launch({headless:true,channel:'msedge'});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1100}});
    const errors=[],sent=[];
    page.on('pageerror',e=>errors.push(e.message));
    page.on('request',r=>{if(r.method()==='POST'&&r.url().endsWith('/api/generate'))sent.push(r.postDataJSON());});
    await page.goto('http://127.0.0.1:8767/');await page.waitForLoadState('networkidle');
    assert.equal(await page.locator('#model option').count(),4);
    await page.locator('#file-input').setInputFiles(path.join(root,'output_img2img_7513.png'));
    await page.locator('#canvas-stack').waitFor({state:'visible'});
    const box=await page.locator('#mask-canvas').boundingBox();
    await page.mouse.move(box.x+box.width*.45,box.y+box.height*.55);
    await page.mouse.down();await page.mouse.move(box.x+box.width*.58,box.y+box.height*.65,{steps:8});await page.mouse.up();
    const pixels=()=>page.evaluate(()=>['image-canvas','mask-canvas'].map(id=>document.getElementById(id).toDataURL()));
    const initial=await pixels();
    await page.locator('#prompt').fill('a bird with orange feathers');
    await page.selectOption('#model','sd15_inpaint');
    await page.locator('#generation-parameters summary').click();
    assert.equal(await page.locator('#steps').inputValue(),'30');
    assert.equal(await page.locator('#guidance').inputValue(),'7.5');
    assert.equal(await page.locator('#control').isVisible(),false);
    await page.locator('#steps').fill('24');await page.locator('#seed').fill('42');
    await page.selectOption('#model','sdxl_inpaint');
    assert.equal(await page.locator('#strength').inputValue(),'0.99');
    assert.equal(await page.locator('#steps').inputValue(),'20');
    await page.selectOption('#model','lama');
    assert.equal(await page.locator('#prompt').isDisabled(),true);
    assert.equal(await page.locator('#generation-parameters').isVisible(),false);
    assert.equal(await page.locator('#prompt').inputValue(),'a bird with orange feathers');
    await page.selectOption('#model','sd15_inpaint');
    assert.equal(await page.locator('#steps').inputValue(),'24');
    assert.equal(await page.locator('#seed').inputValue(),'42');
    assert.deepEqual(await pixels(),initial);
    for(const model of ['sd15_inpaint','lama','sdxl_inpaint','lightning']){
      await page.selectOption('#model',model);
      const count=sent.length;
      await page.locator('#generate').click();
      assert.equal(await page.locator('#model').isDisabled(),true);
      await page.waitForFunction(()=>!document.getElementById('model').disabled);
      assert.equal(sent.length,count+1);
      assert.equal(sent.at(-1).model,model);
      assert.match(await page.locator('#status').innerText(),/完成/);
      assert.ok((await page.locator('#result-seed').innerText()).includes(model==='lama'?'LaMa':model==='lightning'?'Lightning':model==='sdxl_inpaint'?'SDXL':'SD 1.5'));
      if(model==='lama'){
        assert.equal(sent.at(-1).prompt,'');assert.equal('seed' in sent.at(-1),false);
        assert.equal('num_inference_steps' in sent.at(-1),false);
      }
      assert.deepEqual(await pixels(),initial);
    }
    // An API failure keeps the canvas and previous result; the selector unlocks.
    await page.route('**/api/generate',r=>r.fulfill({status:400,contentType:'application/json',body:JSON.stringify({error:'测试模型文件不可用'})}));
    await page.locator('#generate').click();
    await page.waitForFunction(()=>document.getElementById('status').textContent.includes('测试模型文件不可用'));
    assert.equal(await page.locator('#model').isEnabled(),true);assert.deepEqual(await pixels(),initial);
    await page.unroute('**/api/generate');
    await page.selectOption('#model','sd15_inpaint');
    if(!await page.locator('#generation-parameters').evaluate(e=>e.open))await page.locator('#generation-parameters summary').click();
    await page.screenshot({path:path.join(out,'fixture-desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(out,'fixture-mobile.png'),fullPage:true});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    await page.reload();await page.waitForLoadState('networkidle');
    assert.equal(await page.locator('#model').inputValue(),'sd15_inpaint');
    assert.deepEqual(errors,[]);
    const result={ok:true,modelsGenerated:sent.slice(0,4).map(r=>r.model),canvasAndMaskPreserved:true,perModelSettingsPreserved:true,errorRecovery:true,selectionRestoredAfterReload:true,consoleErrors:errors};
    fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
