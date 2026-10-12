# 网页模型切换

运行 `start_web.ps1`，打开 `http://127.0.0.1:8765/`。右侧“补全模型”下拉框提供四种模型：

| 模型 | 工作尺寸 | 默认步数 | 默认引导 | 默认强度 |
|---|---|---|---|---|
| SDXL Lightning + Dreamer ControlNet | 1024 | 2 | 0 | 1.0 |
| SD 1.5 Inpainting | 512 | 30 | 7.5 | 1.0 |
| SDXL Inpainting | 1024 | 20 | 8.0 | 0.99 |
| LaMa | 1024 | 不适用 | 不适用 | 不适用 |

首次打开保留原有 Lightning 默认值，浏览器会记住之后选择的模型。当前页面内各模型分别记住用户调整过的参数；首次选中某个模型时使用上表的默认参数。切换不会修改画布、遮罩、提示词或已有结果。

只有 Lightning 显示 ControlNet 结构约束。LaMa 不使用文字、负向提示词或种子，提示词输入框禁用但保留内容，切回扩散模型后仍可编辑。SDXL 以原图为初始内容，不将遮罩内预填白色。

选择模型不立即加载权重，点击生成时才加载。生成期间禁止切换。服务只缓存一个模型，切换时释放上一个模型及其显存；同模型的后续生成复用缓存。加载和生成状态显示在页面底部，结果下方记录实际使用的模型及种子。显示耗时包含必要的模型加载。

## 本地权重

Lightning 沿用 `models/sdxl_lightning_2step.safetensors` 和 `models/controlnet-inpaint-sdxl`；其他三个模型位于 `models/comparison/sd15_inpaint`、`models/comparison/sdxl_inpaint`、`models/comparison/lama`，即实验时已下载的权重。缺少文件的选项会标记为未就绪并禁用，后端也会拒绝请求，不会接受任意模型路径。

## 接口

- `GET /api/models`：本地模型列表、可用性、默认参数和支持的字段。
- `POST /api/generate`：原有图片、遮罩请求增加 `model` 字段，允许 `lightning`、`sd15_inpaint`、`sdxl_inpaint`、`lama`。省略时继续使用 Lightning。
- `GET /api/jobs/{id}`：结果增加 `model` 和 `model_label`。LaMa 的 `seed` 为 `null`。

模型不使用的采样字段会被忽略；前端不发送这些字段。模型自带检测器若将输出替换为黑图，服务返回生成失败，避免把黑块作为成功结果。画布和上一次结果保留，可重新生成。

## 验证

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -p 'test_*.py' -v
# 主服务已启动、处于空闲状态时：
& '.\.venv\Scripts\python.exe' tests\smoke_model_switching.py
```

浏览器交互测试使用 `tests/serve_fixture.py` 的 8767 替代服务及 `tests/model_switching_browser.cjs`，需已安装 Playwright 的 Node 运行环境。替代服务不运行 AI。真实模型生成验证另走 8765 服务；结果和截图存放于 `test_artifacts/model-switching`。
