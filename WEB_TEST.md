# 局部补全测试页面

在项目目录打开终端，使用装有项目依赖的 Python：

```powershell
python web_server.py
```

当前电脑也可以运行以下脚本，它优先使用项目 `.venv`，其次使用 PyCharm 原来配置的相邻 `AI_ALL_Project/.venv`：

```powershell
powershell -ExecutionPolicy Bypass -File .\start_web.ps1
```

打开 http://127.0.0.1:8765 。保持终端运行，按 Ctrl+C 停止服务。更换端口使用 `python web_server.py --port 8766`，或给 PowerShell 脚本传入 `-Port 8766`。可以通过 `-PythonPath '完整的 python.exe 路径'` 指定环境。

如果生成时出现 `ERR_CONNECTION_RESET`，表示 HTTP 连接被中断，先检查终端中的服务是否还在运行。服务启动、任务进度和 Python 异常会写入 `logs/server.log`；原生库异常栈写入 `logs/native-crash.log`。不能只凭浏览器报错判断具体原因。网页断连时保留画布，重启服务后可重试，无需刷新丢失遮罩。`/favicon.ico` 返回空图标响应，不再产生缺失图标的 404。

服务在 Windows 上独占端口，重复启动会明确报端口占用。先停止旧进程再启动新版，或指定其他端口。`/api/health` 返回当前服务 PID 和是否正在生成，便于确认连接到了哪个进程。

图片在浏览器中会转为无损 PNG，并与遮罩一起编码为 JSON 上传，传输大小可能远大于原始 JPEG 文件。上传上限现在为 192 MiB，以容纳页面允许的最多 1600 万像素原图与遮罩；不通过有损压缩或缩小原图规避限制。页面提交前从服务读取当前上限并检查实际请求大小，超限时直接提示，保留画布。旧版服务的 32 MiB 上限曾造成 HTTP 413 被浏览器显示为连接重置；新版还会区分“服务离线”和“服务仍在运行但上传失败”。

## 使用

1. 选择或拖入 PNG、JPEG、WebP 图片（文件最大 20 MB，最多 1600 万像素）。透明图片会铺白底。
2. 用画笔涂抹需要补全的区域；粉色仅用于预览。使用橡皮擦除遮罩，不会擦掉原图。
3. 调整笔刷大小。大小按原图像素计。按 B 切换画笔，E 切换橡皮，Ctrl/Cmd+Z 撤销，最多保留 20 次图像或遮罩操作（受内存预算限制）。直接画图与重做说明见 [PAINTING.md](PAINTING.md)。
4. 按需输入描述，也可留空，再点击「生成补全」。留空或只输入空白字符时，将空字符串传给模型，不会自动生成提示词。首次加载模型可能较久，页面显示加载或生成状态。
5. 在右侧查看结果，可按住按钮对比原图、下载 PNG，或用结果继续绘制下一轮遮罩。也支持导出黑白遮罩（白色重绘、黑色保留）。

生成期间禁止修改输入，避免结果与当前画布错配。刷新页面会尝试恢复当前标签页的任务，但不会保存原图和未提交的遮罩。服务仅在内存中保留最近一次任务，重启服务后任务丢失；图片需要点击下载保存。多个标签页共用同一个生成实例，已有任务运行时会拒绝新任务。

## 模型与处理

- 后端通过 `inpaint_engine.py` 路由四种模型。Lightning 沿用 `sdxl_lightning_inpaint.py` 的加载逻辑，原来的 `inpaint()` 文件接口仍可使用。完整模型说明见 [MODEL_SWITCHING.md](MODEL_SWITCHING.md)。
- 启动时在主线程初始化 PyTorch、Diffusers 等底层库，避免当前 Windows 环境在后台线程首次导入原生库时退出。首次生成才加载选中的模型，同模型后续生成复用实例，切换时释放旧模型；推理串行执行。
- 模型路径按代码所在目录定位，不依赖终端当前目录。
- 图片等比缩放后填充至模型工作尺寸（SD 1.5 为 512×512，其余为 1024×1024），完成后裁去填充并还原尺寸，使用原始遮罩合成。黑色遮罩区域的输入像素保持不变。极细的遮罩可能在缩小后丢失。
- 默认参数：强度 1、步数 2、引导 0、ControlNet 强度 0.3。依据 [Lightning 官方说明](https://huggingface.co/ByteDance/SDXL-Lightning)，步数与权重匹配，引导默认 0；负面提示词仅在引导大于 1 时生效。强度乘步数必须至少为 1。更换步数或提高引导属于实验设置，可能降低质量。
- 依据 [Dreamer ControlNet 模型说明](https://huggingface.co/destitech/controlnet-inpaint-dreamer-sdxl)，传入 ControlNet 的条件图将遮罩区域涂成纯白。它与原始图片、黑白重绘遮罩分别传入模型。
- 只使用 PyTorch，在加载 Transformers 之前设置 `USE_TF=0`，避免当前环境误加载 TensorFlow。
- 当前共享 Windows 环境还出现了 `pyarrow` 原生导入时访问违规/栈溢出。图像补全入口会关闭已安装 Transformers 4.x 的可选 scikit-learn 可用性标志，避开 `sklearn → pandas → pyarrow` 的文本生成依赖链；不修改或卸载共享环境里的包。这是针对当前环境的兼容处理，升级 Transformers 时需重新验证。
- 不再在补全模块中覆盖代理。需要代理时，在启动前自行设置 `http_proxy` 和 `https_proxy`。首次缺少模型配置或其他组件时，Diffusers 仍可能访问 Hugging Face；页面会显示失败原因，详细堆栈在终端。
- 页面和 HTTP 服务不需要额外前端构建工具或 Web 框架，依赖沿用 `requirements.txt`。仅监听本机 `127.0.0.1`，不用于公共部署。

## 验证

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
```

这些测试验证参数、任务接口、并发拒绝和图像尺寸/遮罩外像素保护，使用替代推理函数，不消耗 GPU。`tests/browser_check.cjs` 使用 Playwright 验证实际画布交互；`tests/serve_fixture.py` 是其专用测试服务，输出明确是测试用的着色结果，不能替代真实模型质量验证。

`tests/real_generation_check.cjs` 是可选的真实 GPU 集成测试，对运行中的 8765 服务上传项目样例、绘制遮罩并连续生成两次。需要 Node.js 能找到 Playwright 包，且安装了 Edge。测试时应停止在其他标签页提交任务。

2026-09-06 已通过 7 项后端测试、桌面和移动宽度的浏览器交互测试，以及真实模型出图验证。最终默认参数下，模型已加载时两次生成分别约 1.4 秒和 1.3 秒（当前机器、1024×1024 样例，非性能保证）。逐像素核对确认输出尺寸一致，遮罩外像素完全保留，遮罩内确实发生变化。实际截图与输出位于 `test_artifacts/workbench-real.png`、`test_artifacts/real-result.png`。
