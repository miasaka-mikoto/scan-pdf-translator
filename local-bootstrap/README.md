# 完全本地模式

这是主程序的扩展包。先解压 `scan-pdf-translator-cloud.zip`，再把本 ZIP 中的四个文件解压到主程序根目录下的 `local-bootstrap` 文件夹（覆盖同名文件即可），然后运行安装脚本。脚本会自动定位其父目录中的 `app.py`。

1. 右键以 PowerShell 运行 `install-and-enable-local.ps1`。
2. 首次安装会创建独立 `.venv-local`、安装 PaddleOCR-VL 并预取官方模型、安装或复用 Ollama、下载 `qwen3:8b` 和 MathJax。安装可重复执行，已缓存组件会复用。
3. 安装完成后启动 `start-local.ps1`，在 WebUI 中选择 `local`。

模型首次下载完成后，OCR、翻译、公式渲染和 PDF 合成可断网运行。CPU 可运行但处理复杂扫描 PDF 较慢；NVIDIA GPU 会自动优先使用 GPU 环境。安装目录会生成 `hardware-report.json` 和 `installed-checksums.json`，记录硬件选择、依赖清单与本地文件 SHA-256，便于审计安装状态。

卸载请运行 `uninstall-local.ps1`。默认不会删除 Ollama 本体或其他模型。
