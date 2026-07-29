# Scan PDF Translator

面向扫描教材、双栏论文、公式、电路图、曲线图、Smith Chart 和密集表格的本地 WebUI 翻译工具。它把 OCR、版面区域翻译、中文重排、公式/图表保真与逐页 QA 串成一条可暂停、恢复和定点修复的流水线。

![Scan PDF Translator 数据流视觉](assets/pdf-dataflow-background-dark.png)

> 当前版本以 Windows 10/11 为主要运行环境。云端模式需要用户自己的硅基流动 API Key；完全本地模式需要额外安装 OCR 与翻译模型。

## 功能

- 支持全文、连续页范围和不连续页码。
- 输出双语纵向对照、双语左右并排或纯中文 PDF。
- 云端模式使用硅基流动进行 OCR、OCR 复核和翻译。
- 本地扩展使用 PaddleOCR-VL 与 Ollama，文档内容无需上传。
- MathJax SVG 重绘公式，失败时回退到原始公式裁图。
- 图表采用确定性矢量描摹或高清原图回退，不用生成式图像篡改技术图形。
- 每页保存检查点，支持暂停、继续、失败恢复和指定页码强制修复。
- 自动补翻“有原文但译文为空”的区域，并输出 JSON QA 报告。
- 明暗主题使用同构原创背景，主题偏好保存在浏览器本地。
- API Key 可使用 Windows DPAPI 加密保存，仅当前 Windows 用户可解密。

## 快速开始

### Windows 一键启动

安装以下依赖：

- Python 3.10–3.12（安装时勾选 `Add Python to PATH`）
- Node.js LTS

克隆项目后双击 `START-WEBUI.bat`。首次启动会建立隔离环境、安装依赖，并打开 [http://127.0.0.1:7860](http://127.0.0.1:7860)。

也可以在 PowerShell 中手动运行：

```powershell
python -m pip install -r requirements.txt
npm.cmd install --omit=dev
.\start.ps1
```

## 使用方式

1. 上传 PDF。
2. 选择“全部”，或输入 `33`、`33,58,90`、`33-40` 等页码。
3. 选择云端或本地模型来源、质量模式和输出版本。
4. 云端模式输入硅基流动 API Key；需要时可从[硅基流动官网](https://cloud.siliconflow.cn/)获取。
5. 可先运行“预检页数与成本”，再开始翻译。
6. 在“交付文件”中下载 PDF 与 QA 报告。

指定页修复支持 `27,142,374-376`。工具会复用已有 OCR，仅重译、重绘目标页。

## API Key 与隐私

- API Key 只在任务内存中传递，不写入任务缓存、日志或项目配置。
- Windows 自动保存使用当前用户的 DPAPI，凭据位于 `%LOCALAPPDATA%\ScanPdfTranslator\credentials`。
- 可通过界面的“清除本机密钥”删除凭据。
- 也可只在当前 PowerShell 会话设置 `SILICONFLOW_API_KEY`。
- 不要把本地 WebUI 端口暴露到公网；若密钥曾出现在聊天、终端或截图中，请立即撤销并重新创建。
- 上传至云端模型的页面内容受所选服务商条款与隐私政策约束。

## 完全本地模式

在 `local-bootstrap` 目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-and-enable-local.ps1 -Mode auto
```

安装器会创建独立 Python 环境、安装 PaddleOCR-VL、安装或复用 Ollama，并下载翻译模型和 MathJax。模型缓存完成后可以离线运行。CPU 可运行但处理复杂页面会很慢；GPU 支持取决于本机 CUDA、驱动和模型兼容性。

## 输出与 QA

任务目录按页保存源图、OCR、版面区域、译文、重构页和 QA，因此中断后可复用成功页面。最终 QA 包含：

- 漏翻区域和自动修复计数；
- 公式 SVG 与原图回退计数；
- 图表保留与单位/数字异常；
- PDF 渲染检查；
- `delivery_gate` 与人工复核提示。

`needs_review` 表示该页需要人工核对，不代表输出 PDF 无法打开。

## 项目结构

```text
app.py                 Gradio WebUI 与任务控制
pipeline_v2.py         云端/本地翻译流水线与 QA
translator.py          硅基流动 API、OCR 和页渲染
rich_renderer.py       区域解析与中文 PDF 重构
secret_store.py        Windows DPAPI 凭据存储
local_backend.py       PaddleOCR-VL / Ollama 本地后端
local-bootstrap/       本地扩展安装与卸载脚本
assets/                原创明暗主题背景
```

## 已知限制

- 扫描质量、旋转页面、手写文字和极端复杂公式仍可能需要人工复核。
- 图表文字翻译与图表几何保真之间优先保证几何关系，必要时保留原图。
- 云端模型名称、限流和价格可能变化，请以服务商当前页面为准。
- 本地模式首次安装体积较大，且不同 GPU 环境需要单独验证。

## 开发验证

```powershell
python -m compileall -q .
python -c "from pipeline_v2 import _all_pages; assert _all_pages('1,3-4', 5) == [1, 3, 4]"
```

## 许可证

源代码及仓库内原创视觉素材采用 [MIT License](LICENSE)。第三方依赖仍遵循各自许可证。
