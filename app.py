from __future__ import annotations

import os
import json
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import gradio as gr
from pypdf import PdfReader

from job_store import JobStore
from pipeline_v2 import JobSpec, _all_pages, estimate_translation_cost_yuan, run_cloud_job, run_local_job
from secret_store import clear_api_key, load_api_key, save_api_key


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs-v2"
OUTPUTS = Path(os.environ.get("PDF_TRANSLATOR_OUTPUT_DIR", ROOT / "outputs"))
BACKGROUND_LIGHT = ROOT / "assets" / "pdf-dataflow-background-light.png"
BACKGROUND_DARK = ROOT / "assets" / "pdf-dataflow-background-dark.png"
BACKGROUND_LIGHT_URL = "/gradio_api/file=" + quote(BACKGROUND_LIGHT.as_posix())
BACKGROUND_DARK_URL = "/gradio_api/file=" + quote(BACKGROUND_DARK.as_posix())
INITIAL_API_KEY = load_api_key()

# Gradio probes its own localhost endpoint at startup.  Keep that probe off a
# corporate/global HTTP proxy, otherwise a healthy local service can receive a
# misleading 502 during startup.
_no_proxy_entries = {entry.strip() for entry in os.environ.get("NO_PROXY", "").split(",") if entry.strip()}
_no_proxy_entries.update({"127.0.0.1", "localhost", "::1"})
os.environ["NO_PROXY"] = ",".join(sorted(_no_proxy_entries))
os.environ["no_proxy"] = os.environ["NO_PROXY"]


class Jobs:
    def __init__(self) -> None:
        self.stores: dict[str, JobStore] = {}
        self.specs: dict[str, JobSpec] = {}
        self.workers: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def _launch(self, job_id: str, store: JobStore, spec: JobSpec) -> None:
        def worker() -> None:
            try:
                store.emit("job_received", f"任务已接收，后端：{spec.source_mode}", 0.01, backend=spec.source_mode)
                (run_local_job if spec.source_mode == "local" else run_cloud_job)(spec, store, OUTPUTS)
            except Exception as exc:
                state = "cancelled" if "取消" in str(exc) else "failed"
                store.save_status(state=state, error=str(exc), completed_at=time.time())
                store.emit(state, str(exc), _last_task_progress(store))

        thread = threading.Thread(target=worker, daemon=True, name=f"pdf-job-{job_id}")
        self.workers[job_id] = thread
        thread.start()

    def start(self, pdf: str, api_key: str, pages: str, source_mode: str, quality: str, outputs: list[str], formula_strategy: str, figure_strategy: str, glossary: str, resume_job_id: str = "", auto_repair_missing: bool = True, repair_pages: str = "") -> str:
        effective_mode = source_mode
        if source_mode == "auto":
            config = ROOT / "local-mode.json"
            if config.exists():
                from local_backend import local_health
                effective_mode = "local" if local_health()["ollama"] else "siliconflow"
            else:
                effective_mode = "siliconflow"
        resolved_key = api_key.strip() or os.environ.get("SILICONFLOW_API_KEY", "").strip()
        if effective_mode != "local" and not resolved_key:
            raise ValueError("云端/自动模式需要硅基流动 API Key。")
        resume_job_id = resume_job_id.strip()
        if resume_job_id:
            store = JobStore(RUNS, resume_job_id)
            if not store.status_path.exists():
                raise ValueError("未找到要恢复的任务 ID。")
            old_status = json.loads(store.status_path.read_text(encoding="utf-8"))
            saved_pdf = Path(old_status.get("input_pdf", ""))
            input_pdf = Path(pdf) if pdf else saved_pdf
            if not input_pdf.exists():
                raise ValueError("恢复任务需要重新上传原 PDF；原临时文件已不存在。")
            preflight = old_status.get("preflight", {})
            selected = preflight.get("selected_pages")
            if selected:
                pages = ",".join(str(page) for page in selected)
            job_id = resume_job_id
            store.save_status(state="queued", input_pdf=str(input_pdf), resumed_with_new_key=True)
        else:
            if not pdf:
                raise ValueError("请上传 PDF。")
            input_pdf = Path(pdf)
            job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            store = JobStore(RUNS, job_id)
            store.save_status(state="queued", input_pdf=str(input_pdf))
        spec = JobSpec(input_pdf, pages, resolved_key, effective_mode, quality, outputs, formula_strategy, figure_strategy, glossary, auto_repair_missing, repair_pages)
        with self._lock:
            self.stores[job_id] = store
            self.specs[job_id] = spec
            self._launch(job_id, store, spec)
        return job_id

    def resume(self, job_id: str) -> str:
        store = self.store(job_id)
        if not store:
            return "任务不存在。"
        store.control.pause_requested.clear()
        with self._lock:
            worker = self.workers.get(job_id)
            spec = self.specs.get(job_id)
            if (not worker or not worker.is_alive()) and spec:
                store.save_status(state="running", error=None, resumed_at=time.time())
                store.emit("resumed", "从已保存检查点继续；成功页面不会重复调用 API", _last_task_progress(store))
                self._launch(job_id, store, spec)
                return "正在从检查点继续任务。"
        store.save_status(state="running")
        store.emit("resumed", "任务继续", _last_task_progress(store))
        return "任务已继续。"

    def store(self, job_id: str) -> JobStore | None:
        with self._lock:
            existing = self.stores.get(job_id)
            if existing:
                return existing
            # A restarted WebUI can still render persisted events, page states
            # and completed downloads.  The task cache still never contains an
            # API key.  If the user enabled encrypted local saving, the WebUI
            # can prefill it from the current Windows user's DPAPI store.
            if job_id and (RUNS / job_id).is_dir():
                recovered = JobStore(RUNS, job_id)
                self.stores[job_id] = recovered
                return recovered
            return None


JOBS = Jobs()
INSTALL_PROCESS: subprocess.Popen | None = None


def _last_task_progress(store: JobStore) -> float:
    """Return progress before control/terminal events; never reset UI to zero."""
    events = store.read_events()
    for event in reversed(events):
        if event.get("stage") not in {"failed", "cancelled", "paused", "resumed"}:
            return float(event.get("progress", 0.0))
    return 0.0


def start_local_install() -> tuple[str, int]:
    """Run the user-approved installer without changing global execution policy."""
    global INSTALL_PROCESS
    if INSTALL_PROCESS and INSTALL_PROCESS.poll() is None:
        return "本地扩展正在安装中。", 1
    script = ROOT / "local-bootstrap" / "install-and-enable-local.ps1"
    if not script.exists():
        raise FileNotFoundError("未找到本地安装脚本。")
    INSTALL_PROCESS = subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-Mode", "auto", "-TranslationModel", "qwen3:8b"],
        cwd=str(ROOT),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return "已启动本地安装：正在检测硬件并创建隔离环境。", 1


def poll_local_install() -> tuple[str, int]:
    log = ROOT / "local-bootstrap" / "install.log"
    latest = "等待安装器输出。"
    if log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines:
            latest = lines[-1]
    stages = [("Detecting hardware", 5), ("Creating isolated", 12), ("Installing PaddleOCR", 30), ("Prefetching PaddleOCR", 55), ("Installing Ollama", 65), ("Downloading local translation", 80), ("Installing MathJax", 92), ("installation completed", 100)]
    progress = next((value for marker, value in stages if marker in latest), 1)
    if INSTALL_PROCESS:
        code = INSTALL_PROCESS.poll()
        if code == 0:
            return "安装完成。请点击“检查本地模式”，然后选择 local。", 100
        if code is not None:
            return f"安装器退出，代码 {code}。最后日志：{latest}", progress
    return f"安装中：{latest}", progress


def prepare_local_bundle() -> str:
    """Create a user-downloadable local extension ZIP from shipped scripts."""
    source = ROOT / "local-bootstrap"
    if not source.exists():
        raise FileNotFoundError("本地扩展脚本缺失，请重新下载完整云端程序包。")
    bundle_root = ROOT / "downloads"
    bundle_root.mkdir(exist_ok=True)
    archive = bundle_root / "scan-pdf-translator-local-bootstrap"
    shutil.make_archive(str(archive), "zip", root_dir=str(source))
    return str(archive.with_suffix(".zip"))


def check_local_mode() -> str:
    config = ROOT / "local-mode.json"
    if not config.exists():
        return "本地模式未安装。点击“下载本地扩展包”，解压到主程序的 local-bootstrap 文件夹后运行安装脚本。"
    try:
        settings = json.loads(config.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return "本地模式配置文件无效，请重新运行安装脚本。"
    from local_backend import local_health
    health = local_health()
    model_info = "、".join(health["models"]) if health["models"] else "未检测到模型"
    return f"本地模式配置：{settings.get('device', 'unknown')}；Ollama：{'运行中' if health['ollama'] else '未运行'}；模型：{model_info}。"


def preflight(pdf: str, pages: str, source_mode: str, quality: str) -> str:
    if not pdf:
        return "请先上传 PDF。"
    reader = PdfReader(pdf)
    total = len(reader.pages)
    sample_chars = sum(len((reader.pages[index].extract_text() or "").strip()) for index in range(min(5, total)))
    kind = "扫描型" if sample_chars < min(5, total) * 100 else "文字型或混合型"
    try:
        selected = len(_all_pages(pages.replace("，", ","), total))
    except ValueError as exc:
        return f"页码格式有误：{exc}"
    effective = "siliconflow_free" if source_mode == "siliconflow_free" else "siliconflow"
    estimate = estimate_translation_cost_yuan(selected, quality, effective)
    cost_text = "当前模型官方标价为免费（仍受限流与价格调整影响）" if effective == "siliconflow_free" else f"翻译 token 预估 ¥{estimate:.4f}，视觉 OCR 以实际账单为准"
    return f"预检：{total} 页，{kind} PDF；选择 {selected} 页。预计约 {max(1, selected) * (45 if quality == 'high' else 20)} 秒；{cost_text}。"


def load_saved_api_key_for_ui() -> tuple[str, str]:
    key = load_api_key()
    status = "已从当前用户的加密凭据中自动填充。" if key else "输入完整 Key 后离开输入框即可自动保存。"
    return key, status


OUTPUT_LABEL_MAP = {
    "双语纵向对照（默认）": "bilingual_vertical",
    "双语左右并排": "bilingual_side_by_side",
    "纯中文": "chinese_only",
}
OUTPUT_VALUES = set(OUTPUT_LABEL_MAP.values())


def normalize_output_selection(output_selection) -> list[str]:
    """Normalize UI/API output selection to stable internal values."""
    selected = output_selection if isinstance(output_selection, (list, tuple)) else [output_selection]
    normalized: list[str] = []
    for item in selected:
        if item in OUTPUT_VALUES:
            value = item
        else:
            value = OUTPUT_LABEL_MAP.get(item)
        if value and value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("至少选择一个输出版本。")
    return normalized


def start_job(pdf, api_key, pages, source_mode, quality, output_labels, formula_label, figure_label, glossary, resume_job_id, auto_repair_missing, repair_pages):
    outputs = normalize_output_selection(output_labels)
    formula = "mathjax" if formula_label.startswith("MathJax") else "source_crop"
    figure = "vector_trace" if figure_label.startswith("SVG") else ("preserve" if figure_label.startswith("确定性") else "source_crop")
    job_id = JOBS.start(pdf, api_key, pages, source_mode, quality, outputs, formula, figure, glossary, resume_job_id, auto_repair_missing, repair_pages)
    return job_id, f"任务 {job_id} 已开始。正在进行真实流式处理。", []


def pages_from_sliders(first_page: int, last_page: int) -> tuple[str, str]:
    first, last = sorted((int(first_page), int(last_page)))
    page_spec = str(first) if first == last else f"{first}-{last}"
    count = last - first + 1
    return page_spec, f"滑条范围：第 {first}–{last} 页，共 {count} 页"


def poll(job_id: str):
    if not job_id:
        return "尚未启动任务。", 0, []
    store = JOBS.store(job_id)
    if not store:
        return f"未找到任务 {job_id}。", 0, []
    status = {}
    if store.status_path.exists():
        import json
        status = json.loads(store.status_path.read_text(encoding="utf-8"))
    events = store.read_events(max(0, store._sequence - 8))
    # A terminal ``failed`` event must never masquerade as completed work.
    # Use the most recent genuine processing checkpoint (or the persisted
    # status checkpoint) so the bar remains truthful and resume is intuitive.
    progress = int(_last_task_progress(store) * 100)
    completed = failed = regions = equations = figures = tokens = 0
    for page_dir in store.pages_dir.glob("page-*"):
        state_path = page_dir / "state.json"
        qa_path = page_dir / "qa.json"
        if state_path.exists():
            try:
                page_state = json.loads(state_path.read_text(encoding="utf-8"))
                completed += int(page_state.get("stage") == "completed")
            except (OSError, json.JSONDecodeError):
                pass
        if qa_path.exists():
            try:
                qa = json.loads(qa_path.read_text(encoding="utf-8"))
                failed += int(qa.get("status") != "passed")
                regions += int(qa.get("regions", 0)); equations += int(qa.get("equations", 0)); figures += int(qa.get("figures", 0))
            except (OSError, json.JSONDecodeError):
                pass
        for usage_path in page_dir.glob("*-usage.json"):
            try:
                tokens += int(json.loads(usage_path.read_text(encoding="utf-8")).get("total_tokens", 0))
            except (OSError, json.JSONDecodeError, ValueError):
                pass
    elapsed = max(0, time.time() - float(status.get("started_at", time.time())))
    eta = (elapsed * (100 - progress) / progress) if progress else None
    eta_text = f"；ETA 约 {int(eta)} 秒" if eta is not None else ""
    lines = [f"状态：{status.get('state', 'running')}；总体进度：{progress}%；已完成 {completed} 页，需复核/失败 {failed} 页；OCR 区域 {regions}，公式 {equations}，图表 {figures}；tokens {tokens}；已运行 {int(elapsed)} 秒{eta_text}"]
    for event in events[-8:]:
        page = f" 第 {event['page']} 页" if event.get("page") else ""
        lines.append(f"- `{event['stage']}`{page}：{event['message']}")
    files = list(status.get("outputs", []))
    if status.get("report"):
        files.append(status["report"])
        html = Path(status["report"]).with_name("QA-report.html")
        if html.exists():
            files.append(str(html))
    return "\n".join(lines), progress, files


def pause(job_id: str) -> str:
    store = JOBS.store(job_id)
    if not store:
        return "任务不存在。"
    store.control.pause_requested.set()
    store.save_status(state="paused")
    store.emit("paused", "将在当前安全检查点暂停", _last_task_progress(store))
    return "已请求暂停；当前 API 请求结束后生效。"


def resume(job_id: str) -> str:
    return JOBS.resume(job_id)


def cancel(job_id: str) -> str:
    store = JOBS.store(job_id)
    if not store:
        return "任务不存在。"
    store.control.cancel_requested.set()
    return "已请求取消；已完成页面和检查点会被保留。"


APP_CSS = """
:root { --ink:#18211f; --muted:#60706c; --line:#d9e2df; --paper:#f7faf9; --surface:#ffffff; --teal:#0f766e; --teal-dark:#075f59; --blue:#2563eb; --coral:#d95f4f; }
.gradio-container { position:relative; z-index:1; max-width:1440px !important; margin:0 auto !important; padding:28px 32px 48px !important; background:rgba(247,250,249,.78); color:var(--ink); font-family:Inter,"Microsoft YaHei UI","Segoe UI",sans-serif !important; backdrop-filter:blur(22px) saturate(115%); -webkit-backdrop-filter:blur(22px) saturate(115%); }
#app-header { display:flex; justify-content:space-between; align-items:flex-end; gap:24px; padding:8px 0 26px; border-bottom:1px solid var(--line); margin-bottom:22px; }
#app-header h1 { margin:0; font-size:30px; font-weight:720; letter-spacing:0; color:var(--ink); }
#app-header p { margin:7px 0 0; color:var(--muted); font-size:14px; line-height:1.55; max-width:760px; }
.header-actions { display:flex; align-items:center; gap:10px; flex-wrap:wrap; justify-content:flex-end; }
.brand-mark { font:700 13px/1 ui-monospace,SFMono-Regular,Consolas,monospace; color:var(--teal); border:1px solid #94cfc8; padding:8px 10px; white-space:nowrap; }
#theme-toggle { min-height:34px; border:1px solid rgba(113,193,184,.74); border-radius:999px; padding:7px 12px; color:var(--ink); background:linear-gradient(135deg,rgba(255,255,255,.7),rgba(220,244,240,.48)); box-shadow:0 7px 18px rgba(23,67,61,.12),inset 0 1px 0 rgba(255,255,255,.84); backdrop-filter:blur(12px) saturate(140%); -webkit-backdrop-filter:blur(12px) saturate(140%); font:650 13px/1.1 Inter,"Microsoft YaHei UI","Segoe UI",sans-serif; cursor:pointer; transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease; }
#theme-toggle:hover { transform:translateY(-2px); border-color:rgba(64,190,176,.95); box-shadow:0 12px 25px rgba(23,67,61,.18),inset 0 1px 0 rgba(255,255,255,.9); }
#theme-toggle:active { transform:translateY(1px) scale(.97); box-shadow:0 3px 8px rgba(23,67,61,.14),inset 0 2px 8px rgba(7,95,89,.18); }
#api-key-guide { margin:0 0 8px; padding:10px 12px; border:1px solid rgba(121,190,181,.55); border-radius:6px; background:rgba(231,247,244,.56); color:var(--muted); font-size:13px; line-height:1.55; backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); }
#api-key-guide a { color:var(--teal); font-weight:750; text-decoration:none; border-bottom:1px solid rgba(15,118,110,.35); }
#api-key-guide a:hover { color:var(--teal-dark); border-bottom-color:currentColor; }
.section-label { margin:0 0 10px; color:var(--teal); font-size:12px; font-weight:750; letter-spacing:0; text-transform:uppercase; }
.workspace { gap:20px !important; align-items:stretch !important; }
.workspace > .column { border:1px solid var(--line); background:var(--surface); padding:18px; border-radius:6px; }
#run-panel { margin-top:20px; padding:18px; background:#eef7f5; border:1px solid #b8dcd7; border-radius:6px; }
#run-panel .prose { margin-top:4px; }
#delivery-tabs { margin-top:20px; border-top:1px solid var(--line); padding-top:8px; }
.gr-button { min-height:38px !important; border-radius:5px !important; font-weight:650 !important; box-shadow:none !important; }
.gr-button-primary { background:var(--teal) !important; border-color:var(--teal) !important; }
.gr-button-primary:hover { background:var(--teal-dark) !important; }
.gr-button-secondary { background:#fff !important; border-color:#aab9b5 !important; color:var(--ink) !important; }
.gr-button-stop { color:#9f2d20 !important; border-color:#e4a99f !important; background:#fff8f7 !important; }
.gr-input, .gr-box, .gr-file { border-radius:5px !important; border-color:#cbd7d3 !important; box-shadow:none !important; }
.gr-file { min-height:188px !important; background:#fbfdfc !important; }
label span, .gr-form label { font-weight:650 !important; color:var(--ink) !important; }
.gr-accordion { border:1px solid var(--line) !important; border-radius:5px !important; margin-top:12px !important; }
.gr-accordion > button { color:var(--ink) !important; }
.gradio-container .tab-nav button { border-radius:0 !important; color:var(--muted) !important; }
.gradio-container .tab-nav button.selected { color:var(--teal) !important; border-bottom-color:var(--teal) !important; }
.status-copy { color:var(--muted); font-size:13px; }
#local-guide { color:var(--muted); font-size:13px; line-height:1.6; }
/* Gradio otherwise follows the operating-system dark preference inside its
   components even when this workbench intentionally uses a light surface. */
body, body.dark { background:transparent !important; }
body.dark .gradio-container { background:rgba(247,250,249,.78) !important; color:var(--ink) !important; }
body.dark .gradio-container .block, body.dark .gradio-container .gr-box, body.dark .gradio-container .form, body.dark .gradio-container .wrap, body.dark .gradio-container .input-container, body.dark .gradio-container .gr-file, body.dark .gradio-container .gr-panel { background:rgba(255,255,255,.69) !important; color:var(--ink) !important; }
body.dark .gradio-container input, body.dark .gradio-container textarea, body.dark .gradio-container select { background:rgba(255,255,255,.82) !important; color:var(--ink) !important; border-color:#cbd7d3 !important; }
body.dark .gradio-container label, body.dark .gradio-container .prose, body.dark .gradio-container .prose *, body.dark .gradio-container .gr-markdown, body.dark .gradio-container .gr-markdown * { color:var(--ink) !important; }
body.dark .gradio-container .gr-file { background:rgba(251,253,252,.72) !important; }
body.dark .gradio-container .gr-file * { color:var(--ink) !important; }
body.dark .gradio-container .gr-button-primary { color:#fff !important; }
body.dark .gradio-container .wrap { background:transparent !important; }
body.dark .gradio-container label { background:transparent !important; color:var(--ink) !important; }
body.dark .gradio-container label[data-testid$="-radio-label"], body.dark .gradio-container label[data-testid$="-checkbox-label"] { background:rgba(255,255,255,.76) !important; border:1px solid #cbd7d3 !important; border-radius:4px !important; padding:7px 10px !important; }
body.dark .gradio-container label[data-testid$="-radio-label"].selected, body.dark .gradio-container label[data-testid$="-checkbox-label"].selected { background:rgba(231,245,242,.72) !important; border-color:#78bfb6 !important; }
body.dark .gradio-container input[type="radio"], body.dark .gradio-container input[type="checkbox"] { accent-color:var(--teal) !important; }
@media (max-width: 760px) { .gradio-container { padding:16px !important; } #app-header { align-items:flex-start; flex-direction:column; gap:14px; } #app-header h1 { font-size:25px; } .workspace > .column { padding:14px; } }
"""
if BACKGROUND_LIGHT.exists() and BACKGROUND_DARK.exists():
    APP_CSS += f"""
    /* The illustration belongs to the page, rather than the workbench itself:
       this lets the semi-transparent container genuinely blur it.  A direct
       background image on .gradio-container cannot be affected by its own
       backdrop-filter. */
    html, body, body.dark {{
      min-height:100%;
      background-color:#dfe7e5 !important;
      background-image:linear-gradient(118deg, rgba(236,249,246,.25), rgba(222,235,232,.38)), url('{BACKGROUND_LIGHT_URL}') !important;
      /* Keep both ends of the document-to-data-sister illustration inside
         the viewport instead of scaling a tall Gradio page from its blank
         centre section. */
      background-size:cover, 100vw auto !important;
      background-position:center, center top 24px !important;
      background-repeat:no-repeat !important;
      background-attachment:fixed !important;
    }}
    #root, gradio-app {{ min-height:100vh; background:transparent !important; }}
    .gradio-container, body.dark .gradio-container {{
      /* High-transmission glass: the signal illustration is still visible,
         while the higher blur preserves a calm surface for dense controls. */
      background-color:rgba(239,249,247,.29) !important;
      background-image:linear-gradient(135deg, rgba(255,255,255,.46), rgba(239,249,247,.29)), url('{BACKGROUND_LIGHT_URL}') !important;
      background-size:cover, 100vw auto !important;
      background-position:center, center top 24px !important;
      background-repeat:no-repeat !important;
      background-attachment:scroll, fixed !important;
      border:1px solid rgba(255,255,255,.78);
      box-shadow:0 28px 90px rgba(23,67,61,.18), inset 0 1px 0 rgba(255,255,255,.72);
      backdrop-filter:blur(24px) saturate(138%) brightness(1.025);
      -webkit-backdrop-filter:blur(24px) saturate(138%) brightness(1.025);
    }}
    .workspace > .column, body.dark .gradio-container .workspace > .column {{
      background:linear-gradient(140deg, rgba(255,255,255,.59), rgba(239,252,248,.37)) !important;
      border-color:rgba(255,255,255,.82) !important;
      box-shadow:0 12px 36px rgba(23,67,61,.08), inset 0 1px 0 rgba(255,255,255,.9);
      backdrop-filter:blur(18px) saturate(128%);
      -webkit-backdrop-filter:blur(18px) saturate(128%);
    }}
    #run-panel {{
      background:rgba(226,246,241,.58) !important;
      border-color:rgba(156,211,201,.72) !important;
      backdrop-filter:blur(14px) saturate(125%);
      -webkit-backdrop-filter:blur(14px) saturate(125%);
    }}
    body.dark .gradio-container .gr-file {{
      background:rgba(255,255,255,.76) !important;
    }}
    /* Tinted glass controls: the highlight tracks hover, then compresses
       into the surface when pressed so a click feels tangible. */
    body .gradio-container button.primary, body .gradio-container button.secondary, body .gradio-container button.stop {{
      position:relative !important;
      overflow:hidden !important;
      isolation:isolate;
      border:1px solid rgba(255,255,255,.76) !important;
      background:linear-gradient(135deg, rgba(255,255,255,.68), rgba(231,245,242,.46)) !important;
      color:var(--ink) !important;
      box-shadow:0 8px 18px rgba(23,67,61,.12), inset 0 1px 0 rgba(255,255,255,.84) !important;
      backdrop-filter:blur(14px) saturate(145%);
      -webkit-backdrop-filter:blur(14px) saturate(145%);
      transition:transform .16s cubic-bezier(.2,.8,.2,1), box-shadow .16s ease, filter .16s ease, border-color .16s ease !important;
    }}
    body .gradio-container button.primary::before, body .gradio-container button.secondary::before, body .gradio-container button.stop::before {{
      content:"";
      position:absolute;
      inset:-2px;
      z-index:-1;
      opacity:.74;
      background:linear-gradient(112deg, rgba(255,255,255,.68) 0%, rgba(255,255,255,.16) 38%, rgba(81,206,193,.18) 70%, rgba(255,255,255,.42) 100%);
      transform:translateX(-58%);
      transition:transform .42s cubic-bezier(.2,.8,.2,1), opacity .18s ease;
    }}
    body .gradio-container button.primary:hover, body .gradio-container button.secondary:hover, body .gradio-container button.stop:hover {{
      transform:translateY(-2px);
      border-color:rgba(122,207,196,.96) !important;
      box-shadow:0 14px 28px rgba(23,67,61,.18), inset 0 1px 0 rgba(255,255,255,.92) !important;
      filter:saturate(1.08) brightness(1.02);
    }}
    body .gradio-container button.primary:hover::before, body .gradio-container button.secondary:hover::before, body .gradio-container button.stop:hover::before {{ transform:translateX(38%); }}
    body .gradio-container button.primary:active, body .gradio-container button.secondary:active, body .gradio-container button.stop:active {{
      transform:translateY(1px) scale(.985);
      box-shadow:0 3px 9px rgba(23,67,61,.13), inset 0 2px 9px rgba(7,95,89,.18), inset 0 1px 0 rgba(255,255,255,.48) !important;
      filter:saturate(1.14) brightness(.98);
      transition-duration:.07s !important;
    }}
    body .gradio-container button.primary:active::before, body .gradio-container button.secondary:active::before, body .gradio-container button.stop:active::before {{ opacity:.96; transform:translateX(0) scale(1.08); transition-duration:.07s; }}
    body .gradio-container button.primary:focus-visible, body .gradio-container button.secondary:focus-visible, body .gradio-container button.stop:focus-visible {{ outline:3px solid rgba(45,181,169,.45) !important; outline-offset:2px; }}
    body .gradio-container button.primary {{
      color:#fff !important;
      border-color:rgba(166,255,245,.56) !important;
      background:linear-gradient(132deg, rgba(15,118,110,.87), rgba(37,99,235,.68)) !important;
      box-shadow:0 11px 25px rgba(7,95,89,.25), inset 0 1px 0 rgba(255,255,255,.3) !important;
    }}
    body .gradio-container button.primary::before {{ background:linear-gradient(112deg, rgba(255,255,255,.44), rgba(255,255,255,.08) 42%, rgba(150,240,255,.33) 73%, rgba(255,255,255,.28)); }}
    body .gradio-container button.stop {{
      color:#8f2b20 !important;
      border-color:rgba(238,157,145,.7) !important;
      background:linear-gradient(132deg, rgba(255,244,241,.82), rgba(255,221,216,.53)) !important;
    }}
    body.app-night {{
      --ink:#e8f4f2;
      --muted:#a9bec7;
      --line:rgba(129,181,198,.34);
      --paper:#071426;
      --surface:rgba(6,21,38,.7);
      --teal:#55d8ca;
      --teal-dark:#8debe2;
      color-scheme:dark;
      background-color:#020a17 !important;
      background-image:linear-gradient(118deg, rgba(2,10,23,.2), rgba(4,17,37,.38)), url('{BACKGROUND_DARK_URL}') !important;
      background-size:cover, 100vw auto !important;
      background-position:center, center top 24px !important;
      background-repeat:no-repeat !important;
      background-attachment:fixed !important;
    }}
    body.app-night .gradio-container {{
      color:var(--ink) !important;
      background-color:rgba(5,19,36,.62) !important;
      background-image:linear-gradient(135deg, rgba(5,19,36,.64), rgba(8,30,49,.44)), url('{BACKGROUND_DARK_URL}') !important;
      background-size:cover, 100vw auto !important;
      background-position:center, center top 24px !important;
      background-repeat:no-repeat !important;
      background-attachment:scroll, fixed !important;
      border-color:rgba(127,211,222,.22) !important;
      box-shadow:0 30px 100px rgba(0,0,0,.48), inset 0 1px 0 rgba(157,235,242,.12) !important;
      backdrop-filter:blur(32px) saturate(132%) brightness(.86);
      -webkit-backdrop-filter:blur(32px) saturate(132%) brightness(.86);
    }}
    body.app-night .gradio-container .workspace > .column,
    body.dark.app-night .gradio-container .workspace > .column,
    html body.dark.app-night .gradio-container .contain .workspace > .column {{
      background:linear-gradient(140deg, rgba(7,25,45,.76), rgba(8,37,55,.55)) !important;
      border-color:rgba(115,195,208,.28) !important;
      box-shadow:0 14px 42px rgba(0,0,0,.28), inset 0 1px 0 rgba(177,238,243,.1) !important;
    }}
    body.app-night .gradio-container .block,
    body.app-night .gradio-container .gr-box,
    body.app-night .gradio-container .form,
    body.app-night .gradio-container .input-container,
    body.app-night .gradio-container .gr-file,
    body.app-night .gradio-container .gr-panel {{
      background:rgba(5,20,37,.68) !important;
      color:var(--ink) !important;
      border-color:rgba(120,184,199,.31) !important;
    }}
    body.app-night .gradio-container .wrap {{ background:transparent !important; }}
    body.app-night .gradio-container input,
    body.app-night .gradio-container textarea,
    body.app-night .gradio-container select {{
      background:rgba(2,12,26,.76) !important;
      color:var(--ink) !important;
      border-color:rgba(118,183,199,.4) !important;
    }}
    body.app-night .gradio-container label,
    body.app-night .gradio-container .prose,
    body.app-night .gradio-container .prose *,
    body.app-night .gradio-container .gr-markdown,
    body.app-night .gradio-container .gr-markdown * {{ color:var(--ink) !important; }}
    body.app-night .gradio-container label[data-testid$="-radio-label"],
    body.app-night .gradio-container label[data-testid$="-checkbox-label"] {{
      background:rgba(6,24,42,.78) !important;
      color:var(--ink) !important;
      border-color:rgba(115,185,200,.32) !important;
    }}
    body.app-night .gradio-container label[data-testid$="-radio-label"].selected,
    body.app-night .gradio-container label[data-testid$="-checkbox-label"].selected {{
      background:rgba(20,104,112,.48) !important;
      border-color:rgba(89,222,208,.62) !important;
    }}
    body.app-night #run-panel {{
      background:rgba(6,33,49,.68) !important;
      border-color:rgba(81,192,187,.38) !important;
    }}
    body.app-night #api-key-guide {{
      background:rgba(6,31,47,.62) !important;
      color:var(--muted) !important;
      border-color:rgba(82,197,192,.35) !important;
    }}
    body.app-night #theme-toggle {{
      color:var(--ink);
      border-color:rgba(105,219,210,.42);
      background:linear-gradient(135deg,rgba(7,30,50,.82),rgba(20,82,91,.5));
      box-shadow:0 8px 21px rgba(0,0,0,.35),inset 0 1px 0 rgba(160,239,241,.16);
    }}
    body.app-night .brand-mark {{ color:#7ce5da; border-color:rgba(111,218,210,.46); background:rgba(4,20,36,.38); }}
    body.app-night .gradio-container button.secondary {{
      color:var(--ink) !important;
      border-color:rgba(115,205,205,.38) !important;
      background:linear-gradient(135deg,rgba(8,32,52,.84),rgba(15,71,78,.52)) !important;
      box-shadow:0 8px 20px rgba(0,0,0,.28),inset 0 1px 0 rgba(175,239,242,.14) !important;
    }}
    body.app-night .gradio-container button.stop {{
      color:#ffd3ce !important;
      border-color:rgba(238,139,137,.42) !important;
      background:linear-gradient(135deg,rgba(74,22,34,.78),rgba(115,42,48,.48)) !important;
    }}
    @media (prefers-reduced-motion:reduce) {{
      body .gradio-container button.primary, body .gradio-container button.secondary, body .gradio-container button.stop,
      body .gradio-container button.primary::before, body .gradio-container button.secondary::before, body .gradio-container button.stop::before {{ transition:none !important; }}
      body .gradio-container button.primary:hover, body .gradio-container button.secondary:hover, body .gradio-container button.stop:hover,
      body .gradio-container button.primary:active, body .gradio-container button.secondary:active, body .gradio-container button.stop:active {{ transform:none; }}
    }}
    """


THEME_HEADER_JS = r"""
const storageKey = "scan-pdf-translator-theme";
const button = element.querySelector("#theme-toggle");
const readSavedTheme = () => {
  try { return localStorage.getItem(storageKey) || "light"; }
  catch (_) { return "light"; }
};
const applyTheme = (theme) => {
  const night = theme === "dark";
  document.body.classList.toggle("app-night", night);
  document.documentElement.style.colorScheme = night ? "dark" : "light";
  if (button) {
    button.setAttribute("aria-pressed", night ? "true" : "false");
    button.title = night ? "切换为日间模式" : "切换为夜间模式";
    const icon = button.querySelector(".theme-icon");
    const label = button.querySelector(".theme-label");
    if (icon) icon.textContent = night ? "☀" : "◐";
    if (label) label.textContent = night ? "日间模式" : "夜间模式";
  }
};
applyTheme(readSavedTheme());
if (button && button.dataset.themeBound !== "1") {
  button.dataset.themeBound = "1";
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    const next = document.body.classList.contains("app-night") ? "light" : "dark";
    try { localStorage.setItem(storageKey, next); } catch (_) {}
    applyTheme(next);
  });
}
"""


with gr.Blocks(title="复杂扫描 PDF 翻译器", analytics_enabled=False) as demo:
    gr.HTML("""
    <div id="app-header">
      <div><h1>复杂扫描 PDF 翻译器</h1><p>面向技术教材、公式与复杂图表。默认输出上半原扫描、下半中文重构的双语 PDF；云端语义处理仅调用硅基流动。</p></div>
      <div class="header-actions">
        <div class="brand-mark">SCAN / TRANSLATE / VERIFY</div>
        <button id="theme-toggle" type="button" aria-pressed="false" title="切换为夜间模式"><span class="theme-icon">◐</span> <span class="theme-label">夜间模式</span></button>
      </div>
    </div>
    """, js_on_load=THEME_HEADER_JS)
    job_id = gr.State("")
    with gr.Row(elem_classes=["workspace"]):
        with gr.Column(scale=5):
            gr.HTML('<p class="section-label">01 / 文档与范围</p>')
            pdf = gr.File(label="上传 PDF", file_types=[".pdf"], type="filepath")
            with gr.Row():
                pages = gr.Textbox(label="页码", value="全部", info="默认翻译全文；手动输入支持不连续页", scale=2)
                with gr.Column(scale=3):
                    with gr.Row():
                        page_start = gr.Slider(1, 1000, value=1, step=1, label="起始页")
                        page_end = gr.Slider(1, 1000, value=1, step=1, label="结束页")
                    range_hint = gr.Markdown("<span class='status-copy'>默认：全文。移动滑条后将改为指定连续范围。</span>")
            apply_page_range = gr.Button("使用滑条范围", variant="secondary")
            preflight_box = gr.Markdown("<span class='status-copy'>上传后可先预检页数、类型和预计成本。</span>")
            inspect = gr.Button("预检页数与成本", variant="secondary")

        with gr.Column(scale=7):
            gr.HTML('<p class="section-label">02 / 翻译策略</p>')
            gr.HTML('<div id="api-key-guide">还没有 API Key？<a href="https://cloud.siliconflow.cn/" target="_blank" rel="noopener noreferrer">打开硅基流动获取 API Key ↗</a></div>')
            api_key = gr.Textbox(label="硅基流动 API Key", type="password", placeholder="sk-...", value=INITIAL_API_KEY, info="离开输入框后自动使用 Windows DPAPI 加密保存，仅当前 Windows 用户可解密")
            with gr.Row():
                api_key_status = gr.Markdown("<span class='status-copy'>" + ("已从当前用户的加密凭据中自动填充。" if INITIAL_API_KEY else "输入完整 Key 后离开输入框即可自动保存。") + "</span>", scale=5)
                clear_key_btn = gr.Button("清除本机密钥", variant="secondary", scale=1)
            with gr.Row():
                source_mode = gr.Radio([("自动选择", "auto"), ("硅基流动标准", "siliconflow"), ("硅基流动免费 API", "siliconflow_free"), ("完全本地", "local")], value="auto", label="模型来源")
                quality = gr.Radio([("快速", "fast"), ("均衡", "balanced"), ("最高质量", "high")], value="balanced", label="质量模式")
            output_labels = gr.CheckboxGroup(["双语纵向对照（默认）", "双语左右并排", "纯中文"], value=["双语纵向对照（默认）"], label="输出版本")
            with gr.Accordion("公式、图表与检查修复", open=False):
                formula = gr.Radio(["MathJax SVG 重绘并回退原公式", "只使用原公式裁图"], value="MathJax SVG 重绘并回退原公式", label="公式策略")
                figure = gr.Radio(["SVG 矢量描摹并保留原图回退", "确定性增强原图", "只保留高清原图"], value="SVG 矢量描摹并保留原图回退", label="图表策略")
                glossary = gr.Textbox(label="术语表", info="每行：英文=中文", lines=3)
                auto_repair_missing = gr.Checkbox(value=True, label="检查并修复（完成后自动补翻空译文）", info="仅重翻存在原文但译文为空的区块")
                repair_pages = gr.Textbox(label="检查并修复指定页码", info="例如 27,142,374-376；复用 OCR，只重译、重绘目标页")
                resume_job_id = gr.Textbox(label="恢复任务 ID", info="失败任务重新输入 Key 后可复用成功页面")
            start = gr.Button("开始翻译", variant="primary")

    with gr.Column(elem_id="run-panel"):
        gr.HTML('<p class="section-label">03 / 运行与交付</p>')
        with gr.Row():
            pause_btn = gr.Button("暂停", variant="secondary")
            resume_btn = gr.Button("继续", variant="secondary")
            cancel_btn = gr.Button("取消", variant="stop")
        status_box = gr.Markdown("<span class='status-copy'>尚未启动任务。</span>")
        progress_bar = gr.Slider(0, 100, value=0, interactive=False, label="真实任务进度")

    with gr.Tabs(elem_id="delivery-tabs"):
        with gr.Tab("交付文件"):
            downloads = gr.Files(label="PDF 与 QA 报告")
        with gr.Tab("本地扩展"):
            with gr.Row():
                local_bundle = gr.DownloadButton("下载本地扩展包（ZIP）", value=prepare_local_bundle())
                local_prepare = gr.Button("重新生成扩展包", variant="secondary")
                local_check = gr.Button("检查本地模式", variant="secondary")
                local_install = gr.Button("安装并启用", variant="primary")
            local_status = gr.Markdown("<span class='status-copy'>安装后可在模型来源中选择“完全本地”。</span>")
            install_progress = gr.Slider(0, 100, value=0, interactive=False, label="本地扩展安装进度")
            gr.Markdown("<div id='local-guide'>手动安装：在 <code>local-bootstrap</code> 目录运行 <code>powershell -NoProfile -ExecutionPolicy Bypass -File .\\install-and-enable-local.ps1 -Mode auto</code>。若 GPU 依赖失败，可使用 <code>-Mode cpu</code>。</div>")

    inspect.click(preflight, [pdf, pages, source_mode, quality], preflight_box)
    demo.load(load_saved_api_key_for_ui, outputs=[api_key, api_key_status], queue=False, show_progress="hidden", api_visibility="private")
    api_key.blur(save_api_key, inputs=api_key, outputs=api_key_status, queue=False, show_progress="hidden", api_visibility="private")
    api_key.submit(save_api_key, inputs=api_key, outputs=api_key_status, queue=False, show_progress="hidden", api_visibility="private")
    clear_key_btn.click(clear_api_key, outputs=[api_key, api_key_status], queue=False, show_progress="hidden", api_visibility="private")
    apply_page_range.click(pages_from_sliders, [page_start, page_end], [pages, range_hint])
    page_start.change(pages_from_sliders, [page_start, page_end], [pages, range_hint])
    page_end.change(pages_from_sliders, [page_start, page_end], [pages, range_hint])
    start.click(start_job, [pdf, api_key, pages, source_mode, quality, output_labels, formula, figure, glossary, resume_job_id, auto_repair_missing, repair_pages], [job_id, status_box, downloads])
    pause_btn.click(pause, job_id, status_box)
    resume_btn.click(resume, job_id, status_box)
    cancel_btn.click(cancel, job_id, status_box)
    local_prepare.click(prepare_local_bundle, outputs=local_bundle)
    local_check.click(check_local_mode, outputs=local_status)
    local_install.click(start_local_install, outputs=[local_status, install_progress])
    timer = gr.Timer(1.0)
    timer.tick(poll_local_install, outputs=[local_status, install_progress])
    timer.tick(poll, job_id, [status_box, progress_bar, downloads])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=int(os.environ.get("PDF_TRANSLATOR_PORT", "7860")), inbrowser=os.environ.get("PDF_TRANSLATOR_OPEN_BROWSER", "1") == "1", css=APP_CSS, allowed_paths=[str(ROOT / "assets")])
