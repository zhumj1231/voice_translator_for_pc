# 部署与资源下载

## 1. Python 与硬件

使用 [Python 官方 Windows 下载页](https://www.python.org/downloads/windows/) 的 **Python 3.12 x64**，安装 Tcl/Tk 和 `py` launcher。当前代码使用 Windows `msvcrt`、WASAPI 及 `audioop`，不支持直接切换到 Python 3.13+。

在仓库根目录执行 `setup.cmd`；若不使用 `py` launcher，可在 PowerShell 中传入自己的 Python 路径：

```powershell
.\setup.ps1 -Python 'C:\Python312\python.exe'
```

脚本只安装 `requirements.txt` 中固定版本的 Python 包。没有模型自动下载，也不修改 VPN、系统代理或路由规则。下载大文件时请自行选用合适的网络路径，断点续传由下载工具处理。

主模型通过 Vulkan 在本机推理。本项目曾在 RTX 4080 Laptop 12 GB 显存、32 GB 内存配置上开发；这是已用配置，不是最低硬件要求或性能保证。7B Q8 翻译模型与 1.7B Q8 转写模型合计需要约 10 GB 量级的权重存储，运行时还需要额外内存。当前版本没有自动选择更小模型的安装流程。

## 2. 手动下载模型和运行时

下表为当前适配代码使用的版本和文件名。路径均相对于仓库根目录。运行时请解压完整分发包，保留 DLL、配置和许可文件；若压缩包多嵌套了一层目录，调整到下列实际路径。

| 用途 | 下载来源 / 当前版本 | 必须存在的文件 |
| --- | --- | --- |
| Qwen ASR 推理运行时 | [CrispASR v0.8.32](https://github.com/CrispStrobe/CrispASR/releases/tag/v0.8.32)，下载 `crispasr-windows-x86_64-vulkan.zip` | `assets/tools/crispasr/crispasr-windows-x86_64-vulkan/crispasr.exe` |
| 翻译推理运行时 | [llama.cpp b10809](https://github.com/ggml-org/llama.cpp/releases/tag/b10809)，下载 `llama-b10809-bin-win-vulkan-x64.zip` | `assets/tools/llama/llama-server.exe` |
| 主转写模型 | [cstr Qwen3-ASR 1.7B GGUF Q8](https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF/blob/dc60551b654abd7ba71ca0c19c3f6ba7e85c9242/qwen3-asr-1.7b-q8_0.gguf)；[原模型](https://github.com/QwenLM/Qwen3-ASR) | `assets/tools/models/qwen3-asr-1.7b-q8_0.gguf` |
| 中文翻译模型 | [Tencent HY-MT2-7B Q8 GGUF](https://huggingface.co/tencent/Hy-MT2-7B-GGUF/blob/bf7cf0410641d7c476a9bb4ab040e8541fe9f102/HY-MT2-7B-Q8_0.gguf) | `assets/tools/models/HY-MT2-7B-Q8_0.gguf` |
| 语音活动检测 | [ggml-org whisper-vad](https://huggingface.co/ggml-org/whisper-vad/tree/main)，下载 `ggml-silero-v6.2.0.bin` | `assets/tools/models/ggml-silero-v6.2.0.bin` |
| 本地复核（可选） | [sherpa-onnx ASR 模型](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)，下载 `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2` | `assets/review-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx` 及同目录 `tokens.txt` |
| 说话人分段（可选） | [sherpa-onnx 分段模型](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models)，选择 `sherpa-onnx-pyannote-segmentation-3-0` | `assets/speaker-models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx` |
| 声纹向量（可选） | [sherpa-onnx 声纹模型](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)，下载 `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` | `assets/speaker-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` |

GitHub release 页面可能需要展开 Assets。Hugging Face 文件页面使用 Download 下载真实文件，不能把网页或 Git LFS 指针另存为模型。第三方权重的许可证可能与推理库不同，请分别查阅来源。

推荐目录结构：

```text
voice_translator_for_pc/
  .venv/
  config.json
  assets/
    tools/
      crispasr/crispasr-windows-x86_64-vulkan/crispasr.exe  (+ DLL 等)
      llama/llama-server.exe                               (+ DLL 等)
      models/
        qwen3-asr-1.7b-q8_0.gguf
        HY-MT2-7B-Q8_0.gguf
        ggml-silero-v6.2.0.bin
    review-models/
      sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/
        model.int8.onnx
        tokens.txt
    speaker-models/
      sherpa-onnx-pyannote-segmentation-3-0/model.onnx
      3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
  data/sessions/                                           (运行后生成)
```

不使用可选模型时，在 `config.json` 设 `local_review.enabled=false` 或 `speakers.enabled=false`。默认两者开启。关闭 `vad` 可不加载 Silero，但会影响静音检测和切段效果。

已有模型无需重复复制或下载，可通过下列环境变量指定根目录，内部文件层级仍按上表：

| 环境变量 | 默认目录 |
| --- | --- |
| `VOICE_TRANSLATOR_TOOLS_DIR` | `assets/tools` |
| `VOICE_TRANSLATOR_REVIEW_MODELS_DIR` | `assets/review-models` |
| `VOICE_TRANSLATOR_SPEAKER_MODELS_DIR` | `assets/speaker-models` |
| `VOICE_TRANSLATOR_DATA_DIR` | `data` |

绝对路径可指向其他磁盘；相对路径始终相对于项目根目录，而非终端工作目录。设置后重新启动应用。运行检查：

```powershell
.\.venv\Scripts\python.exe -X utf8 check_setup.py
```

该检查只验证平台、Python 包与文件存在，不加载模型、不捕获声音、不调用云端，不代表 DLL、显卡驱动和推理结果已验证。

## 可选云端复核

### xAI

新安装的 `cloud.enabled=false`。双击 `setup-xai.cmd`，在隐藏输入的本地提示中填写 Key；脚本保存到当前 Windows 用户的 `XAI_API_KEY` 环境变量，不写入项目文件。环境变量不是加密保险库。关闭并重新打开字幕工具，使新进程读取变量。

已经通过自己的方法配置 `XAI_API_KEY` 时，无需再次运行脚本。不要把 Key 写进 `config.json`。

切换 provider：

```powershell
.\.venv\Scripts\python.exe set-cloud-provider.py xai
```

然后在 UI 勾选“云端复核”，或将 `config.json` 中的 `cloud.enabled` 改成 `true` 后重新启动。该命令只切换 provider，不会自动启用付费请求。默认模型为 `grok-voice-transcribe-2.0`，通过 `POST https://api.x.ai/v1/stt` 上传音频片段。

默认语言自动识别（`force_lang` 为空），`diarization=true`，返回的说话人编号只在当前云端片段内有效，不能等同于本地会话 A/B/C。`keyterms` 可按需填写术语。

默认 `proxy_url=null`，使用直连套接字，不自动读取系统代理或 `HTTP_PROXY`。如网络需要本机代理，显式设置 `cloud.proxy_url`，例如 `http://127.0.0.1:7890`，端口以本机代理实际配置为准；目前只接受本机 HTTP 代理。TUN/VPN 仍可影响系统路由，服务商的地区、账户及网络策略也会影响请求，应用不保证任意节点都可访问。

单文件最小测试（会提交音频并产生服务商费用）：

```powershell
.\.venv\Scripts\python.exe -X utf8 xai_stt.py .\clip.mp3
# 如需本机代理：
.\.venv\Scripts\python.exe -X utf8 xai_stt.py .\clip.mp3 --proxy http://127.0.0.1:7890 --json
```

`clip.mp3` 由使用者自行提供。可用 `--force-lang ja` 指定语言，通常保持自动识别。CLI 默认直连，独立于应用 `config.json`；它也不受会话预算约束。`--max-retries 0` 可关闭 429 重试。

应用默认只复核不确定片段，每会话最多提交 **60 秒音频、12 次请求尝试**。429 重试也计入应用预算，网络超时不自动重传，避免未知结果下重复计费。额度可以在 `config.json` 调整；这不是云端账户的消费封顶，其他程序使用同一个 Key 不受本工具约束。

### Gemini 兼容入口

原有 Gemini adapter 保留，可执行 `set-cloud-provider.py gemini`，通过 `setup-gemini.cmd` 保存到 Windows 凭据管理器，或设置 `GEMINI_API_KEY`。切换后的 model、endpoint 应与账户实际可用 API 匹配。此仓库的离线测试不验证服务商当前可用性；xAI 为默认集成路径。

Gemini 旧 adapter 的显式代理目前只接受 `http://127.0.0.1:7890`；未配置时使用 Python `urllib` 默认网络行为，可能读取系统代理。这与上面的 xAI 套接字直连实现不同。

## 故障定位

- 启动无窗口：在终端运行 `.\.venv\Scripts\python.exe -X utf8 app.py` 查看 Python 错误，先检查 3.12/Tcl/Tk/依赖。
- 模型启动失败：查看当前 `data/sessions/<会话>/asr.log`、`translation.log`；确认不是 LFS 指针、目录没有多套一层、DLL 完整、Vulkan 驱动可用。
- 未采集到视频声音：确认播放器输出到 Windows 默认播放设备；换设备后停止并重新开始。
- 延迟增加：先停掉其他占用显卡的程序。`refresh_seconds` 是刷新目标而非固定响应时间，首次加载和预热更慢。
- 角色分组或本地复核失败：查看同一会话里的 `speakers.log`、`local-review.log` 和 `events.jsonl`；可关闭相应功能继续主转写。
- 云端失败：核对 provider、Key、余额及 `proxy_url`，按界面提示处理；本地结果仍保留。不要发布包含私人录音或对话的整个会话目录。

升级 CrispASR 时需重新验证 `backends.py` 中的命令参数、VAD schema 和语言日志解析；本项目当前按上表固定版本适配。
