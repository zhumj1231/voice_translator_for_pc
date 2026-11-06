# 代码结构

当前保持原应用的平铺 Python 模块结构，整理资源路径和部署入口，避免为发布重写 UI 与识别流程。

| 文件 | 职责 |
| --- | --- |
| `app.py` | Tkinter UI、WASAPI loopback、会话生命周期、互斥锁与 WAV 验证回放 |
| `engine.py` | 音频环形缓冲、切段、原文修订、翻译/复核队列、预算与字幕导出 |
| `backends.py` | 本地推理进程/HTTP adapter、中文翻译、`CloudReviewer` provider 分发 |
| `xai_stt.py` | xAI multipart REST、响应解析、网络/429 错误处理、独立文件转写 CLI |
| `gemini_network.py` / `gemini_credentials.py` | Gemini 网络请求与 Windows 凭据读取/设置 |
| `audio_quality.py` / `languages.py` | 音频/文本可靠性判断、语言检测结果映射 |
| `speakers.py` / `speaker_service.py` | 会话内声纹分组、独立 CPU 服务 |
| `local_review_service.py` | SenseVoice CPU 复核服务 |
| `paths.py` | 可覆盖的资源与数据目录、示例配置回退 |
| `check_setup.py` / `setup.ps1` / `start.cmd` | 环境检查、Python 安装流程、启动入口 |
| `tests/` | 合成输入与 mock adapter 的回归检查，无需真实音视频或云端 Key |

音频统一到 16 kHz 单声道 PCM，流入环形缓冲。主转写循环生成和修订 cue；翻译、复核独立工作，队列合并旧版本任务并限制积压。译文记录对应的原文和 revision，旧请求不能覆盖新版译文。低可信原文会标记待复核，避免将猜测翻译成看似确定的中文。

云端入口是 `backends.CloudReviewer.review()`，当 provider 为 `xai` 时调用 `xai_stt.transcribe_once()`。`engine.py` 决定何时提交和执行预算限制。云端返回保留在 `review_original`、`review_language`、`review_words` 等字段中；不会静默覆盖本地主字幕，复核候选可有自己的中文译文。xAI 的词时间戳转换为会话相对时间后保存。

语言标签来源于当前转写请求对应的原生识别信息；文本脚本只能作为部分可靠性检查，不能把拉丁字母直接判成英文。主 ASR 始终自动识别语言，中文翻译由独立模型完成。

## 本地生成数据

`data/sessions/<时间-随机标识>/` 包含配置快照、`captions.json`、`events.jsonl`、`bilingual.srt` 和各推理服务日志。开启诊断时还可包含 `uncertain-audio/`。默认不保存完整播放音轨。

SRT 使用采集时间轴。将其作为电影外挂字幕时，需要手动校准开始时间及中途暂停/跳转造成的偏移；当前没有播放器同步接口。复核候选和词级信息主要在 UI/JSON 中查看，SRT 导出使用主字幕及其匹配译文。

## 验证范围

单元测试覆盖队列积压、修订一致性、静音和弱语音处理、说话人分组、语言标签、云端预算、multipart 顺序及响应校验。`scripts_check_release.py` 检查 Git index 中只包含小型文本源码和文档，并拦截模型、私人配置、媒体、明显凭据及用户目录路径；它不是完整秘密扫描器。

CI 不下载模型、不采集系统声音、不访问付费 API，因此不代表实际推理精度、实时性能或云端服务可用性已通过端到端测试。真实识别评估应在本地另行运行，测试录音及结果不提交到仓库。
