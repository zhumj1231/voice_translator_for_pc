# Voice Translator for PC · 本地双语字幕

Windows 系统播放声音的实时转写与中文翻译。适用于浏览器视频、B 站、本地播放器和游戏对白；通过 WASAPI loopback 采集默认播放设备，无需麦克风，也无需上传整部视频。

原文保持音频中的语言，译文固定为简体中文。原文先显示并持续修订，翻译和复核可以稍后返回；界面会标明旧版原文对应的译文，避免将两版内容混在一起。

## 功能与当前边界

- 本地转写：Qwen3-ASR 1.7B Q8，语言自动识别；语言标签是识别结果，不能用于强制改变转写语言。
- 本地翻译：HY-MT2-7B Q8，经 llama.cpp 运行。
- 可选本地复核：SenseVoiceSmall，支持中文、英文、日文、韩文、粤语；这不是主转写模型的完整语言范围。
- 可选角色区分：说话人分段与声纹相似度匹配，显示本次会话的 A/B/C 分组。不会识别角色姓名；短句、重叠说话、变声和强 BGM 可能导致误分或待确认。
- 可选云端复核：xAI STT 为默认 provider，保留 Gemini adapter。新安装默认关闭云端；启用后只提交触发复核的音频片段，失败时保留本地结果。
- 会话自动保存 JSON、事件日志和双语 SRT；默认最多保存 120 秒待复核音频，可关闭。

这是一套持续修订的字幕工具。它没有预先读取播放器时间轴；SRT 从采集开始计时，导入 PotPlayer 前需自行对齐视频时间。处理赶不上播放时会跳过积压音频并记录缺口。没有保证所有对白、角色或语言都识别正确。

## 安装与启动

需要 **Windows x64、Python 3.12 x64（含 Tcl/Tk）以及支持 Vulkan 的显卡驱动**。当前默认模型占用数 GB 磁盘与较多显存，详见部署文档；尚未针对低显存设备整理自动降级方案。

```powershell
git clone https://github.com/zhumj1231/voice_translator_for_pc.git
cd voice_translator_for_pc
.\setup.cmd
```

`setup.cmd` 创建 `.venv`，安装 Python 依赖并生成本地 `config.json`。**它不会下载模型权重或推理程序**。首次检查提示资源缺失是预期结果。

1. 按 [部署与资源下载](docs/DEPLOYMENT.md) 放好模型、运行时及附带 DLL。
2. 运行 `.\.venv\Scripts\python.exe -X utf8 check_setup.py` 检查文件和依赖。
3. 双击 `start.cmd`。点击“开始”，等待模型加载、预热完成后播放视频；点击“停止”结束采集。
4. 点击“查看记录”打开当前会话目录。启动入口可自行创建桌面快捷方式。

配置 xAI Key、代理、资源目录及单文件 API 测试也见 [部署说明](docs/DEPLOYMENT.md#可选云端复核)。

## 数据与资源

仓库只包含源码、测试、示例配置和文档。模型、运行时分发包、Python 虚拟环境、API Key、私人配置、录音、字幕及会话日志均不纳入 Git。

本地数据默认位于 `data/sessions/`；`config.json` 为本机配置。云端默认关闭，启用前应了解会向所选服务商发送短音频。`diagnostics.save_uncertain_audio=false` 可关闭待复核音频落盘；文字和事件记录仍保存在本机。模型及第三方程序的许可见各自上游仓库，下载时保留其许可证文件。

## 开发与验证

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p "test_*.py"
git add .
.\.venv\Scripts\python.exe scripts_check_release.py
```

单元测试不需要模型或 API Key，不发起付费请求。GitHub Actions 使用 Windows + Python 3.12 执行同一套测试和 Git 暂存区检查；它验证逻辑与发布内容，不验证显卡推理速度或识别准确率。

主流程和模块位置见 [代码结构](docs/ARCHITECTURE.md)。
