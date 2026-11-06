"""Display metadata only. None of these values controls ASR or translation."""
import re

LANGUAGE_NAMES = dict(zip(
    'zh en yue ar de fr es pt id it ko ru th vi ja tr hi ms nl sv da fi pl cs fil fa el hu mk ro'.split(),
    '中文 英语 粤语 阿拉伯语 德语 法语 西班牙语 葡萄牙语 印度尼西亚语 意大利语 韩语 俄语 泰语 越南语 日语 土耳其语 印地语 马来语 荷兰语 瑞典语 丹麦语 芬兰语 波兰语 捷克语 菲律宾语 波斯语 希腊语 匈牙利语 马其顿语 罗马尼亚语'.split()))
ENGLISH_NAMES = dict(zip(
    'chinese english cantonese arabic german french spanish portuguese indonesian italian korean russian thai vietnamese japanese turkish hindi malay dutch swedish danish finnish polish czech filipino persian greek hungarian macedonian romanian'.split(),
    LANGUAGE_NAMES))
REVIEW_LANGUAGES = {'zh', 'en', 'ja', 'ko', 'yue'}


def native_language(log_slice):
    # The installed server's JSON echoes "auto". Its per-request diagnostic carries
    # the Qwen-generated language token. p=1.000 is NOT a confidence estimate.
    found = re.findall(r"^crispasr\[qwen3\]: detected '([^'\r\n]+)' \(p=[^\r\n]+\) via model output\r?$",
                       log_slice, re.MULTILINE)
    if len(found) != 1:
        return ''  # Missing/ambiguous metadata is not a reason to reuse a prior result.
    code = found[0].strip().lower()
    code = ENGLISH_NAMES.get(code, code)
    return code if code in LANGUAGE_NAMES else ''


def detected_language_label(cue):
    # Old recordings used an out-of-domain SenseVoice label. Do not display those
    # as if they were the primary model's recognized language.
    if cue.get('detected_language_source') != 'qwen3':
        return '待识别'
    return LANGUAGE_NAMES.get(cue.get('detected_language'), '待识别')
