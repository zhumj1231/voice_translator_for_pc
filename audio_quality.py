"""Observable audio/text evidence, not calibrated model confidence."""
import audioop
import re


def content_key(text):
    return ''.join(c.lower() for c in text if c.isalnum())


def audio_evidence(pcm):
    if not pcm:
        return {'rms':0,'peak':0,'active_seconds':0.,'leading_quiet_seconds':0.}
    rms = audioop.rms(pcm,2)
    active = [i for i in range(0,len(pcm),640) if audioop.rms(pcm[i:i+640],2)>16]
    return {'rms':rms,'peak':audioop.max(pcm,2),
            'active_seconds':round(sum(min(640,len(pcm)-i) for i in active)/32000,3),
            'leading_quiet_seconds':round((active[0] if active else len(pcm))/32000,3)}


def weak_transcript(text, ranges, duration):
    """Short words without acoustic speech evidence need review, never deletion."""
    if ranges != [] or duration < 2:
        return False
    key = content_key(text)
    if not key:
        return True
    short = len(key)<8 or (text.isascii() and len(re.findall(r"[a-zA-Z]+",text))<=3)
    repetition = len(key)>=3 and len(set(key))<=2
    return short or repetition
