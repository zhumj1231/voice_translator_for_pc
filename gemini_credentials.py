"""Task-specific Windows credential storage. Never prints credential contents."""
import ctypes
from ctypes import wintypes as wt
import os

TARGET = 'LocalSubtitles/GeminiAPI'


class Credential(ctypes.Structure):
    _fields_ = [('Flags', wt.DWORD), ('Type', wt.DWORD), ('TargetName', wt.LPWSTR),
                ('Comment', wt.LPWSTR), ('LastWritten', wt.FILETIME),
                ('CredentialBlobSize', wt.DWORD), ('CredentialBlob', ctypes.POINTER(wt.BYTE)),
                ('Persist', wt.DWORD), ('AttributeCount', wt.DWORD),
                ('Attributes', ctypes.c_void_p), ('TargetAlias', wt.LPWSTR), ('UserName', wt.LPWSTR)]


def _api():
    api = ctypes.WinDLL('Advapi32.dll', use_last_error=True)
    api.CredWriteW.argtypes = [ctypes.POINTER(Credential), wt.DWORD]
    api.CredWriteW.restype = wt.BOOL
    api.CredReadW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD,
                            ctypes.POINTER(ctypes.POINTER(Credential))]
    api.CredReadW.restype = wt.BOOL
    api.CredFree.argtypes = [ctypes.c_void_p]
    api.CredFree.restype = None
    return api


def save_key(key):
    key = key.strip()
    if not 16 <= len(key) <= 2000 or any(ord(c) < 33 or ord(c) > 126 for c in key):
        raise ValueError('请输入完整的 API Key，不要包含空格或换行。')
    blob = key.encode('utf-8')
    buffer = (wt.BYTE * len(blob)).from_buffer_copy(blob)
    cred = Credential(Type=1, TargetName=TARGET, Comment='Local subtitles Gemini API',
                      CredentialBlobSize=len(blob), CredentialBlob=buffer,
                      Persist=2, UserName='GeminiAPI')
    try:
        if not _api().CredWriteW(ctypes.byref(cred), 0):
            raise OSError(ctypes.get_last_error(), 'Windows 凭据保存失败')
    finally:
        ctypes.memset(buffer, 0, len(blob))


def get_key(env_name='GEMINI_API_KEY'):
    key = os.environ.get(env_name, '').strip()
    if key:
        return key
    if os.name != 'nt':
        return ''
    api = _api()
    pointer = ctypes.POINTER(Credential)()
    if not api.CredReadW(TARGET, 1, 0, ctypes.byref(pointer)):
        if ctypes.get_last_error() == 1168:
            return ''
        raise OSError(ctypes.get_last_error(), 'Windows 凭据读取失败')
    try:
        return ctypes.string_at(pointer.contents.CredentialBlob,
                                pointer.contents.CredentialBlobSize).decode('utf-8')
    finally:
        api.CredFree(pointer)


def key_entry():
    import tkinter as tk
    from tkinter import ttk
    window = tk.Tk()
    window.title('本地字幕工具 · Gemini API Key')
    window.geometry('620x260')
    window.resizable(False, False)
    panel = ttk.Frame(window, padding=22)
    panel.pack(fill='both', expand=True)
    ttk.Label(panel, text='请在这里粘贴 Gemini API Key（不要发进聊天）', font=('Microsoft YaHei UI', 12)).pack(anchor='w')
    ttk.Label(panel, text='保存到当前 Windows 用户的凭据管理器；此窗口不联网、不收费。').pack(anchor='w', pady=(10, 12))
    entry = ttk.Entry(panel, show='●', width=76)
    entry.pack(fill='x')
    status = ttk.Label(panel, text='保存后重新打开字幕工具即可使用。')
    status.pack(anchor='w', pady=14)
    def save():
        try:
            save_key(entry.get())
        except ValueError as error:
            status.config(text=str(error))
            return
        except OSError:
            status.config(text='保存失败，未输出密钥。请检查当前用户的凭据管理器权限。')
            return
        entry.delete(0, 'end')
        entry.config(state='disabled')
        button.config(state='disabled')
        status.config(text='已保存。请关闭此窗口并重新打开字幕工具。')
    button = ttk.Button(panel, text='保存密钥', command=save)
    button.pack(side='left')
    ttk.Button(panel, text='关闭', command=window.destroy).pack(side='right')
    entry.focus_set()
    window.mainloop()


if __name__ == '__main__':
    key_entry()
