"""Audit the Git index (also in CI), never print credential matches.

Only small UTF-8 source/config/documentation files may enter the repository.
Runtime downloads and user-generated data belong outside Git.
"""
from pathlib import PurePosixPath
import re
import subprocess


def main():
    names = subprocess.check_output(['git', 'ls-files', '-z']).decode('utf-8').split('\0')
    failures, total, count = [], 0, 0
    allowed = {'.py', '.md', '.txt', '.json', '.cmd', '.ps1', '.yml', '.yaml'}
    special = {'.gitignore', '.gitattributes'}
    excluded = {'data', 'assets', 'models', 'tools', 'sessions', 'validation',
                'benchmark', '.venv', 'venv', '__pycache__', 'speaker-packages'}
    secrets = re.compile(
        r'(?<![\w-])(?:xai-[A-Za-z0-9_-]{32,}|AIza[A-Za-z0-9_-]{30,}|'
        r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})|'
        r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
    for name in filter(None, names):
        path = PurePosixPath(name)
        if (path.suffix.lower() not in allowed and name not in special
                or set(path.parts) & excluded
                or path.name in {'config.json', 'settings.json', 'captions.json'}
                or path.name.startswith('.env')):
            failures.append((name, 'not a publishable source/documentation file'))
            continue
        data = subprocess.check_output(['git', 'show', ':' + name])
        count += 1
        total += len(data)
        if len(data) > 256 * 1024:
            failures.append((name, 'larger than 256 KiB'))
        try:
            content = data.decode('utf-8-sig')
        except UnicodeDecodeError:
            failures.append((name, 'not UTF-8 text'))
            continue
        if '\0' in content:
            failures.append((name, 'binary content'))
        if secrets.search(content):
            failures.append((name, 'possible credential; value omitted'))
        if re.search(r'[A-Za-z]:[\\/]Users[\\/][^\s\\/]+', content):
            failures.append((name, 'personal Windows profile path'))
    if not count:
        failures.append(('index', 'no staged/tracked source files'))
    if total > 2 * 1024 * 1024:
        failures.append(('index', 'total source exceeds 2 MiB; review additions'))
    for name, reason in failures:
        print(f'FAIL {name}: {reason}')
    print(f'Checked {count} text files, {total:,} bytes; {len(failures)} issue(s).')
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
