#!/usr/bin/env python3
from pathlib import Path
import shutil

root = Path('.').resolve()
for p in root.rglob('*'):
    if p.is_file() and p.name == 'db.sqlite3': p.unlink()
    elif p.is_dir() and p.name == '__pycache__': shutil.rmtree(p)
    elif p.is_dir() and p.name == 'migrations':
        [f.unlink() for f in p.glob('*.py') if f.name != '__init__.py']

