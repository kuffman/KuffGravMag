# -*- mode: python ; coding: utf-8 -*-
import sys
import os

block_cipher = None

# Determine platform-specific icon safely
if sys.platform == 'win32':
    icon_file = 'assets/icon.ico' if os.path.exists('assets/icon.ico') else None
else:
    icon_file = 'assets/icon.icns' if os.path.exists('assets/icon.icns') else None

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('assets', 'assets')],
    hiddenimports=['engine.forward', 'engine.inversion', 'engine.report', 'engine.grid', 'reportlab', 'scipy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='KuffGravMag',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='KuffGravMag',
)

app = BUNDLE(
    coll,
    name='KuffGravMag.app',
    icon=icon_file,
    bundle_identifier='com.kuffouropoku.kuffgravmag',
)