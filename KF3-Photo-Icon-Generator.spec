# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['KF3-Photo-Icon-Generator.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
a.datas += [
    ('resource\\frame_3_1.png', '.\\resource\\frame_3_1.png', 'DATA'),
    ('resource\\frame_3_2.png', '.\\resource\\frame_3_2.png', 'DATA'),
    ('resource\\frame_4_1.png', '.\\resource\\frame_4_1.png', 'DATA'),
    ('resource\\frame_4_2.png', '.\\resource\\frame_4_2.png', 'DATA')
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='KF3-Photo-Icon-Generator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
