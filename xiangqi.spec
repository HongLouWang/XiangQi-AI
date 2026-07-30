# ruff: noqa: F821
"""PyInstaller configuration for the macOS 中国象棋.app bundle."""

from PyInstaller.utils.hooks import collect_submodules

service_hidden = []
for package in ("fastapi", "pydantic", "pydantic_core", "starlette", "uvicorn"):
    service_hidden += collect_submodules(package)

analysis = Analysis(
    ["src/xiangqi/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=[],
    hiddenimports=service_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="中国象棋",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    name="中国象棋",
)
app = BUNDLE(
    collection,
    name="中国象棋.app",
    icon=None,
    bundle_identifier="com.xiangqi.desktop",
    info_plist={
        "CFBundleDisplayName": "中国象棋",
        "NSHighResolutionCapable": True,
    },
)
