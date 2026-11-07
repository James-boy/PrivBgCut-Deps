# PrivBgCut-Deps

Prebuilt binary dependencies for [PrivBgCut](https://github.com/James-boy/PrivBgCut),
kept in a separate repository so they don't pollute the application's release stream.

This repo has **no source code**. It only hosts release assets that the main
PrivBgCut build workflows download at build time.

## Why a separate repository?

PrivBgCut's public release repo (`PrivBgCut-Releases`) serves end-user installers
via `/releases/latest`. Adding dependency artifacts there would make them the
"latest release" and silently break the download redirect. Keeping deps here
isolates that concern entirely.

## Contents

| Tag | Assets | Used by |
|---|---|---|
| `deps-ort-v1.23.2` | `libonnxruntime-1.23.2-{x86_64,arm64,universal2}.dylib` | macOS build |

## ONNX Runtime dylibs

Three architectures are published per version, so builds never need to
re-download PyPI wheels or merge them:

| Asset | Architecture | Use case |
|---|---|---|
| `libonnxruntime-1.23.2-x86_64.dylib` | Intel only | Intel-only macOS build |
| `libonnxruntime-1.23.2-arm64.dylib` | Apple Silicon only | ARM-only macOS build |
| `libonnxruntime-1.23.2-universal2.dylib` | Both (fat binary) | Universal macOS build |

Source: official PyPI wheels `onnxruntime==1.23.2` for `macosx_*_x86_64`
and `macosx_*_arm64`, merged with `llvm-lipo`.

### Direct download URLs

    https://github.com/James-boy/PrivBgCut-Deps/releases/download/deps-ort-v1.23.2/libonnxruntime-1.23.2-x86_64.dylib
    https://github.com/James-boy/PrivBgCut-Deps/releases/download/deps-ort-v1.23.2/libonnxruntime-1.23.2-arm64.dylib
    https://github.com/James-boy/PrivBgCut-Deps/releases/download/deps-ort-v1.23.2/libonnxruntime-1.23.2-universal2.dylib

### Usage in PrivBgCut build

Place the dylib at `src-tauri/libonnxruntime.dylib` before `tauri build`.
The file is picked up by `bundle.macOS.frameworks` in `tauri.conf.json` and
signed into the resulting `.app`.

Example (Intel-only build):

    curl -fL -o src-tauri/libonnxruntime.dylib \
      https://github.com/James-boy/PrivBgCut-Deps/releases/download/deps-ort-v1.23.2/libonnxruntime-1.23.2-x86_64.dylib

### Compatibility

- **ORT version** must match the `ort` crate's `api-XX` feature in
  `src-tauri/Cargo.toml`. For example, `api-23` ↔ ONNX Runtime 1.23.x.
  Upgrading one requires upgrading the other.
- If `ort` is bumped to `api-24`, publish a new tag
  `deps-ort-v1.24.0` here with freshly merged dylibs for that version.

## Regenerating assets

The dylibs are produced by `.github/workflows/fetch-ort-dylibs.yml`. To publish
a new ORT version:

1. Go to **Actions → fetch-ort-dylibs → Run workflow**
2. Enter the ORT version (e.g. `1.24.0`)
3. A new tag `deps-ort-v1.24.0` is created with three assets

The workflow is idempotent — re-running the same version replaces the existing
assets in that tag.
