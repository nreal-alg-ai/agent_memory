#!/usr/bin/env python3
"""Build Chaquopy's _sqlite3 extension with SQLite FTS5 statically enabled."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile


SQLITE_URL = "https://www.sqlite.org/2025/sqlite-amalgamation-3500400.zip"
SQLITE_SHA256 = "1d3049dd0f830a025a53105fc79fd2ab9431aea99e137809d064d8ee8356b032"
PYTHON_URL = "https://www.python.org/ftp/python/3.11.14/Python-3.11.14.tgz"
PYTHON_SHA256 = "563d2a1b2a5ba5d5409b5ecd05a0e1bf9b028cf3e6a6f0c87a5dc8dc3f2d9182"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, target: Path, expected_sha256: str) -> None:
    if target.is_file() and sha256(target) == expected_sha256:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, temporary)
    actual = sha256(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {url}: {actual}")
    temporary.replace(target)


def ensure_sources(cache_dir: Path) -> tuple[Path, Path]:
    sqlite_archive = cache_dir / "sqlite-amalgamation-3500400.zip"
    python_archive = cache_dir / "cpython-3.11.14.tar.gz"
    download_verified(SQLITE_URL, sqlite_archive, SQLITE_SHA256)
    download_verified(PYTHON_URL, python_archive, PYTHON_SHA256)

    sqlite_dir = cache_dir / "sqlite-amalgamation-3500400"
    if not (sqlite_dir / "sqlite3.c").is_file():
        with zipfile.ZipFile(sqlite_archive) as archive:
            archive.extractall(cache_dir)

    python_dir = cache_dir / "Python-3.11.14"
    if not (python_dir / "Modules/_sqlite/module.c").is_file():
        with tarfile.open(python_archive, "r:gz") as archive:
            archive.extractall(cache_dir)
    return sqlite_dir, python_dir


def find_clang(ndk_root: Path, abi: str, api: int) -> Path:
    host = "darwin-x86_64" if os.uname().sysname == "Darwin" else "linux-x86_64"
    tool = {
        "arm64-v8a": "aarch64-linux-android",
        "armeabi-v7a": "armv7a-linux-androideabi",
        "x86_64": "x86_64-linux-android",
        "x86": "i686-linux-android",
    }[abi]
    result = ndk_root / "toolchains" / "llvm" / "prebuilt" / host / "bin" / f"{tool}{api}-clang"
    if not result.is_file():
        raise FileNotFoundError(f"Android NDK clang not found: {result}")
    return result


def find_target_root(target_zip: Path, cache_dir: Path) -> Path:
    target_root = cache_dir / "chaquopy-target"
    marker = target_root / "include/python3.11/pyconfig.h"
    if not marker.is_file():
        shutil.rmtree(target_root, ignore_errors=True)
        target_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target_zip) as archive:
            archive.extractall(target_root)
    return target_root


def build_extension(
    *,
    output: Path,
    sqlite_dir: Path,
    python_dir: Path,
    target_root: Path,
    clang: Path,
    abi: str,
    api: int,
) -> None:
    sources = sorted((python_dir / "Modules/_sqlite").glob("*.c"))
    include_dir = target_root / "include/python3.11"
    native_dir = target_root / "jniLibs" / abi
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    command = [
        str(clang),
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c11",
        "-DANDROID",
        "-DSQLITE_ENABLE_FTS5",
        "-DSQLITE_ENABLE_JSON1",
        "-DSQLITE_THREADSAFE=1",
        "-Wl,-Bsymbolic",
        f"-I{include_dir}",
        f"-I{python_dir / 'Include'}",
        f"-I{python_dir}",
        f"-I{sqlite_dir}",
        *(str(source) for source in sources),
        str(sqlite_dir / "sqlite3.c"),
        f"-L{native_dir}",
        "-Wl,-rpath,$ORIGIN",
        "-Wl,-soname,_sqlite3.cpython-311.so",
        "-Wl,-z,max-page-size=16384",
        "-lm",
        "-lsqlite3_python",
        "-lpython3.11",
        "-ldl",
        "-o",
        str(output),
    ]
    print("Building FTS5-enabled _sqlite3:", " ".join(command))
    subprocess.run(command, check=True)


def patch_stdlib(stdlib_imy: Path, extension: Path) -> None:
    if not stdlib_imy.is_file():
        raise FileNotFoundError(f"Chaquopy stdlib archive not found: {stdlib_imy}")
    with tempfile.TemporaryDirectory(prefix="chaquopy-stdlib-") as temporary:
        temporary_dir = Path(temporary)
        with zipfile.ZipFile(stdlib_imy) as archive:
            archive.extractall(temporary_dir)
        target = temporary_dir / "_sqlite3.cpython-311.so"
        if not target.is_file():
            raise FileNotFoundError("_sqlite3.cpython-311.so is missing from the Chaquopy stdlib archive")
        shutil.copy2(extension, target)

        replacement = stdlib_imy.with_suffix(stdlib_imy.suffix + ".part")
        replacement.unlink(missing_ok=True)
        with zipfile.ZipFile(replacement, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(temporary_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(temporary_dir).as_posix())
        replacement.replace(stdlib_imy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--target-zip", type=Path, required=True)
    parser.add_argument("--stdlib-imy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ndk-root", type=Path, required=True)
    parser.add_argument("--abi", default="arm64-v8a")
    parser.add_argument("--api", type=int, default=26)
    args = parser.parse_args()

    sqlite_dir, python_dir = ensure_sources(args.cache_dir)
    target_root = find_target_root(args.target_zip, args.cache_dir)
    clang = find_clang(args.ndk_root, args.abi, args.api)
    build_extension(
        output=args.output,
        sqlite_dir=sqlite_dir,
        python_dir=python_dir,
        target_root=target_root,
        clang=clang,
        abi=args.abi,
        api=args.api,
    )
    patch_stdlib(args.stdlib_imy, args.output)


if __name__ == "__main__":
    main()
