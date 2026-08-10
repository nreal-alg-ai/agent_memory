#!/usr/bin/env bash
# 一键安装：向 ADB 设备安装 APK + 本地模型包，并启动 App。
# 用法：bash scripts/setup_device.sh --serial <DEVICE_SERIAL> [--apk <path>] [--pack-dir <path>]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
APK="$ROOT_DIR/android/app/build/outputs/apk/debug/app-debug.apk"
PACK_DIR="$ROOT_DIR/android/model-pack/x4000-sherpa-1.13.4-v2"
INSTALL_TOOL="$ROOT_DIR/android/tools/install_local_model_pack.py"
PACKAGE="com.agentmemory.test"

SERIAL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial)
      SERIAL="${2:-}"
      shift 2
      ;;
    --apk)
      APK="${2:-}"
      shift 2
      ;;
    --pack-dir)
      PACK_DIR="${2:-}"
      shift 2
      ;;
    *)
      echo "未知参数: $1" >&2
      echo "用法: $0 --serial <DEVICE_SERIAL> [--apk <path>] [--pack-dir <path>]" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SERIAL" ]]; then
  echo "缺少 --serial 参数" >&2
  echo "用法: $0 --serial <DEVICE_SERIAL> [--apk <path>] [--pack-dir <path>]" >&2
  exit 2
fi

if [[ ! -f "$APK" ]]; then
  echo "找不到 APK: $APK" >&2
  echo "请先构建：cd android && ./gradlew assembleDebug" >&2
  exit 1
fi

if [[ ! -f "$PACK_DIR/manifest.json" ]]; then
  echo "找不到模型包: $PACK_DIR（缺少 manifest.json）" >&2
  echo "请把模型包目录放到 android/model-pack/x4000-sherpa-1.13.4-v2/（见 android/model-pack/README.md）" >&2
  exit 1
fi

echo "[1/4] 检查设备 $SERIAL ..."
adb -s "$SERIAL" get-state >/dev/null

echo "[2/4] 安装 APK ..."
adb -s "$SERIAL" install -r "$APK"

echo "[3/4] 安装本地模型包 ..."
python3 "$INSTALL_TOOL" --serial "$SERIAL" --pack-dir "$PACK_DIR"

echo "[4/4] 启动 App ..."
adb -s "$SERIAL" shell am start -n "$PACKAGE/.MainActivity"

echo "完成。请在手机上完成模型自检后再开始使用。"
