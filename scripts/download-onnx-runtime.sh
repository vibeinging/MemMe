#!/usr/bin/env bash
# Download the pinned ONNX Runtime dynamic library used by MemMe's local
# ONNX embedder (ort 2.0.0-rc.0 with load-dynamic needs ORT_DYLIB_PATH).
#
# Prints the absolute library path on success (for $(...) capture), or writes
# ORT_DYLIB_PATH to $GITHUB_ENV when running inside GitHub Actions.
set -euo pipefail
umask 077

version="1.19.2"

fail() {
    echo "$*" >&2
    exit 1
}

sha256_file() {
    case "$(uname -s)" in
        Darwin) shasum -a 256 "$1" | awk '{ print $1 }' ;;
        Linux) sha256sum "$1" | awk '{ print $1 }' ;;
        *) fail "Unsupported checksum host" ;;
    esac
}

detect_host() {
    if [[ -n "${MEMME_ONNXRUNTIME_HOST:-}" ]]; then
        echo "${MEMME_ONNXRUNTIME_HOST}"
        return
    fi
    if command -v rustc >/dev/null 2>&1; then
        rustc -vV | awk '/^host:/ { print $2 }'
        return
    fi
    case "$(uname -s):$(uname -m)" in
        Darwin:x86_64) echo "x86_64-apple-darwin" ;;
        Darwin:arm64|Darwin:aarch64) echo "aarch64-apple-darwin" ;;
        Linux:x86_64) echo "x86_64-unknown-linux-gnu" ;;
        Linux:arm64|Linux:aarch64) echo "aarch64-unknown-linux-gnu" ;;
        *) fail "Cannot select an ONNX Runtime release for this host" ;;
    esac
}

rust_host="$(detect_host)"

case "${rust_host}" in
    x86_64-apple-darwin)
        package="onnxruntime-osx-x86_64-${version}"
        expected_sha256="6536e36d7ea92e32d53dad7ddd0fdf10be5b62d1dace85a13e1295ff81e9b5d4"
        library_name="libonnxruntime.dylib"
        ;;
    aarch64-apple-darwin)
        package="onnxruntime-osx-arm64-${version}"
        expected_sha256="370c49770e2e1f243e17c7b227bb7f4b3da793b847d02f38016dc0e46c30fbe1"
        library_name="libonnxruntime.dylib"
        ;;
    x86_64-unknown-linux-gnu)
        package="onnxruntime-linux-x64-${version}"
        expected_sha256="eb00c64e0041f719913c4080e0fed7d9963dc3aa9b54664df6036d8308dbcd33"
        library_name="libonnxruntime.so"
        ;;
    aarch64-unknown-linux-gnu)
        package="onnxruntime-linux-aarch64-${version}"
        expected_sha256="5e30145277d6d6fcb0e8f14f0d0ab5048af7b13ffd608023bb1e2875621fab07"
        library_name="libonnxruntime.so"
        ;;
    *)
        fail "No ONNX Runtime ${version} release for host: ${rust_host}"
        ;;
esac

if [[ -n "${MEMME_ONNXRUNTIME_DIR:-}" ]]; then
    root="${MEMME_ONNXRUNTIME_DIR}"
elif [[ -n "${XDG_CACHE_HOME:-}" ]]; then
    root="${XDG_CACHE_HOME}/memme/onnxruntime/${version}/${rust_host}"
elif [[ -n "${HOME:-}" ]]; then
    root="${HOME}/.cache/memme/onnxruntime/${version}/${rust_host}"
else
    root="$(mktemp -d "${TMPDIR:-/tmp}/memme-onnxruntime-${version}.XXXXXX")"
fi

[[ ! -L "${root}" ]] || fail "Refusing symlinked cache directory: ${root}"
mkdir -p "${root}"
[[ -d "${root}" ]] || fail "Cannot create cache directory: ${root}"
chmod 700 "${root}" 2>/dev/null || true

archive_path="${root}/${package}.tgz"
download_url="https://github.com/microsoft/onnxruntime/releases/download/v${version}/${package}.tgz"

if [[ ! -f "${archive_path}" ]]; then
    archive_tmp="$(mktemp "${root}/.${package}.download.XXXXXX")"
    curl --fail --location --retry 3 --output "${archive_tmp}" "${download_url}"
    if [[ "$(sha256_file "${archive_tmp}")" != "${expected_sha256}" ]]; then
        fail "ONNX Runtime archive checksum mismatch for ${package}.tgz"
    fi
    mv "${archive_tmp}" "${archive_path}"
fi

if [[ "$(sha256_file "${archive_path}")" != "${expected_sha256}" ]]; then
    fail "ONNX Runtime archive checksum mismatch for ${package}.tgz"
fi

package_path="${root}/${package}"
library_path="${package_path}/lib/${library_name}"

if [[ ! -f "${library_path}" ]]; then
    staging_dir="$(mktemp -d "${root}/.extract.XXXXXX")"
    tar -xzf "${archive_path}" -C "${staging_dir}"
    staged_library="${staging_dir}/${package}/lib/${library_name}"
    [[ -f "${staged_library}" ]] || fail "ONNX Runtime library missing from archive: ${package}.tgz"
    rm -rf "${package_path}"
    mv "${staging_dir}/${package}" "${package_path}"
    rmdir "${staging_dir}" 2>/dev/null || true
fi

[[ -f "${library_path}" ]] || fail "ONNX Runtime library not found after extraction: ${library_path}"

if [[ -n "${GITHUB_ENV:-}" ]]; then
    echo "ORT_DYLIB_PATH=${library_path}" >> "${GITHUB_ENV}"
else
    echo "${library_path}"
fi
