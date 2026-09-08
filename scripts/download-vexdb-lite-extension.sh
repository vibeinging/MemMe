#!/usr/bin/env bash
set -euo pipefail
umask 077

release_tag="v0.0.17"

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

owner_uid() {
    case "$(uname -s)" in
        Darwin) stat -f '%u' -- "$1" ;;
        Linux) stat -c '%u' -- "$1" ;;
        *) fail "Unsupported ownership-check host" ;;
    esac
}

require_owned_file() {
    local path="$1"
    [[ -f "${path}" && ! -L "${path}" ]] || fail "Expected a regular, non-symlink file: ${path}"
    [[ "$(owner_uid "${path}")" == "$(id -u)" ]] || fail "File is not owned by the current user: ${path}"
}

prepare_private_dir() {
    local path="$1"
    [[ ! -L "${path}" ]] || fail "Refusing symlinked cache directory: ${path}"
    if [[ -e "${path}" && ! -d "${path}" ]]; then
        fail "Cache path exists but is not a directory: ${path}"
    fi
    mkdir -p "${path}"
    [[ -d "${path}" && ! -L "${path}" ]] || fail "Cannot create a private cache directory: ${path}"
    [[ "$(owner_uid "${path}")" == "$(id -u)" ]] || fail "Cache directory is not owned by the current user: ${path}"
    chmod 700 "${path}"
}

detect_host() {
    if [[ -n "${MEMME_VEXDB_LITE_HOST:-}" ]]; then
        echo "${MEMME_VEXDB_LITE_HOST}"
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
        *) fail "Cannot select a VexDB-Lite desktop release for this host" ;;
    esac
}

rust_host="$(detect_host)"

case "${rust_host}" in
    x86_64-apple-darwin)
        asset="vexdb-lite-sqlite-macos-x86_64.tar.gz"
        expected_sha256="3bb5f5f6167fd0a59f5a3162a5f907b4cd43306d131203073a7bbdc4745c1b12"
        expected_library_sha256="8e2bbe3c4767f59a1a78cc505cbd897ad69a7f9ea80f986ed9019d6c3cc4c699"
        library_name="vexdb_lite.dylib"
        ;;
    aarch64-apple-darwin)
        asset="vexdb-lite-sqlite-macos-arm64.tar.gz"
        expected_sha256="cf1443a86531a0a93074a26ada767700982cba90807321a88f04d55a1460baf6"
        expected_library_sha256="54eb0c94274d41070fa2944eb4374cf945c92abe53c4ce2ac2e9c005a1010c68"
        library_name="vexdb_lite.dylib"
        ;;
    x86_64-unknown-linux-gnu)
        asset="vexdb-lite-sqlite-linux-x86_64.tar.gz"
        expected_sha256="0a72c8ce91ac3ae6e4315715a22d42ce835c8f65e5131836f75c25290f4e6b4d"
        expected_library_sha256="0df0199b237289a2732528c67397357a8462b05fea2e6d8df646befe7615a5f7"
        library_name="vexdb_lite.so"
        ;;
    aarch64-unknown-linux-gnu)
        asset="vexdb-lite-sqlite-linux-aarch64.tar.gz"
        expected_sha256="afc742064cf4fed5a82b8b1c17609dd6ca150beba1746584241afa79bfa02569"
        expected_library_sha256="5df13758eb81431b11bc99da309f0b9de480b39312cbe6ed951397770db48cdb"
        library_name="vexdb_lite.so"
        ;;
    *)
        echo "No VexDB-Lite ${release_tag} desktop release for host: ${rust_host}" >&2
        exit 1
        ;;
esac

if [[ -n "${MEMME_VEXDB_LITE_RELEASE_DIR:-}" ]]; then
    release_root="${MEMME_VEXDB_LITE_RELEASE_DIR}"
elif [[ -n "${RUNNER_TEMP:-}" ]]; then
    release_root="${RUNNER_TEMP}/memme-vexdb-lite/${release_tag}/${rust_host}"
elif [[ -n "${XDG_CACHE_HOME:-}" ]]; then
    release_root="${XDG_CACHE_HOME}/memme/vexdb-lite/${release_tag}/${rust_host}"
elif [[ -n "${HOME:-}" ]]; then
    release_root="${HOME}/.cache/memme/vexdb-lite/${release_tag}/${rust_host}"
else
    release_root="$(mktemp -d "${TMPDIR:-/tmp}/memme-vexdb-lite-${release_tag}.XXXXXX")"
fi

case "${release_root}" in
    *$'\n'*|*$'\r'*) fail "Cache path must not contain newlines" ;;
esac

prepare_private_dir "${release_root}"
archive_path="${release_root}/${asset}"
download_url="https://github.com/VexDB-THU/VexDB-Lite/releases/download/${release_tag}/${asset}"

if [[ -e "${archive_path}" || -L "${archive_path}" ]]; then
    require_owned_file "${archive_path}"
else
    archive_tmp="$(mktemp "${release_root}/.${asset}.download.XXXXXX")"
    curl --fail --location --retry 3 --output "${archive_tmp}" "${download_url}"
    require_owned_file "${archive_tmp}"
    if [[ "$(sha256_file "${archive_tmp}")" != "${expected_sha256}" ]]; then
        fail "VexDB-Lite release checksum mismatch for ${asset}"
    fi
    mv "${archive_tmp}" "${archive_path}"
fi

require_owned_file "${archive_path}"
if [[ "$(sha256_file "${archive_path}")" != "${expected_sha256}" ]]; then
    fail "VexDB-Lite release checksum mismatch for ${asset}"
fi

archive_entries="$(tar -tzf "${archive_path}")"
[[ -n "${archive_entries}" ]] || fail "Empty VexDB-Lite archive: ${asset}"
while IFS= read -r entry; do
    case "${entry}" in
        ""|/*|".."|../*|*/../*|*/..) fail "Unsafe archive entry in ${asset}: ${entry}" ;;
    esac
done <<< "${archive_entries}"

first_entry="$(printf '%s\n' "${archive_entries}" | sed -n '1p')"
if [[ "${first_entry}" == */* ]]; then
    package_dir="${first_entry%%/*}"
    while IFS= read -r entry; do
        case "${entry}" in
            "${package_dir}"|"${package_dir}"/*) ;;
            *) fail "Archive has more than one top-level path: ${asset}" ;;
        esac
    done <<< "${archive_entries}"
    flat_archive=0
else
    package_dir="${asset%.tar.gz}"
    while IFS= read -r entry; do
        [[ "${entry}" != */* ]] || fail "Archive mixes flat and nested paths: ${asset}"
    done <<< "${archive_entries}"
    flat_archive=1
fi
case "${package_dir}" in
    ""|"."|".."|*[!A-Za-z0-9._-]*) fail "Unsafe package directory in ${asset}" ;;
esac

package_path="${release_root}/${package_dir}"
extension_path="${package_path}/${library_name}"

if [[ -e "${package_path}" || -L "${package_path}" ]]; then
    [[ -d "${package_path}" && ! -L "${package_path}" ]] || fail "Refusing unsafe cached package path: ${package_path}"
    [[ "$(owner_uid "${package_path}")" == "$(id -u)" ]] || fail "Cached package is not owned by the current user: ${package_path}"
else
    staging_dir="$(mktemp -d "${release_root}/.extract.XXXXXX")"
    chmod 700 "${staging_dir}"
    if [[ "${flat_archive}" == 1 ]]; then
        mkdir "${staging_dir}/${package_dir}"
        tar -xzf "${archive_path}" -C "${staging_dir}/${package_dir}"
    else
        tar -xzf "${archive_path}" -C "${staging_dir}"
    fi
    staged_extension="${staging_dir}/${package_dir}/${library_name}"
    require_owned_file "${staged_extension}"
    if [[ "$(sha256_file "${staged_extension}")" != "${expected_library_sha256}" ]]; then
        fail "VexDB-Lite dynamic library checksum mismatch for ${asset}"
    fi
    mv "${staging_dir}/${package_dir}" "${package_path}"
    rmdir "${staging_dir}"
fi

require_owned_file "${extension_path}"
if [[ "$(sha256_file "${extension_path}")" != "${expected_library_sha256}" ]]; then
    fail "Cached VexDB-Lite dynamic library checksum mismatch: ${extension_path}"
fi

if [[ -n "${GITHUB_ENV:-}" ]]; then
    echo "MEMME_VEXDB_LITE_EXTENSION=${extension_path}" >> "${GITHUB_ENV}"
else
    echo "${extension_path}"
fi
