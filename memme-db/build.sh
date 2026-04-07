#!/bin/bash
# MemMe Database Engine Builder
#
# Builds DuckDB with VSS (HNSW vector search) + FTS + JSON statically linked.
# DuckDB source is auto-downloaded on first run (pinned to v1.5.0).
#
# Usage:
#   ./memme-db/build.sh [release|debug|clean]
#
# Output:
#   memme-db/build/<mode>/libduckdb_static.a   (merged static library)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DUCKDB_DIR="$SCRIPT_DIR/duckdb"
DUCKDB_VERSION="v1.5.0"
DUCKDB_REPO="https://github.com/duckdb/duckdb.git"
NCPU=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 8)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CMD="${1:-release}"

# Auto-download DuckDB source if not present
if [ ! -d "$DUCKDB_DIR/src" ]; then
    echo -e "${YELLOW}[Setup] Downloading DuckDB ${DUCKDB_VERSION}...${NC}"
    git clone --depth 1 --branch "$DUCKDB_VERSION" "$DUCKDB_REPO" "$DUCKDB_DIR"
    echo -e "${GREEN}[Setup] DuckDB ${DUCKDB_VERSION} ready${NC}"
fi

# Link vss extension into DuckDB extension directory
if [ ! -L "$DUCKDB_DIR/extension/vss" ] && [ ! -d "$DUCKDB_DIR/extension/vss" ]; then
    ln -s "$SCRIPT_DIR/vss" "$DUCKDB_DIR/extension/vss"
    echo -e "${GREEN}[Setup] vss extension linked${NC}"
fi

case "$CMD" in
    release)
        BUILD_DIR="$SCRIPT_DIR/build/release"
        BUILD_TYPE="Release"
        ;;
    debug)
        BUILD_DIR="$SCRIPT_DIR/build/debug"
        BUILD_TYPE="Debug"
        ;;
    clean)
        echo -e "${YELLOW}[Clean] Removing build directories...${NC}"
        rm -rf "$SCRIPT_DIR/build"
        echo -e "${GREEN}[Done]${NC}"
        exit 0
        ;;
    *)
        echo "Usage: $0 [release|debug|clean]"
        exit 1
        ;;
esac

echo -e "${YELLOW}[Configure] $BUILD_TYPE → $BUILD_DIR${NC}"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo -e "${YELLOW}[Build] jobs=$NCPU${NC}"
cmake --build "$BUILD_DIR" \
    --target duckdb_static \
            duckdb_generated_extension_loader \
            vss_extension \
            fts_extension \
            json_extension \
            core_functions_extension \
    -j "$NCPU"

echo -e "${YELLOW}[Merge] Creating unified libduckdb_static.a ...${NC}"

MERGED_LIB="$BUILD_DIR/libduckdb_static.a"
rm -f "$MERGED_LIB"

libtool -static -o "$MERGED_LIB" \
    $(find "$BUILD_DIR/duckdb" -path "*/CMakeFiles/*" -name "*.o" \
        ! -path "*_loadable_*" \
        ! -path "*_subbuild*" \
        | sort) \
    2>&1 | grep -v "^libtool: warning" || true

if [ -f "$MERGED_LIB" ]; then
    SIZE=$(du -h "$MERGED_LIB" | cut -f1)
    echo -e "${GREEN}[Done] $MERGED_LIB ($SIZE)${NC}"
    echo ""
    echo "To use with MemMe:"
    echo "  export DUCKDB_LIB_DIR=$BUILD_DIR"
    echo "  export DUCKDB_INCLUDE_DIR=$DUCKDB_DIR/src/include"
    echo "  export DUCKDB_STATIC=1"
    echo "  cargo build -p memme-core --no-default-features --features memme-db"
else
    echo -e "${RED}[Error] Merged libduckdb_static.a not found${NC}"
    exit 1
fi
