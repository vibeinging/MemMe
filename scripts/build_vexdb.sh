#!/bin/bash
# Build DuckDB with MemMe-DB extension for Rust integration.
#
# This builds the DuckDB shared library (libduckdb.dylib / libduckdb.so) with
# the MemMe-DB vector extension statically linked, enabling HNSW in MemMe.
#
# Usage:
#   ./scripts/build_memme_db.sh [DUCKDB_MEMME_DB_DIR]
#
# Arguments:
#   DUCKDB_MEMME_DB_DIR  Path to the memme_db/duckdb directory (default: ../memme_db/duckdb)
#
# After running this script, build MemMe with:
#   export DUCKDB_LIB_DIR=/path/to/memme_db/duckdb/build/release/src
#   export DUCKDB_INCLUDE_DIR=/path/to/memme_db/duckdb/src/include
#   cargo build -p memme-core --no-default-features --features memme-db

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Determine DuckDB source directory
DUCKDB_DIR="${1:-${DUCKDB_MEMME_DB_DIR:-$WORKSPACE_DIR/../memme_db/duckdb}}"
DUCKDB_DIR="$(cd "$DUCKDB_DIR" 2>/dev/null && pwd || echo "$DUCKDB_DIR")"

if [ ! -f "$DUCKDB_DIR/CMakeLists.txt" ]; then
    echo "ERROR: DuckDB source not found at: $DUCKDB_DIR"
    echo "       Pass the path as argument or set DUCKDB_MEMME_DB_DIR env var."
    echo "       Expected: path to the memme_db/duckdb directory containing CMakeLists.txt"
    exit 1
fi

# Check that vex extension exists
if [ ! -d "$DUCKDB_DIR/extension/vex" ]; then
    echo "ERROR: MemMe-DB extension not found at: $DUCKDB_DIR/extension/vex"
    echo "       Make sure you're pointing to the memme_db fork, not upstream DuckDB."
    exit 1
fi

NCPU=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
BUILD_DIR="$DUCKDB_DIR/build/release"

echo "=== Building DuckDB + MemMe-DB ==="
echo "  Source:  $DUCKDB_DIR"
echo "  Build:   $BUILD_DIR"
echo "  Jobs:    $NCPU"
echo ""

# Configure — shared lib includes statically-linked extensions (including vex)
echo "[1/3] Configuring CMake (Release)..."
cmake -S "$DUCKDB_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHELL=OFF \
    -DBUILD_UNITTESTS=OFF \
    -DEXTENSION_STATIC_BUILD=OFF

# Build both shared and static targets
echo ""
echo "[2/3] Building DuckDB (shared + static)..."
cmake --build "$BUILD_DIR" --target duckdb duckdb_static -j"$NCPU"

# Verify outputs
echo ""
echo "[3/3] Verifying build outputs..."

REQUIRED_FILES=(
    "$BUILD_DIR/src/libduckdb_static.a"
    "$BUILD_DIR/extension/vex/libvex_extension.a"
)

# Also check for the shared library (platform dependent)
if [ "$(uname)" = "Darwin" ]; then
    REQUIRED_FILES+=("$BUILD_DIR/src/libduckdb.dylib")
else
    REQUIRED_FILES+=("$BUILD_DIR/src/libduckdb.so")
fi

ALL_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  OK: $f ($SIZE)"
    else
        echo "  MISSING: $f"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "ERROR: Some required files are missing. Build may have failed."
    exit 1
fi

LIB_DIR="$BUILD_DIR/src"
INCLUDE_DIR="$DUCKDB_DIR/src/include"

echo ""
echo "=== Build complete ==="
echo ""
echo "To build MemMe with MemMe-DB support, set env vars and build:"
echo ""
echo "  export DUCKDB_LIB_DIR=$LIB_DIR"
echo "  export DUCKDB_INCLUDE_DIR=$INCLUDE_DIR"
echo "  cargo build -p memme-core --no-default-features --features memme-db"
echo ""
echo "Or use the convenience variable:"
echo ""
echo "  export DUCKDB_MEMME_DB_DIR=$DUCKDB_DIR"
echo "  cargo build -p memme-core --no-default-features --features memme-db"

echo ""
echo "To run tests verifying MemMe-DB is loaded:"
echo ""
echo "  DUCKDB_LIB_DIR=$LIB_DIR DUCKDB_INCLUDE_DIR=$INCLUDE_DIR \\"
echo "    cargo test -p memme-core --no-default-features --features memme-db -- memme_db"
