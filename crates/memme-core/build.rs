/// Build script for memme-core.
///
/// When the `memme-db` feature is enabled (and `bundled` is disabled), this script
/// locates the precompiled DuckDB+MemMe-DB+FTS+JSON static library and
/// configures the linker search paths.
///
/// The merged `libduckdb_static.a` is produced by `./memme-db/build.sh release`.
/// It contains DuckDB core + all extensions + the generated extension loader.
///
/// Environment variables (set these BEFORE running cargo):
///   DUCKDB_LIB_DIR      — directory containing libduckdb_static.a
///   DUCKDB_INCLUDE_DIR  — directory containing duckdb.h
///   DUCKDB_STATIC=1     — tell libduckdb-sys to link statically
///
/// Usage:
///   ./memme-db/build.sh release
///   DUCKDB_LIB_DIR=memme-db/build/release \
///   DUCKDB_INCLUDE_DIR=memme-db/duckdb/src/include \
///   DUCKDB_STATIC=1 \
///   cargo build -p memme-core --no-default-features --features memme-db
fn main() {
    #[cfg(all(feature = "memme-db", not(feature = "bundled")))]
    configure_memme_db();

    #[cfg(all(feature = "memme-db", feature = "bundled"))]
    {
        println!(
            "cargo:warning=Both `memme-db` and `bundled` features are enabled. \
             `bundled` takes priority — MemMe-DB will NOT be available. \
             Use --no-default-features --features memme-db to use MemMe-DB."
        );
    }
}

#[cfg(all(feature = "memme-db", not(feature = "bundled")))]
fn configure_memme_db() {
    use std::env;
    use std::path::PathBuf;

    println!("cargo:rerun-if-env-changed=DUCKDB_LIB_DIR");
    println!("cargo:rerun-if-env-changed=DUCKDB_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=DUCKDB_STATIC");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = PathBuf::from(&manifest_dir).join("../..");

    // Resolve the library directory
    let lib_dir = if let Ok(dir) = env::var("DUCKDB_LIB_DIR") {
        PathBuf::from(dir)
    } else {
        let auto_dir = workspace_root.join("memme-db/build/release");
        if auto_dir.join("libduckdb_static.a").exists() {
            let canonical = auto_dir.canonicalize().unwrap_or(auto_dir.clone());
            println!(
                "cargo:warning=Auto-detected DuckDB at memme-db/. \
                 For reliable builds, set DUCKDB_LIB_DIR={}",
                canonical.display()
            );
            canonical
        } else {
            let include_dir = workspace_root.join("memme-db/duckdb/src/include");
            panic!(
                "\n\nmemme-db feature is enabled but DuckDB library not found.\n\n\
                 Run: ./memme-db/build.sh release\n\
                 Then: export DUCKDB_LIB_DIR={}\n\
                 \x20     export DUCKDB_INCLUDE_DIR={}\n\
                 \x20     export DUCKDB_STATIC=1\n\n",
                auto_dir.display(),
                include_dir.display()
            );
        }
    };

    // On macOS, the linker makes a single pass through static archives and
    // won't resolve cross-references between objects within the same archive.
    // -force_load forces ALL objects from the archive to be loaded, resolving
    // the circular dependencies between the extension loader and extension
    // implementations (VssExtension, FtsExtension, JsonExtension, etc.).
    let static_lib = lib_dir.join("libduckdb_static.a");
    if static_lib.exists() {
        if cfg!(target_os = "macos") {
            println!(
                "cargo:rustc-link-arg=-Wl,-force_load,{}",
                static_lib.display()
            );
        } else {
            // On Linux, --whole-archive achieves the same effect
            println!(
                "cargo:rustc-link-arg=-Wl,--whole-archive,{},--no-whole-archive",
                static_lib.display()
            );
        }
    }

    // Link C++ standard library (required for DuckDB's C++ code)
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
