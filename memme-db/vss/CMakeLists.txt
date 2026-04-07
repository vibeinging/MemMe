cmake_minimum_required(VERSION 2.8.12...3.29)

project(VssExtension)

include_directories(include)

set(EXTENSION_SOURCES
    vss_extension.cpp
    hnsw/hnsw_index.cpp
    hnsw/hnsw_index_macros.cpp
    hnsw/hnsw_index_physical_create.cpp
    hnsw/hnsw_index_pragmas.cpp
    hnsw/hnsw_index_scan.cpp
    hnsw/hnsw_index_plan.cpp
    hnsw/hnsw_topk_operator.cpp
    hnsw/hnsw_optimize_expr.cpp
    hnsw/hnsw_optimize_join.cpp
    hnsw/hnsw_optimize_topk.cpp
    hnsw/hnsw_optimize_scan.cpp)

build_static_extension(vss ${EXTENSION_SOURCES})
build_loadable_extension(vss " " ${EXTENSION_SOURCES})

install(
  TARGETS vss_extension
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
