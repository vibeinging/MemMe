#pragma once

#include "duckdb/common/helper.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/table/table_scan.hpp"

#include <unordered_set>

namespace duckdb {

class Index;

//! Describes a simple column equality filter: column_storage_id = constant_value
struct HNSWColumnEqualityFilter {
	storage_t column_storage_id;  //! Physical storage column ID
	idx_t column_index;           //! Logical column index in the table
	Value value;                  //! The constant value to match
};

// This is created by the optimizer rule
struct HNSWIndexScanBindData final : public TableScanBindData {
	explicit HNSWIndexScanBindData(TableCatalogEntry &table, Index &index, idx_t limit,
	                               unsafe_unique_array<float> query)
	    : TableScanBindData(table), index(index), limit(limit), query(std::move(query)) {
	}

	//! The index to use
	Index &index;

	//! The limit of the scan
	idx_t limit;

	//! The query vector
	unsafe_unique_array<float> query;

	//! Optional: set of valid row IDs for filtered search.
	//! When non-empty, only these row IDs are considered during ANN search.
	std::shared_ptr<std::unordered_set<row_t>> filter_row_ids;

	//! Optional: equality filters extracted from the query's WHERE clause.
	//! Used at execution time to pre-scan matching row IDs for filtered_search.
	vector<HNSWColumnEqualityFilter> equality_filters;

public:
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<HNSWIndexScanBindData>();
		return &other.table == &table;
	}
};

struct HNSWIndexScanFunction {
	static TableFunction GetFunction();
};

} // namespace duckdb
