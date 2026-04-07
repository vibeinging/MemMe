#include "hnsw/hnsw_index_scan.hpp"

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/dependency_list.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/index.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/storage/storage_index.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"

#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"

namespace duckdb {

BindInfo HNSWIndexScanBindInfo(const optional_ptr<FunctionData> bind_data_p) {
	auto &bind_data = bind_data_p->Cast<HNSWIndexScanBindData>();
	return BindInfo(bind_data.table);
}

//-------------------------------------------------------------------------
// Global State
//-------------------------------------------------------------------------
struct HNSWIndexScanGlobalState : public GlobalTableFunctionState {
	ColumnFetchState fetch_state;
	TableScanState local_storage_state;
	vector<StorageIndex> column_ids;

	// Index scan state
	unique_ptr<IndexScanState> index_state;
	Vector row_ids = Vector(LogicalType::ROW_TYPE);

	DataChunk all_columns;
	vector<idx_t> projection_ids;
};

static unique_ptr<GlobalTableFunctionState> HNSWIndexScanInitGlobal(ClientContext &context,
                                                                    TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<HNSWIndexScanBindData>();

	auto result = make_uniq<HNSWIndexScanGlobalState>();

	// Setup the scan state for the local storage
	auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
	result->column_ids.reserve(input.column_ids.size());

	// Figure out the storage column ids
	for (auto &id : input.column_ids) {
		storage_t col_id = id;
		if (id != DConstants::INVALID_INDEX) {
			col_id = bind_data.table.GetColumn(LogicalIndex(id)).StorageOid();
		}
		result->column_ids.emplace_back(col_id);
	}

	// Initialize the storage scan state
	result->local_storage_state.Initialize(result->column_ids, context, input.filters);
	local_storage.InitializeScan(bind_data.table.GetStorage(), result->local_storage_state.local_state, input.filters);

	// If equality filters were extracted by the optimizer, pre-scan the table
	// to collect matching row IDs for use with usearch's filtered_search.
	auto &hnsw_index = bind_data.index.Cast<HNSWIndex>();
	if (!bind_data.equality_filters.empty()) {
		auto valid_ids = std::make_shared<std::unordered_set<row_t>>();

		auto &data_table = bind_data.table.GetStorage();
		auto &transaction = DuckTransaction::Get(context, bind_data.table.catalog);

		// Build column IDs for scanning: filter columns + row_id
		vector<StorageIndex> scan_col_ids;
		for (auto &f : bind_data.equality_filters) {
			scan_col_ids.emplace_back(f.column_storage_id);
		}
		// Add row_id as the last column
		scan_col_ids.emplace_back(COLUMN_IDENTIFIER_ROW_ID);

		// Build types for the scan chunk
		vector<LogicalType> scan_types;
		for (auto &f : bind_data.equality_filters) {
			scan_types.push_back(f.value.type());
		}
		scan_types.push_back(LogicalType::ROW_TYPE);

		DataChunk scan_chunk;
		scan_chunk.Initialize(context, scan_types);

		TableScanState scan_state;
		scan_state.Initialize(scan_col_ids, context);
		data_table.InitializeScan(context, transaction, scan_state, scan_col_ids);

		// Scan all rows and check filter conditions
		auto num_filters = bind_data.equality_filters.size();
		while (true) {
			scan_chunk.Reset();
			data_table.Scan(transaction, scan_chunk, scan_state);
			if (scan_chunk.size() == 0) {
				break;
			}

			auto row_id_data = FlatVector::GetData<row_t>(scan_chunk.data[num_filters]);
			for (idx_t row_idx = 0; row_idx < scan_chunk.size(); row_idx++) {
				bool matches = true;
				for (idx_t filter_idx = 0; filter_idx < num_filters; filter_idx++) {
					auto val = scan_chunk.data[filter_idx].GetValue(row_idx);
					if (val != bind_data.equality_filters[filter_idx].value) {
						matches = false;
						break;
					}
				}
				if (matches) {
					valid_ids->insert(row_id_data[row_idx]);
				}
			}
		}

		if (!valid_ids->empty()) {
			result->index_state =
			    hnsw_index.InitializeFilteredScan(bind_data.query.get(), bind_data.limit, context, *valid_ids);
		} else {
			// No matching rows — return empty result
			result->index_state =
			    hnsw_index.InitializeScan(bind_data.query.get(), 0, context);
		}
	} else {
		result->index_state =
		    hnsw_index.InitializeScan(bind_data.query.get(), bind_data.limit, context);
	}

	if (!input.CanRemoveFilterColumns()) {
		return std::move(result);
	}

	// We need this to project out what we scan from the underlying table.
	result->projection_ids = input.projection_ids;

	auto &duck_table = bind_data.table.Cast<DuckTableEntry>();
	const auto &columns = duck_table.GetColumns();
	vector<LogicalType> scanned_types;
	for (const auto &col_idx : input.column_indexes) {
		if (col_idx.IsRowIdColumn()) {
			scanned_types.emplace_back(LogicalType::ROW_TYPE);
		} else {
			scanned_types.push_back(columns.GetColumn(col_idx.ToLogical()).Type());
		}
	}
	result->all_columns.Initialize(context, scanned_types);

	return std::move(result);
}

//-------------------------------------------------------------------------
// Execute
//-------------------------------------------------------------------------
static void HNSWIndexScanExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {

	auto &bind_data = data_p.bind_data->Cast<HNSWIndexScanBindData>();
	auto &state = data_p.global_state->Cast<HNSWIndexScanGlobalState>();
	auto &transaction = DuckTransaction::Get(context, bind_data.table.catalog);

	// Scan the index for row id's
	auto row_count = bind_data.index.Cast<HNSWIndex>().Scan(*state.index_state, state.row_ids);
	if (row_count == 0) {
		// Short-circuit if the index had no more rows
		output.SetCardinality(0);
		return;
	}

	// Fetch the data from the local storage given the row ids
	if (state.projection_ids.empty()) {
		bind_data.table.GetStorage().Fetch(transaction, output, state.column_ids, state.row_ids, row_count,
		                                   state.fetch_state);
		return;
	}

	// Otherwise, we need to first fetch into our scan chunk, and then project out the result
	state.all_columns.Reset();
	bind_data.table.GetStorage().Fetch(transaction, state.all_columns, state.column_ids, state.row_ids, row_count,
	                                   state.fetch_state);
	output.ReferenceColumns(state.all_columns, state.projection_ids);
}

//-------------------------------------------------------------------------
// Statistics
//-------------------------------------------------------------------------
static unique_ptr<BaseStatistics> HNSWIndexScanStatistics(ClientContext &context, const FunctionData *bind_data_p,
                                                          column_t column_id) {
	auto &bind_data = bind_data_p->Cast<HNSWIndexScanBindData>();
	auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
	if (local_storage.Find(bind_data.table.GetStorage())) {
		// we don't emit any statistics for tables that have outstanding transaction-local data
		return nullptr;
	}
	return bind_data.table.GetStatistics(context, column_id);
}

//-------------------------------------------------------------------------
// Dependency
//-------------------------------------------------------------------------
void HNSWIndexScanDependency(LogicalDependencyList &entries, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<HNSWIndexScanBindData>();
	entries.AddDependency(bind_data.table);

	// TODO: Add dependency to index here?
}

//-------------------------------------------------------------------------
// Cardinality
//-------------------------------------------------------------------------
unique_ptr<NodeStatistics> HNSWIndexScanCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<HNSWIndexScanBindData>();
	return make_uniq<NodeStatistics>(bind_data.limit, bind_data.limit);
}

//-------------------------------------------------------------------------
// ToString
//-------------------------------------------------------------------------
static InsertionOrderPreservingMap<string> HNSWIndexScanToString(TableFunctionToStringInput &input) {
	D_ASSERT(input.bind_data);
	InsertionOrderPreservingMap<string> result;
	auto &bind_data = input.bind_data->Cast<HNSWIndexScanBindData>();
	result["Table"] = bind_data.table.name;
	result["HNSW Index"] = bind_data.index.GetIndexName();
	return result;
}

//-------------------------------------------------------------------------
// Get Function
//-------------------------------------------------------------------------
TableFunction HNSWIndexScanFunction::GetFunction() {
	TableFunction func("hnsw_index_scan", {}, HNSWIndexScanExecute);
	func.init_local = nullptr;
	func.init_global = HNSWIndexScanInitGlobal;
	func.statistics = HNSWIndexScanStatistics;
	func.dependency = HNSWIndexScanDependency;
	func.cardinality = HNSWIndexScanCardinality;
	func.pushdown_complex_filter = nullptr;
	func.to_string = HNSWIndexScanToString;
	func.table_scan_progress = nullptr;
	func.projection_pushdown = true;
	func.filter_pushdown = false;
	func.get_bind_info = HNSWIndexScanBindInfo;

	return func;
}

//-------------------------------------------------------------------------
// Register
//-------------------------------------------------------------------------
void HNSWModule::RegisterIndexScan(ExtensionLoader &loader) {
	loader.RegisterFunction(HNSWIndexScanFunction::GetFunction());
}

} // namespace duckdb
