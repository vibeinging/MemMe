#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/index.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/storage/table/data_table_info.hpp"
#include "duckdb/storage/table/table_index_list.hpp"

#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"
#include "hnsw/hnsw_index_scan.hpp"

namespace duckdb {

//-----------------------------------------------------------------------------
// Plan rewriter
//-----------------------------------------------------------------------------
class HNSWIndexScanOptimizer : public OptimizerExtension {
public:
	HNSWIndexScanOptimizer() {
		optimize_function = Optimize;
	}

	static bool TryOptimize(ClientContext &context, unique_ptr<LogicalOperator> &plan) {
		// Look for a TopN operator
		auto &op = *plan;

		if (op.type != LogicalOperatorType::LOGICAL_TOP_N) {
			return false;
		}

		auto &top_n = op.Cast<LogicalTopN>();

		if (top_n.orders.size() != 1) {
			// We can only optimize if there is a single order by expression right now
			return false;
		}

		const auto &order = top_n.orders[0];

		if (order.type != OrderType::ASCENDING) {
			// We can only optimize if the order by expression is ascending
			return false;
		}

		if (order.expression->type != ExpressionType::BOUND_COLUMN_REF) {
			// The expression has to reference the child operator (a projection with the distance function)
			return false;
		}
		const auto &bound_column_ref = order.expression->Cast<BoundColumnRefExpression>();

		// find the expression that is referenced
		if (top_n.children.size() != 1 || top_n.children.front()->type != LogicalOperatorType::LOGICAL_PROJECTION) {
			// The child has to be a projection
			return false;
		}

		auto &projection = top_n.children.front()->Cast<LogicalProjection>();

		// This the expression that is referenced by the order by expression
		const auto projection_index = bound_column_ref.binding.column_index;
		const auto &projection_expr = projection.expressions[projection_index];

		// The projection must sit on top of a get, or on top of a filter that sits on top of a get.
		// Pattern 1: TopN → Projection → Get
		// Pattern 2: TopN → Projection → Filter → Get  (filter will be kept for post-processing)
		if (projection.children.size() != 1) {
			return false;
		}

		LogicalFilter *filter_node = nullptr;
		unique_ptr<LogicalOperator> *get_ptr_ref = &projection.children.front();

		if ((*get_ptr_ref)->type == LogicalOperatorType::LOGICAL_FILTER) {
			// Pattern 2: Projection → Filter → Get
			filter_node = &(*get_ptr_ref)->Cast<LogicalFilter>();
			if (filter_node->children.size() != 1 ||
			    filter_node->children.front()->type != LogicalOperatorType::LOGICAL_GET) {
				return false;
			}
			get_ptr_ref = &filter_node->children.front();
		} else if ((*get_ptr_ref)->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}

		auto &get_ptr = *get_ptr_ref;
		auto &get = get_ptr->Cast<LogicalGet>();
		// Check if the get is a table scan
		if (get.function.name != "seq_scan") {
			return false;
		}

		if (get.dynamic_filters && get.dynamic_filters->HasFilters()) {
			// Cant push down!
			return false;
		}

		// We have a top-n operator on top of a table scan
		// We can replace the function with a custom index scan (if the table has a custom index)

		// Get the table
		auto &table = *get.GetTable();
		if (!table.IsDuckTable()) {
			// We can only replace the scan if the table is a duck table
			return false;
		}

		auto &duck_table = table.Cast<DuckTableEntry>();
		auto &table_info = *table.GetStorage().GetDataTableInfo();

		// Find the index
		unique_ptr<HNSWIndexScanBindData> bind_data = nullptr;
		vector<reference<Expression>> bindings;

		table_info.BindIndexes(context, HNSWIndex::TYPE_NAME);
		for(auto &index : table_info.GetIndexes().Indexes()) {
			if (!index.IsBound() || HNSWIndex::TYPE_NAME != index.GetIndexType()) {
				continue;
			}
			auto &cast_index = index.Cast<HNSWIndex>();

			// Reset the bindings
			bindings.clear();

			// Check that the projection expression is a distance function that matches the index
			if (!cast_index.TryMatchDistanceFunction(projection_expr, bindings)) {
				continue;
			}
			// Check that the HNSW index actually indexes the expression
			unique_ptr<Expression> index_expr;
			if (!cast_index.TryBindIndexExpression(get, index_expr)) {
				continue;
			}

			// Now, ensure that one of the bindings is a constant vector, and the other our index expression
			auto &const_expr_ref = bindings[1];
			auto &index_expr_ref = bindings[2];

			if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT || !index_expr->Equals(index_expr_ref)) {
				// Swap the bindings and try again
				std::swap(const_expr_ref, index_expr_ref);
				if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT ||
				    !index_expr->Equals(index_expr_ref)) {
					// Nope, not a match, we can't optimize.
					continue;
				}
			}

			const auto vector_size = cast_index.GetVectorSize();
			const auto &matched_vector = const_expr_ref.get().Cast<BoundConstantExpression>().value;
			auto query_vector = make_unsafe_uniq_array<float>(vector_size);
			auto vector_elements = ArrayValue::GetChildren(matched_vector);
			for (idx_t i = 0; i < vector_size; i++) {
				query_vector[i] = vector_elements[i].GetValue<float>();
			}

			bind_data = make_uniq<HNSWIndexScanBindData>(duck_table, cast_index, top_n.limit, std::move(query_vector));
			break;
		}

		if (!bind_data) {
			// No index found
			return false;
		}

		// --- Extract equality filters from the Filter node (if present) ---
		// These will be used at execution time to pre-scan matching row IDs
		// and pass them to usearch's filtered_search for in-graph filtering.
		bool all_filters_extracted = false;
		if (filter_node) {
			all_filters_extracted = true;
			for (auto &expr : filter_node->expressions) {
				// Look for patterns: col = constant  or  constant = col
				if (expr->type == ExpressionType::COMPARE_EQUAL) {
					auto &comp = expr->Cast<BoundComparisonExpression>();
					BoundColumnRefExpression *col_ref = nullptr;
					BoundConstantExpression *const_val = nullptr;

					if (comp.left->type == ExpressionType::BOUND_COLUMN_REF &&
					    comp.right->type == ExpressionType::VALUE_CONSTANT) {
						col_ref = &comp.left->Cast<BoundColumnRefExpression>();
						const_val = &comp.right->Cast<BoundConstantExpression>();
					} else if (comp.right->type == ExpressionType::BOUND_COLUMN_REF &&
					           comp.left->type == ExpressionType::VALUE_CONSTANT) {
						col_ref = &comp.right->Cast<BoundColumnRefExpression>();
						const_val = &comp.left->Cast<BoundConstantExpression>();
					}

					if (col_ref && const_val) {
						HNSWColumnEqualityFilter f;
						f.column_index = col_ref->binding.column_index;
						// Resolve to physical storage column ID
						auto &table_column_ids = get.GetColumnIds();
						if (f.column_index < table_column_ids.size()) {
							auto primary_idx = table_column_ids[f.column_index].GetPrimaryIndex();
							f.column_storage_id = duck_table.GetColumn(LogicalIndex(primary_idx)).StorageOid();
						} else {
							f.column_storage_id = f.column_index;
						}
						f.value = const_val->value;
						bind_data->equality_filters.push_back(std::move(f));
						continue;
					}
				}
				// Non-equality or complex filter — can't extract, keep for post-filter
				all_filters_extracted = false;
			}
		}

		// Replace the Get function with our index scan
		const auto cardinality = get.function.cardinality(context, bind_data.get());
		get.function = HNSWIndexScanFunction::GetFunction();
		get.has_estimated_cardinality = cardinality->has_estimated_cardinality;
		get.estimated_cardinality = cardinality->estimated_cardinality;
		get.bind_data = std::move(bind_data);

		if (filter_node && all_filters_extracted) {
			// All filter expressions were extracted as equality filters.
			// Remove the Filter node — filtering will be done in the index via filtered_search.
			// Rewire: projection → get (skip the filter node)
			projection.children.front() = std::move(filter_node->children.front());
		}
		// else: filter_node stays in place for post-processing of non-extractable filters

		if (get.table_filters.filters.empty() && !filter_node) {
			// No filters at all — simple case
			plan = std::move(top_n.children[0]);
			return true;
		}

		if (!get.table_filters.filters.empty()) {
			// Pull up table_filters as a LogicalFilter (existing behavior)
			get.projection_ids.clear();
			get.types.clear();

			auto new_filter = make_uniq<LogicalFilter>();
			auto &column_ids = get.GetColumnIds();
			for (const auto &entry : get.table_filters.filters) {
				idx_t column_id = entry.first;
				auto &type = get.returned_types[column_id];
				bool found = false;
				for (idx_t i = 0; i < column_ids.size(); i++) {
					if (column_ids[i].GetPrimaryIndex() == column_id) {
						column_id = i;
						found = true;
						break;
					}
				}
				if (!found) {
					throw InternalException("Could not find column id for filter");
				}
				auto column = make_uniq<BoundColumnRefExpression>(type, ColumnBinding(get.table_index, column_id));
				new_filter->expressions.push_back(entry.second->ToExpression(*column));
			}
			new_filter->children.push_back(std::move(get_ptr));
			new_filter->ResolveOperatorTypes();
			get_ptr = std::move(new_filter);
		}

		// Remove the TopN operator
		plan = std::move(top_n.children[0]);
		return true;
	}

	static bool OptimizeChildren(ClientContext &context, unique_ptr<LogicalOperator> &plan) {

		auto ok = TryOptimize(context, plan);
		// Recursively optimize the children
		for (auto &child : plan->children) {
			ok |= OptimizeChildren(context, child);
		}
		return ok;
	}

	static void MergeProjections(unique_ptr<LogicalOperator> &plan) {
		if (plan->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			if (plan->children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &child = plan->children[0];

				if (child->children[0]->type == LogicalOperatorType::LOGICAL_GET &&
				    child->children[0]->Cast<LogicalGet>().function.name == "hnsw_index_scan") {
					auto &parent_projection = plan->Cast<LogicalProjection>();
					auto &child_projection = child->Cast<LogicalProjection>();

					column_binding_set_t referenced_bindings;
					for (auto &expr : parent_projection.expressions) {
						ExpressionIterator::EnumerateExpression(expr, [&](Expression &expr_ref) {
							if (expr_ref.type == ExpressionType::BOUND_COLUMN_REF) {
								auto &bound_column_ref = expr_ref.Cast<BoundColumnRefExpression>();
								referenced_bindings.insert(bound_column_ref.binding);
							}
						});
					}

					auto child_bindings = child_projection.GetColumnBindings();
					for (idx_t i = 0; i < child_projection.expressions.size(); i++) {
						auto &expr = child_projection.expressions[i];
						auto &outgoing_binding = child_bindings[i];

						if (referenced_bindings.find(outgoing_binding) == referenced_bindings.end()) {
							// The binding is not referenced
							// We can remove this expression. But positionality matters so just replace with int.
							expr = make_uniq_base<Expression, BoundConstantExpression>(Value(LogicalType::TINYINT));
						}
					}
					return;
				}
			}
		}
		for (auto &child : plan->children) {
			MergeProjections(child);
		}
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		auto did_use_hnsw_scan = OptimizeChildren(input.context, plan);
		if (did_use_hnsw_scan) {
			MergeProjections(plan);
		}
	}
};

//-----------------------------------------------------------------------------
// Register
//-----------------------------------------------------------------------------
void HNSWModule::RegisterScanOptimizer(DatabaseInstance &db) {
	// Register the optimizer extension
	OptimizerExtension::Register(db.config, HNSWIndexScanOptimizer());
}

} // namespace duckdb
