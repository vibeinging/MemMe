#pragma once

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
namespace duckdb {

struct HNSWModule {
public:
	static void Register(ExtensionLoader &loader) {

		auto &db = loader.GetDatabaseInstance();

		RegisterIndex(db);
		RegisterIndexScan(loader);
		RegisterIndexPragmas(loader);
		RegisterMacros(loader);

		// Optimizers
		RegisterExprOptimizer(db);
		RegisterScanOptimizer(db);
		RegisterTopKOptimizer(db);
		RegisterJoinOptimizer(db);
	}

private:
	static void RegisterIndex(DatabaseInstance &ldb);
	static void RegisterIndexScan(ExtensionLoader &loader);
	static void RegisterIndexPragmas(ExtensionLoader &loader);
	static void RegisterMacros(ExtensionLoader &loader);
	static void RegisterTopKOperator(DatabaseInstance &db);

	static void RegisterExprOptimizer(DatabaseInstance &db);
	static void RegisterTopKOptimizer(DatabaseInstance &db);
	static void RegisterScanOptimizer(DatabaseInstance &db);
	static void RegisterJoinOptimizer(DatabaseInstance &db);
};

} // namespace duckdb
