import Foundation

// MARK: - DefaultHttpClient

/// URLSession-based HttpClient that satisfies the UniFFI callback interface.
/// FFI calls are synchronous (blocking), so we use DispatchSemaphore.
/// MUST be called from a background thread — never from the main thread.
class DefaultHttpClient: HttpClient {
    func post(url: String, headers: [String], body: String) throws -> String {
        guard let requestUrl = URL(string: url) else {
            throw MemmeError.Runtime(msg: "Invalid URL: \(url)")
        }
        var request = URLRequest(url: requestUrl)
        request.httpMethod = "POST"
        request.httpBody = body.data(using: .utf8)
        var i = 0
        while i + 1 < headers.count {
            request.setValue(headers[i + 1], forHTTPHeaderField: headers[i])
            i += 2
        }
        var responseBody = ""
        var responseError: Error?
        let semaphore = DispatchSemaphore(value: 0)
        URLSession.shared.dataTask(with: request) { data, _, error in
            if let error = error { responseError = error }
            else if let data = data, let text = String(data: data, encoding: .utf8) {
                responseBody = text
            }
            semaphore.signal()
        }.resume()
        semaphore.wait()
        if let error = responseError {
            throw MemmeError.Runtime(msg: error.localizedDescription)
        }
        return responseBody
    }
}

/// Result type for smart add operations (LLM-based extraction + optional graph).
struct SmartAddResult {
    let memories: [MemoryResult]
    let graph: GraphSearchResult?
}

// MARK: - i18n

enum AppLanguage: String {
    case en, zh

    var label: String { self == .en ? "EN" : "中" }
}

struct L10n {
    let lang: AppLanguage

    // Config
    var configuration: String { lang == .zh ? "配置" : "Configuration" }
    var embedding: String { lang == .zh ? "向量嵌入" : "Embedding" }
    var llmForCompact: String { lang == .zh ? "LLM（用于 Push + Compact）" : "LLM (for Push + Compact)" }
    var apiKey: String { "API Key" }
    var model: String { lang == .zh ? "模型" : "Model" }
    var dims: String { lang == .zh ? "维度" : "Dims" }
    var baseUrl: String { "Base URL" }
    var connectEmbedding: String { lang == .zh ? "连接 Embedding" : "Connect Embedding" }
    var reconnectEmbedding: String { lang == .zh ? "重新连接" : "Reconnect Embedding" }
    var setLLM: String { lang == .zh ? "设置 LLM" : "Set LLM" }
    var saved: String { lang == .zh ? "已保存" : "Saved" }
    var embeddingConnected: String { lang == .zh ? "Embedding 已连接" : "Embedding connected" }
    var llmSet: String { lang == .zh ? "LLM 已设置" : "LLM set" }

    // Chat
    var welcome: String { lang == .zh ? "欢迎使用 MemMe！使用 Push 直接存储，或 Push + Compact 进行 AI 智能提取。" : "Welcome to MemMe! Use Push for raw storage, or Push + Compact for AI-powered memory extraction." }
    var push: String { "Push" }
    var pushAndCompact: String { "Push + Compact" }
    var memoryAdded: String { lang == .zh ? "记忆已添加" : "Memory added" }
    var extracting: String { lang == .zh ? "正在用 AI 提取事实..." : "Extracting facts with AI..." }
    var extracted: String { lang == .zh ? "已提取" : "Extracted" }
    var memories: String { lang == .zh ? "条记忆" : "memories" }
    var entities: String { lang == .zh ? "个实体" : "entities" }
    var relations: String { lang == .zh ? "个关系" : "relations" }

    // Search
    var searchMemories: String { lang == .zh ? "搜索记忆..." : "Search memories..." }
    var search: String { lang == .zh ? "搜索" : "Search" }
    var noResults: String { lang == .zh ? "暂无结果" : "No results" }
    var searchToStart: String { lang == .zh ? "输入关键词开始搜索" : "Type a query to search" }
    var knowledgeGraph: String { lang == .zh ? "知识图谱" : "Knowledge Graph" }

    // Stats
    var stats: String { lang == .zh ? "统计" : "Stats" }
    var topEntities: String { lang == .zh ? "热门实体" : "Top Entities" }
    var noEntities: String { lang == .zh ? "暂无实体" : "No entities yet" }
    var links: String { lang == .zh ? "个关联" : "links" }

    // Test data
    var loadTestData: String { lang == .zh ? "加载测试数据" : "Load Test Data" }
    var testDataLoaded: String { lang == .zh ? "测试数据已加载" : "Test data loaded" }
    var loading: String { lang == .zh ? "加载中..." : "Loading..." }
    var testConversations: String { lang == .zh ? "测试对话" : "Test Conversations" }
    var tapToLoad: String { lang == .zh ? "点击对话将其写入记忆引擎" : "Tap a conversation to push it into the memory engine" }
    var pushed: String { lang == .zh ? "已写入" : "Pushed" }
    var configHint: String { lang == .zh ? "支持 OpenAI 兼容格式（Base URL 会自动补 /v1）\nOpenAI: https://api.openai.com\nDashScope: https://dashscope.aliyuncs.com/compatible-mode" : "Supports OpenAI-compatible APIs (Base URL auto-appends /v1)\nOpenAI: https://api.openai.com\nDashScope: https://dashscope.aliyuncs.com/compatible-mode" }

    // One-click test
    var oneClickTest: String { lang == .zh ? "一键测试" : "One-Click Test" }
    var configEmbeddingFirst: String { lang == .zh ? "请先配置 Embedding（填写 API Key 后点击「连接 Embedding」）" : "Please configure Embedding first (enter API Key and click Connect Embedding)" }
    var testStep1: String { lang == .zh ? "Step 1: 写入测试对话..." : "Step 1: Pushing test conversations..." }
    var testStep2: String { lang == .zh ? "Step 2: 搜索验证..." : "Step 2: Searching to verify..." }
    var testStep1Done: String { lang == .zh ? "写入完成" : "Push complete" }
    var testSearchQuery1: String { lang == .zh ? "Alex 在哪里工作" : "Where does Alex work" }
    var testSearchQuery2: String { lang == .zh ? "谁喜欢摄影" : "Who likes photography" }
    var testSearchQuery3: String { lang == .zh ? "钢琴" : "piano" }
    var testComplete: String { lang == .zh ? "测试完成！请查看右侧 Search 页签的搜索结果。" : "Test complete! Check the Search tab on the right for results." }
    var testSearching: String { lang == .zh ? "搜索" : "Searching" }
    var testFound: String { lang == .zh ? "找到" : "Found" }
    var testResults: String { lang == .zh ? "条结果" : "results" }
}

@MainActor
class MemmeAPI: ObservableObject {
    @Published var userId: String = "demo_user"
    @Published var isReady: Bool = false
    @Published var errorMessage: String?
    @Published var language: AppLanguage = .en

    var l10n: L10n { L10n(lang: language) }

    // Auto-search trigger (set by one-click test, consumed by SearchView)
    @Published var autoSearchQuery: String? = nil

    // Embedding config (tracks current embedder)
    @Published var embedModel: String? = nil  // nil = mock

    // LLM config (needed for smartAdd / graph)
    @Published var llmApiKey: String = ""
    @Published var llmModel: String = "gpt-4.1-nano"
    @Published var llmBaseUrl: String = "https://api.openai.com"

    private var store: MemoryStore?

    private let queue = DispatchQueue(label: "com.memme.ffi", qos: .userInitiated)

    init() {
        initMock()
    }

    // MARK: - Initialization

    /// Initialize with mock embedder (no API key needed, good for basic CRUD).
    func initMock(dbPath: String = ":memory:", dims: UInt32 = 384) {
        do {
            store = try MemoryStore.newMock(dbPath: dbPath, dims: dims)
            isReady = true
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
            isReady = false
        }
    }

    /// Initialize with an API key (uses DefaultHttpClient backed by URLSession).
    func initWithApiKey(apiKey: String, dbPath: String = ":memory:", baseUrl: String? = nil, model: String? = nil, dims: UInt32? = nil) {
        do {
            store = try MemoryStore.newWithHttpClient(
                dbPath: dbPath,
                httpClient: DefaultHttpClient(),
                apiKey: apiKey,
                embeddingModel: model,
                embeddingDims: dims,
                llmBaseUrl: baseUrl
            )
            embedModel = model ?? "text-embedding-3-small"
            isReady = true
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
            isReady = false
        }
    }

    // MARK: - Health Check (local: always ready if store initialized)

    func checkHealth() async -> Bool {
        return isReady
    }

    // MARK: - CRUD

    func addMemory(content: String) async throws -> MemoryResult {
        guard let store else { throw MemmeAPIError.notInitialized }
        let s = store
        let uid = userId
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    let result = try s.add(content: content, userId: uid, agentId: nil, runId: nil, metadata: nil)
                    // Rebuild FTS index so hybrid search can find newly added memories
                    try? s.rebuildFtsIndex()
                    cont.resume(returning: result)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    func getMemory(id: String) async throws -> MemoryResult? {
        guard let store else { throw MemmeAPIError.notInitialized }
        let s = store
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    let result = try s.get(id: id)
                    cont.resume(returning: result)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    func updateMemory(id: String, content: String) async throws -> MemoryResult {
        guard let store else { throw MemmeAPIError.notInitialized }
        let s = store
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    let result = try s.update(id: id, content: content)
                    cont.resume(returning: result)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    func deleteMemory(id: String) async throws {
        guard let store else { throw MemmeAPIError.notInitialized }
        let s = store
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    try s.delete(id: id)
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - List & Search

    func listMemories(limit: UInt32 = 50) async throws -> [MemoryResult] {
        guard let store else { throw MemmeAPIError.notInitialized }
        let s = store
        let uid = userId
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    let result = try s.list(userId: uid, agentId: nil, runId: nil, limit: limit)
                    cont.resume(returning: result)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    func search(query: String, limit: UInt32 = 20) async throws -> [MemoryResult] {
        guard let store else { throw MemmeAPIError.notInitialized }
        let s = store
        let uid = userId
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    let result = try s.search(query: query, userId: uid, agentId: nil, runId: nil, limit: limit, threshold: nil)
                    cont.resume(returning: result)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    func hybridSearch(query: String, limit: UInt32 = 20) async throws -> [MemoryResult] {
        guard let store else { throw MemmeAPIError.notInitialized }
        let s = store
        let uid = userId
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    let result = try s.hybridSearch(query: query, userId: uid, agentId: nil, runId: nil, limit: limit, vectorWeight: nil, ftsWeight: nil)
                    cont.resume(returning: result)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Smart Add (LLM-powered extraction)

    func smartAdd(text: String) async throws -> SmartAddResult {
        guard let store else { throw MemmeAPIError.notInitialized }
        let s = store
        let uid = userId
        let model = llmModel.isEmpty ? nil : llmModel
        let baseUrl = llmBaseUrl.isEmpty ? nil : llmBaseUrl
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    let memories = try s.addSmart(
                        text: text, userId: uid,
                        llmModel: model, llmBaseUrl: baseUrl,
                        agentId: nil, runId: nil, metadata: nil
                    )
                    // Rebuild FTS index so hybrid search can find newly added memories
                    try? s.rebuildFtsIndex()
                    // Search graph to return current state (addSmart already extracts graph internally)
                    var graph: GraphSearchResult? = nil
                    do {
                        graph = try s.searchGraph(query: text, userId: uid, depth: 2)
                    } catch {
                        // Graph search is optional; ignore errors
                    }
                    cont.resume(returning: SmartAddResult(memories: memories, graph: graph))
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    func smartAddMessages(_ messages: [ChatMessage]) async throws -> [MemoryResult] {
        guard let store else { throw MemmeAPIError.notInitialized }
        let s = store
        let uid = userId
        let model = llmModel.isEmpty ? nil : llmModel
        let baseUrl = llmBaseUrl.isEmpty ? nil : llmBaseUrl
        let msgs = messages
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    let result = try s.addSmartMessages(
                        messages: msgs, userId: uid,
                        llmModel: model, llmBaseUrl: baseUrl,
                        agentId: nil, runId: nil, metadata: nil
                    )
                    // Rebuild FTS index so hybrid search can find newly added memories
                    try? s.rebuildFtsIndex()
                    cont.resume(returning: result)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Unified Search (memories + graph combined)

    struct UnifiedSearchResult {
        let memories: [MemoryResult]
        let graph: GraphSearchResult?
    }

    func unifiedSearch(query: String, limit: UInt32 = 20) async throws -> UnifiedSearchResult {
        async let memories = hybridSearch(query: query, limit: limit)
        async let graph: GraphSearchResult? = {
            do { return try await searchGraph(query: query) }
            catch { return nil }
        }()
        return try await UnifiedSearchResult(memories: memories, graph: graph)
    }

    // MARK: - Graph

    func searchGraph(query: String, depth: UInt32 = 2) async throws -> GraphSearchResult {
        guard let store else { throw MemmeAPIError.notInitialized }
        let s = store
        let uid = userId
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    let result = try s.searchGraph(query: query, userId: uid, depth: depth)
                    cont.resume(returning: result)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Analytics

    func userStats() async throws -> UserStats {
        guard let store else { throw MemmeAPIError.notInitialized }
        let s = store
        let uid = userId
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    let result = try s.userStats(userId: uid)
                    cont.resume(returning: result)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    func topEntities(limit: UInt32 = 10) async throws -> [EntityStat] {
        guard let store else { throw MemmeAPIError.notInitialized }
        let s = store
        let uid = userId
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    let result = try s.topEntities(userId: uid, limit: limit)
                    cont.resume(returning: result)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    func memoryFrequency(granularity: String = "day", limit: UInt32 = 30) async throws -> [TimeBucket] {
        guard let store else { throw MemmeAPIError.notInitialized }
        let s = store
        let uid = userId
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    let result = try s.memoryFrequency(userId: uid, granularity: granularity, limit: limit)
                    cont.resume(returning: result)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - History

    func history(memoryId: String) async throws -> [HistoryRecord] {
        guard let store else { throw MemmeAPIError.notInitialized }
        let s = store
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    let result = try s.history(memoryId: memoryId)
                    cont.resume(returning: result)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Bulk Operations

    func deleteAll() async throws -> UInt64 {
        guard let store else { throw MemmeAPIError.notInitialized }
        let s = store
        let uid = userId
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    let count = try s.deleteAll(userId: uid, agentId: nil, runId: nil)
                    cont.resume(returning: count)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    func reset() async throws {
        guard let store else { throw MemmeAPIError.notInitialized }
        let s = store
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    try s.reset()
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }
}

enum MemmeAPIError: LocalizedError {
    case notInitialized
    case noLlmKey

    var errorDescription: String? {
        switch self {
        case .notInitialized: return "Memory store not initialized"
        case .noLlmKey: return "LLM API key not configured. Set it in Settings."
        }
    }
}
