import SwiftUI
import Foundation

// MARK: - BenchmarkView

/// Comprehensive iOS benchmark for paper evaluation.
/// Measures: cold start, add/search latency at multiple scales,
/// memory usage, DB file size, and exports results as JSON.
struct BenchmarkView: View {
    @EnvironmentObject var api: MemmeAPI

    @StateObject private var runner = BenchmarkRunner()

    @State private var embedApiKey = ""
    @State private var embedModel = "text-embedding-v3"
    @State private var embedBaseUrl = "https://dashscope.aliyuncs.com/compatible-mode"
    @State private var embedDims: String = "1024"
    @State private var numTrials: Int = 3

    // Scale selection
    @State private var selectedScales: Set<Int> = [100, 500]
    private let availableScales = [100, 500, 1000, 5000, 10000, 100000]

    // Export
    @State private var showShareSheet = false
    @State private var exportFileURL: URL?

    private var isConfigured: Bool { !embedApiKey.isEmpty }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                configSection
                scaleSection
                controlSection
                if runner.isRunning { progressSection }
                if let r = runner.result {
                    deviceInfoSection(r)
                    resultsSection(r)
                    exportSection(r)
                }
                if let error = runner.errorMessage { errorSection(error) }
            }
            .padding()
        }
        #if os(iOS)
        .navigationBarTitle("Benchmark", displayMode: .inline)
        .sheet(isPresented: $showShareSheet) {
            if let url = exportFileURL {
                ActivityView(activityItems: [url])
            }
        }
        #else
        .navigationTitle("Benchmark")
        #endif
    }

    // MARK: - Config

    private var configSection: some View {
        GroupBox(label: Label("Embedding", systemImage: "cpu")) {
            VStack(alignment: .leading, spacing: 6) {
                fieldRow("API Key") { SecureField("sk-...", text: $embedApiKey).font(.caption.monospaced()).textFieldStyle(.roundedBorder) }
                HStack(spacing: 8) {
                    fieldRow("Model") { TextField("model", text: $embedModel).font(.caption.monospaced()).textFieldStyle(.roundedBorder) }
                    fieldRow("Dims") { TextField("1024", text: $embedDims).font(.caption.monospaced()).textFieldStyle(.roundedBorder).frame(width: 60) }
                }
                fieldRow("Base URL") {
                    TextField("https://...", text: $embedBaseUrl).font(.caption.monospaced()).textFieldStyle(.roundedBorder)
                    #if os(iOS)
                    .autocapitalization(.none)
                    #endif
                }
                fieldRow("Trials") {
                    Picker("", selection: $numTrials) {
                        Text("1").tag(1)
                        Text("3").tag(3)
                        Text("5").tag(5)
                    }
                    .pickerStyle(.segmented)
                }
            }
            .padding(.vertical, 4)
        }
    }

    private func fieldRow<V: View>(_ label: String, @ViewBuilder content: () -> V) -> some View {
        HStack {
            Text(label).font(.caption).foregroundColor(.secondary).frame(width: 60, alignment: .leading)
            content()
        }
    }

    // MARK: - Scale Selection

    private var scaleSection: some View {
        GroupBox(label: Label("Memory Scales", systemImage: "chart.line.uptrend.xyaxis")) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Select scales to benchmark (memories count):")
                    .font(.caption).foregroundColor(.secondary)
                HStack(spacing: 8) {
                    ForEach(availableScales, id: \.self) { scale in
                        Toggle(isOn: Binding(
                            get: { selectedScales.contains(scale) },
                            set: { if $0 { selectedScales.insert(scale) } else { selectedScales.remove(scale) } }
                        )) {
                            Text(formatScale(scale))
                                .font(.caption.monospaced())
                        }
                        .toggleStyle(.button)
                        .buttonStyle(.bordered)
                        .tint(selectedScales.contains(scale) ? .blue : .gray)
                    }
                }
                Text("100 queries per scale · limit=10 · file-backed DB · \(numTrials) trial\(numTrials > 1 ? "s" : "")")
                    .font(.caption2).foregroundColor(.secondary)
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Control

    private var controlSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button(action: startBenchmark) {
                Label("Run Benchmark", systemImage: "play.fill").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!isConfigured || runner.isRunning || selectedScales.isEmpty)

            if !isConfigured {
                Text("Enter an Embedding API Key to enable.").font(.caption).foregroundColor(.orange)
            }
        }
    }

    // MARK: - Progress

    private var progressSection: some View {
        GroupBox(label: Label("Progress", systemImage: "hourglass")) {
            VStack(alignment: .leading, spacing: 8) {
                Text(runner.statusText).font(.callout.monospaced())
                ProgressView(value: runner.progress, total: 1.0).progressViewStyle(.linear)
                Text("\(Int(runner.progress * 100))%").font(.caption2).foregroundColor(.secondary)
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Device Info

    private func deviceInfoSection(_ r: FullBenchmarkResult) -> some View {
        GroupBox(label: Label("Device", systemImage: "iphone")) {
            VStack(alignment: .leading, spacing: 4) {
                metricRow("Model", value: r.device.model)
                metricRow("Chip", value: r.device.chip)
                metricRow("OS", value: r.device.osVersion)
                metricRow("RAM", value: r.device.totalRAM)
                metricRow("Timestamp", value: r.timestamp)
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Results

    private func resultsSection(_ r: FullBenchmarkResult) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(r.scaleResults.sorted(by: { $0.scale < $1.scale }), id: \.scale) { sr in
                scaleResultSection(sr)
            }
        }
    }

    private func scaleResultSection(_ sr: ScaleResult) -> some View {
        GroupBox(label: Label("\(formatScale(sr.scale)) memories", systemImage: "chart.bar")) {
            VStack(alignment: .leading, spacing: 10) {
                sectionHeader("Cold Start")
                metricRow("Init", value: fmt(sr.coldStartMs) + ciLabel(sr.coldStartCi95))

                Divider()

                sectionHeader("Add (\(sr.scale) memories)")
                metricRow("Total", value: fmt(sr.addTotalMs) + ciLabel(sr.addTotalCi95))
                metricRow("Rate", value: String(format: "%.2f ms/mem", sr.addTotalMs / Double(sr.scale)))
                metricRow("Throughput", value: String(format: "%.0f mem/s", Double(sr.scale) / (sr.addTotalMs / 1000.0)))
                statsRows(sr.addStats)

                Divider()

                sectionHeader("Search (100 queries, limit=10)")
                statsRows(sr.searchStats)

                Divider()

                sectionHeader("FTS Rebuild")
                metricRow("Time", value: fmt(sr.ftsRebuildMs) + ciLabel(sr.ftsRebuildCi95))

                Divider()

                sectionHeader("Resources")
                metricRow("Memory Δ", value: String(format: "%.1f MB", sr.memoryDeltaMB))
                metricRow("Peak Mem", value: String(format: "%.1f MB", sr.peakMemoryMB))
                metricRow("DB Size", value: String(format: "%.2f MB", sr.dbSizeMB))

                if sr.thermalStateChanged {
                    metricRow("Thermal", value: "\(sr.thermalStateStart) → \(sr.thermalStateEnd) ⚠️")
                } else {
                    metricRow("Thermal", value: sr.thermalStateStart)
                }
            }
            .padding(.vertical, 4)
        }
    }

    private func statsRows(_ s: LatencyStats) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            metricRow("Min", value: fmt(s.min))
            metricRow("Max", value: fmt(s.max))
            metricRow("Mean", value: fmt(s.mean) + ciLabel(s.ci95))
            metricRow("Median", value: fmt(s.median))
            metricRow("P95", value: fmt(s.p95))
            metricRow("P99", value: fmt(s.p99))
            metricRow("Std Dev", value: fmt(s.stdDev))
            metricRow("Total", value: fmt(s.total))
            metricRow("Count", value: "\(s.count)")
        }
    }

    // MARK: - Export

    private func exportSection(_ r: FullBenchmarkResult) -> some View {
        GroupBox(label: Label("Export", systemImage: "square.and.arrow.up")) {
            VStack(alignment: .leading, spacing: 8) {
                Button(action: { exportJSON(r) }) {
                    Label("Export JSON Report", systemImage: "doc.text").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)

                Button(action: { copyMarkdown(r) }) {
                    Label("Copy Markdown Table", systemImage: "doc.on.clipboard").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Error

    private func errorSection(_ message: String) -> some View {
        GroupBox(label: Label("Error", systemImage: "exclamationmark.triangle")) {
            Text(message).font(.caption).foregroundColor(.red)
        }
    }

    // MARK: - Helpers

    private func sectionHeader(_ title: String) -> some View {
        Text(title).font(.subheadline.bold())
    }

    private func metricRow(_ label: String, value: String) -> some View {
        HStack {
            Text(label).font(.caption).foregroundColor(.secondary).frame(width: 90, alignment: .leading)
            Spacer()
            Text(value).font(.caption.monospaced())
        }
    }

    private func fmt(_ ms: Double) -> String { String(format: "%.2f ms", ms) }

    private func ciLabel(_ ci: Double?) -> String {
        guard let ci = ci, ci > 0 else { return "" }
        return String(format: " ±%.2f", ci)
    }

    private func formatScale(_ n: Int) -> String {
        n >= 1000 ? "\(n/1000)K" : "\(n)"
    }

    // MARK: - Actions

    private func startBenchmark() {
        var url: String? = embedBaseUrl.trimmingCharacters(in: .whitespacesAndNewlines)
        if let u = url, !u.isEmpty {
            if !u.hasSuffix("/v1") && !u.hasSuffix("/v1/") { url = u + "/v1" }
        } else { url = nil }

        runner.run(
            apiKey: embedApiKey,
            baseUrl: url,
            model: embedModel.isEmpty ? nil : embedModel,
            dims: UInt32(embedDims),
            scales: selectedScales.sorted(),
            numTrials: numTrials
        )
    }

    private func exportJSON(_ r: FullBenchmarkResult) {
        let json = r.toJSON()
        let fileName = "memme_bench_\(r.device.model.replacingOccurrences(of: ",", with: "_"))_\(ISO8601DateFormatter().string(from: Date())).json"
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
        do {
            try json.write(to: tempURL, atomically: true, encoding: .utf8)
            exportFileURL = tempURL
            #if os(iOS)
            showShareSheet = true
            #else
            NSWorkspace.shared.activateFileViewerSelecting([tempURL])
            #endif
        } catch {
            runner.errorMessage = "Export failed: \(error.localizedDescription)"
        }
    }

    private func copyMarkdown(_ r: FullBenchmarkResult) {
        let md = r.toMarkdown()
        #if os(iOS)
        UIPasteboard.general.string = md
        #else
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(md, forType: .string)
        #endif
    }
}

// MARK: - ActivityView (iOS share sheet)

#if os(iOS)
struct ActivityView: UIViewControllerRepresentable {
    let activityItems: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }
    func updateUIViewController(_ uvc: UIActivityViewController, context: Context) {}
}
#endif

// MARK: - Data Types

struct LatencyStats {
    let min, max, mean, median, p95, p99, stdDev, total: Double
    let count: Int
    let ci95: Double  // 95% confidence interval half-width for mean

    init(latencies: [Double]) {
        let sorted = latencies.sorted()
        let n = sorted.count
        self.count = n
        guard n > 0 else {
            min = 0; max = 0; mean = 0; median = 0
            p95 = 0; p99 = 0; stdDev = 0; total = 0; ci95 = 0
            return
        }
        self.min = sorted.first!
        self.max = sorted.last!
        self.total = sorted.reduce(0, +)
        self.mean = total / Double(n)
        self.median = n % 2 == 0 ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0 : sorted[n/2]
        self.p95 = sorted[Swift.min(Int(Double(n) * 0.95), n - 1)]
        self.p99 = sorted[Swift.min(Int(Double(n) * 0.99), n - 1)]
        if n > 1 {
            let variance = sorted.map { ($0 - self.mean) * ($0 - self.mean) }.reduce(0, +) / Double(n - 1)
            self.stdDev = sqrt(variance)
            self.ci95 = 1.96 * self.stdDev / sqrt(Double(n))
        } else {
            self.stdDev = 0
            self.ci95 = 0
        }
    }

    func toDict() -> [String: Any] {
        ["count": count, "min": min, "max": max, "mean": mean, "median": median,
         "p95": p95, "p99": p99, "stdDev": stdDev, "total": total, "ci95": ci95]
    }
}

struct DeviceInfo {
    let model: String
    let chip: String
    let osVersion: String
    let totalRAM: String

    static func current() -> DeviceInfo {
        #if os(iOS)
        let device = UIDevice.current
        let osVer = "\(device.systemName) \(device.systemVersion)"
        #else
        let osVer = "macOS \(ProcessInfo.processInfo.operatingSystemVersionString)"
        #endif

        var machineId = "unknown"
        var size: Int = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        if size > 0 {
            var machine = [CChar](repeating: 0, count: size)
            sysctlbyname("hw.machine", &machine, &size, nil, 0)
            machineId = String(cString: machine)
        }

        var chip = "unknown"
        size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        if size > 0 {
            var brand = [CChar](repeating: 0, count: size)
            sysctlbyname("machdep.cpu.brand_string", &brand, &size, nil, 0)
            chip = String(cString: brand)
        } else {
            // iOS doesn't expose brand_string; use hw.cpufamily
            var family: UInt32 = 0
            size = MemoryLayout<UInt32>.size
            sysctlbyname("hw.cpufamily", &family, &size, nil, 0)
            chip = "Apple Silicon (cpufamily: \(String(format: "0x%08X", family)))"
        }

        let totalBytes = ProcessInfo.processInfo.physicalMemory
        let totalGB = String(format: "%.1f GB", Double(totalBytes) / 1_073_741_824.0)

        return DeviceInfo(model: machineId, chip: chip, osVersion: osVer, totalRAM: totalGB)
    }
}

struct ScaleResult {
    let scale: Int
    let coldStartMs: Double
    let coldStartCi95: Double?
    let addTotalMs: Double
    let addTotalCi95: Double?
    let addStats: LatencyStats
    let searchStats: LatencyStats
    let memoryDeltaMB: Double   // phys_footprint increase during this scale
    let peakMemoryMB: Double    // true peak phys_footprint sampled during operations
    let dbSizeMB: Double        // actual file size on disk
    let ftsRebuildMs: Double
    let ftsRebuildCi95: Double?
    let thermalStateStart: String
    let thermalStateEnd: String
    let thermalStateChanged: Bool
    let addLatenciesRaw: [Double]
    let searchLatenciesRaw: [Double]

    func toDict() -> [String: Any] {
        var d: [String: Any] = [
            "scale": scale, "coldStartMs": coldStartMs, "addTotalMs": addTotalMs,
            "addStats": addStats.toDict(), "searchStats": searchStats.toDict(),
            "memoryDeltaMB": memoryDeltaMB, "peakMemoryMB": peakMemoryMB, "dbSizeMB": dbSizeMB,
            "ftsRebuildMs": ftsRebuildMs,
            "thermalState": ["start": thermalStateStart, "end": thermalStateEnd, "changed": thermalStateChanged],
            "addLatenciesRaw": addLatenciesRaw,
            "searchLatenciesRaw": searchLatenciesRaw
        ]
        if let ci = coldStartCi95 { d["coldStartCi95"] = ci }
        if let ci = addTotalCi95 { d["addTotalCi95"] = ci }
        if let ci = ftsRebuildCi95 { d["ftsRebuildCi95"] = ci }
        return d
    }
}

struct FullBenchmarkResult {
    let device: DeviceInfo
    let timestamp: String
    let embeddingModel: String
    let embeddingDims: Int
    let numTrials: Int
    let scaleResults: [ScaleResult]

    func toJSON() -> String {
        let methodology: [String: Any] = [
            "dbMode": "file-backed",
            "trials": numTrials,
            "warmup": true,
            "queryCount": 100,
            "limit": 10
        ]
        let dict: [String: Any] = [
            "device": ["model": device.model, "chip": device.chip, "os": device.osVersion, "ram": device.totalRAM],
            "timestamp": timestamp,
            "embedding": ["model": embeddingModel, "dims": embeddingDims],
            "methodology": methodology,
            "scales": scaleResults.map { $0.toDict() }
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: dict, options: [.prettyPrinted, .sortedKeys]),
              let str = String(data: data, encoding: .utf8) else { return "{}" }
        return str
    }

    func toMarkdown() -> String {
        var md = "# MemMe iOS Benchmark\n\n"
        md += "> **Methodology**: file-backed DB, \(numTrials) trial\(numTrials > 1 ? "s" : ""), warmup phase, 100 queries per scale, limit=10\n\n"
        md += "- **Device**: \(device.model) (\(device.chip))\n"
        md += "- **OS**: \(device.osVersion)\n"
        md += "- **RAM**: \(device.totalRAM)\n"
        md += "- **Embedding**: \(embeddingModel) (\(embeddingDims)d)\n"
        md += "- **Timestamp**: \(timestamp)\n\n"

        let sorted = scaleResults.sorted(by: { $0.scale < $1.scale })

        // Summary table
        md += "| Scale | Cold Start | Add Rate | Search Mean ±CI | Search P95 | Search P99 | Memory Δ | DB Size | Thermal |\n"
        md += "|------:|----------:|---------:|----------------:|----------:|----------:|---------:|--------:|:--------|\n"
        for sr in sorted {
            let addRate = String(format: "%.1f ms/mem", sr.addTotalMs / Double(sr.scale))
            let searchCI = String(format: "%.1f ±%.1f ms", sr.searchStats.mean, sr.searchStats.ci95)
            let thermalFlag = sr.thermalStateChanged ? " ⚠️" : ""
            md += "| \(sr.scale) | \(String(format: "%.0f ms", sr.coldStartMs)) | \(addRate) "
            md += "| \(searchCI) "
            md += "| \(String(format: "%.1f ms", sr.searchStats.p95)) "
            md += "| \(String(format: "%.1f ms", sr.searchStats.p99)) "
            md += "| \(String(format: "%.1f MB", sr.memoryDeltaMB)) "
            md += "| \(String(format: "%.1f MB", sr.dbSizeMB)) "
            md += "| \(sr.thermalStateStart)\(thermalFlag) |\n"
        }

        // Detailed per-scale
        for sr in sorted {
            md += "\n## \(sr.scale) Memories\n\n"
            md += "### Add Latency\n"
            md += statsMarkdown(sr.addStats)
            md += "\n### Search Latency (100 queries, limit=10)\n"
            md += statsMarkdown(sr.searchStats)
            md += "\n### FTS Rebuild\n"
            md += "- Time: \(String(format: "%.2f ms", sr.ftsRebuildMs))\n"
            md += "\n### Resources\n"
            md += "- Memory Δ: \(String(format: "%.1f MB", sr.memoryDeltaMB))\n"
            md += "- Peak Memory: \(String(format: "%.1f MB", sr.peakMemoryMB))\n"
            md += "- DB Size: \(String(format: "%.2f MB", sr.dbSizeMB))\n"
            md += "- Thermal: \(sr.thermalStateStart)"
            if sr.thermalStateChanged { md += " → \(sr.thermalStateEnd) (changed!)" }
            md += "\n"
        }
        return md
    }

    private func statsMarkdown(_ s: LatencyStats) -> String {
        """
        | Metric | Value |
        |--------|------:|
        | Min | \(String(format: "%.2f ms", s.min)) |
        | Max | \(String(format: "%.2f ms", s.max)) |
        | Mean | \(String(format: "%.2f ms ±%.2f", s.mean, s.ci95)) |
        | Median | \(String(format: "%.2f ms", s.median)) |
        | P95 | \(String(format: "%.2f ms", s.p95)) |
        | P99 | \(String(format: "%.2f ms", s.p99)) |
        | Std Dev | \(String(format: "%.2f ms", s.stdDev)) |
        | Total | \(String(format: "%.2f ms", s.total)) |

        """
    }
}

// MARK: - Memory Measurement

func getPhysFootprintMB() -> Double {
    var vmInfo = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &vmInfo) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return 0 }
    return Double(vmInfo.phys_footprint) / 1_048_576.0
}

// MARK: - Thermal State

func thermalStateString() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

// MARK: - BenchmarkRunner

class BenchmarkRunner: ObservableObject {
    @Published var isRunning = false
    @Published var progress: Double = 0
    @Published var statusText: String = ""
    @Published var result: FullBenchmarkResult?
    @Published var errorMessage: String?

    private let queue = DispatchQueue(label: "com.memme.benchmark", qos: .userInitiated)

    func run(apiKey: String, baseUrl: String?, model: String?, dims: UInt32?, scales: [Int], numTrials: Int) {
        guard !isRunning else { return }
        DispatchQueue.main.async {
            self.isRunning = true
            self.progress = 0
            self.statusText = "Starting..."
            self.result = nil
            self.errorMessage = nil
        }

        queue.async { [weak self] in
            guard let self else { return }
            do {
                let result = try self.execute(apiKey: apiKey, baseUrl: baseUrl, model: model, dims: dims, scales: scales, numTrials: numTrials)
                DispatchQueue.main.async {
                    self.result = result
                    self.progress = 1.0
                    self.statusText = "Done"
                    self.isRunning = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = error.localizedDescription
                    self.isRunning = false
                }
            }
        }
    }

    private func execute(apiKey: String, baseUrl: String?, model: String?, dims: UInt32?, scales: [Int], numTrials: Int) throws -> FullBenchmarkResult {
        let device = DeviceInfo.current()
        let memories = BenchmarkData.memories  // 500 entries
        let queries = BenchmarkData.queries    // 100 entries
        let totalScales = scales.count
        var scaleResults: [ScaleResult] = []
        let httpClient = DefaultHttpClient()   // reuse across scales
        let embDims = Int(dims ?? 1024)

        // ---- Warmup phase ----
        update("Warmup: initializing...", progress: 0)
        do {
            let warmupDbPath = FileManager.default.temporaryDirectory
                .appendingPathComponent("memme_warmup_\(UUID().uuidString).db").path
            defer { cleanupDbFiles(warmupDbPath) }
            let warmupStore = try MemoryStore.newWithHttpClient(
                dbPath: warmupDbPath,
                httpClient: httpClient,
                apiKey: apiKey,
                embeddingModel: model,
                embeddingDims: dims,
                llmBaseUrl: baseUrl
            )
            for i in 0..<5 {
                _ = try warmupStore.add(content: memories[i], userId: "warmup", agentId: nil, runId: nil, metadata: nil)
            }
            for i in 0..<3 {
                _ = try warmupStore.search(query: queries[i], userId: "warmup", agentId: nil, runId: nil, limit: 10, threshold: nil)
            }
            update("Warmup: done", progress: 0.02)
        }

        let warmupReserve = 0.02  // 2% of progress reserved for warmup

        for (scaleIdx, scale) in scales.enumerated() {
            let scaleLabel = scale >= 1000 ? "\(scale/1000)K" : "\(scale)"
            let baseProgress = warmupReserve + (1.0 - warmupReserve) * Double(scaleIdx) / Double(totalScales)
            let scaleWeight = (1.0 - warmupReserve) / Double(totalScales)

            var trialColdStarts: [Double] = []
            var trialAddTotals: [Double] = []
            var trialFtsRebuilds: [Double] = []
            var allAddLatencies: [Double] = []
            allAddLatencies.reserveCapacity(scale * numTrials)
            var allSearchLatencies: [Double] = []
            allSearchLatencies.reserveCapacity(queries.count * numTrials)
            var lastMemoryDelta: Double = 0
            var overallPeakMem: Double = 0
            var lastDbSizeMB: Double = 0
            var lastThermalStart: String = "nominal"
            var lastThermalEnd: String = "nominal"
            var lastThermalChanged: Bool = false

            for trial in 0..<numTrials {
                let trialLabel = numTrials > 1 ? " T\(trial+1)/\(numTrials)" : ""
                let trialBase = baseProgress + scaleWeight * Double(trial) / Double(numTrials)
                let trialWeight = scaleWeight / Double(numTrials)

                let thermalStart = thermalStateString()
                let memBefore = getPhysFootprintMB()
                var peakMem = memBefore

                let dbPath = FileManager.default.temporaryDirectory
                    .appendingPathComponent("memme_bench_\(UUID().uuidString).db").path
                defer { cleanupDbFiles(dbPath) }

                update("[\(scaleIdx+1)/\(totalScales)] \(scaleLabel)\(trialLabel): Creating store...", progress: trialBase)
                let t0 = CFAbsoluteTimeGetCurrent()
                let store = try MemoryStore.newWithHttpClient(
                    dbPath: dbPath,
                    httpClient: httpClient,
                    apiKey: apiKey,
                    embeddingModel: model,
                    embeddingDims: dims,
                    llmBaseUrl: baseUrl
                )
                let coldStartMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
                trialColdStarts.append(coldStartMs)

                // ---- Add memories ----
                var addLatencies: [Double] = []
                addLatencies.reserveCapacity(scale)

                let addStart = CFAbsoluteTimeGetCurrent()
                for i in 0..<scale {
                    let content: String
                    if i < memories.count {
                        content = memories[i]
                    } else {
                        // Generate unique content beyond 500 to avoid dedup
                        content = "\(memories[i % memories.count]) [instance \(i)]"
                    }

                    let ta = CFAbsoluteTimeGetCurrent()
                    _ = try store.add(content: content, userId: "bench_user", agentId: nil, runId: nil, metadata: nil)
                    addLatencies.append((CFAbsoluteTimeGetCurrent() - ta) * 1000.0)

                    // Sample peak memory every 50 adds
                    if (i + 1) % 50 == 0 {
                        let currentMem = getPhysFootprintMB()
                        if currentMem > peakMem { peakMem = currentMem }
                    }

                    if (i + 1) % 100 == 0 || i + 1 == scale {
                        let p = trialBase + trialWeight * (Double(i + 1) / Double(scale + queries.count + 1)) * 0.8
                        let elapsed = (CFAbsoluteTimeGetCurrent() - addStart) * 1000.0
                        update("[\(scaleIdx+1)/\(totalScales)] \(scaleLabel)\(trialLabel): Adding \(i+1)/\(scale) (\(String(format: "%.0f", elapsed)) ms)", progress: p)
                    }
                }
                let addTotalMs = (CFAbsoluteTimeGetCurrent() - addStart) * 1000.0
                trialAddTotals.append(addTotalMs)

                // Rebuild FTS and time it
                update("[\(scaleIdx+1)/\(totalScales)] \(scaleLabel)\(trialLabel): Rebuilding FTS...", progress: trialBase + trialWeight * 0.82)
                let ftsStart = CFAbsoluteTimeGetCurrent()
                try store.rebuildFtsIndex()
                let ftsRebuildMs = (CFAbsoluteTimeGetCurrent() - ftsStart) * 1000.0
                trialFtsRebuilds.append(ftsRebuildMs)

                // ---- Search ----
                var searchLatencies: [Double] = []
                searchLatencies.reserveCapacity(queries.count)

                for (qi, query) in queries.enumerated() {
                    let ts = CFAbsoluteTimeGetCurrent()
                    _ = try store.search(query: query, userId: "bench_user", agentId: nil, runId: nil, limit: 10, threshold: nil)
                    searchLatencies.append((CFAbsoluteTimeGetCurrent() - ts) * 1000.0)

                    // Sample peak memory during search
                    if (qi + 1) % 20 == 0 {
                        let currentMem = getPhysFootprintMB()
                        if currentMem > peakMem { peakMem = currentMem }

                        let p = trialBase + trialWeight * (0.82 + 0.18 * Double(qi + 1) / Double(queries.count))
                        update("[\(scaleIdx+1)/\(totalScales)] \(scaleLabel)\(trialLabel): Searching \(qi+1)/\(queries.count)", progress: p)
                    }
                }

                let memAfter = getPhysFootprintMB()
                if memAfter > peakMem { peakMem = memAfter }

                // Measure actual DB file size
                var dbSizeMB: Double = 0
                if let attrs = try? FileManager.default.attributesOfItem(atPath: dbPath),
                   let fileSize = attrs[.size] as? UInt64 {
                    dbSizeMB = Double(fileSize) / 1_048_576.0
                }

                let thermalEnd = thermalStateString()
                lastMemoryDelta = memAfter - memBefore
                if peakMem > overallPeakMem { overallPeakMem = peakMem }
                lastDbSizeMB = dbSizeMB
                lastThermalStart = thermalStart
                lastThermalEnd = thermalEnd
                lastThermalChanged = (thermalStart != thermalEnd)

                allAddLatencies.append(contentsOf: addLatencies)
                allSearchLatencies.append(contentsOf: searchLatencies)
            }

            let coldStartMean = trialColdStarts.reduce(0, +) / Double(trialColdStarts.count)
            let addTotalMean = trialAddTotals.reduce(0, +) / Double(trialAddTotals.count)
            let ftsRebuildMean = trialFtsRebuilds.reduce(0, +) / Double(trialFtsRebuilds.count)

            let coldStartCi95 = computeCI95(trialColdStarts)
            let addTotalCi95 = computeCI95(trialAddTotals)
            let ftsRebuildCi95 = computeCI95(trialFtsRebuilds)

            let aggregateAddStats = LatencyStats(latencies: allAddLatencies)
            let aggregateSearchStats = LatencyStats(latencies: allSearchLatencies)

            let sr = ScaleResult(
                scale: scale,
                coldStartMs: coldStartMean,
                coldStartCi95: coldStartCi95,
                addTotalMs: addTotalMean,
                addTotalCi95: addTotalCi95,
                addStats: aggregateAddStats,
                searchStats: aggregateSearchStats,
                memoryDeltaMB: lastMemoryDelta,
                peakMemoryMB: overallPeakMem,
                dbSizeMB: lastDbSizeMB,
                ftsRebuildMs: ftsRebuildMean,
                ftsRebuildCi95: ftsRebuildCi95,
                thermalStateStart: lastThermalStart,
                thermalStateEnd: lastThermalEnd,
                thermalStateChanged: lastThermalChanged,
                addLatenciesRaw: allAddLatencies,
                searchLatenciesRaw: allSearchLatencies
            )
            scaleResults.append(sr)
        }

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]

        return FullBenchmarkResult(
            device: device,
            timestamp: formatter.string(from: Date()),
            embeddingModel: model ?? "text-embedding-3-small",
            embeddingDims: Int(dims ?? 1024),
            numTrials: numTrials,
            scaleResults: scaleResults
        )
    }

    private func computeCI95(_ values: [Double]) -> Double? {
        let n = values.count
        guard n > 1 else { return nil }
        let mean = values.reduce(0, +) / Double(n)
        let variance = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(n - 1)
        let stdDev = sqrt(variance)
        return 1.96 * stdDev / sqrt(Double(n))
    }

    private func cleanupDbFiles(_ path: String) {
        for suffix in ["", ".wal", ".tmp"] {
            try? FileManager.default.removeItem(atPath: path + suffix)
        }
    }

    private func update(_ text: String, progress: Double) {
        DispatchQueue.main.async {
            self.statusText = text
            self.progress = progress
        }
    }
}
