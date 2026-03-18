import SwiftUI

struct UIChatMessage: Identifiable {
    let id = UUID()
    let text: String
    let isUser: Bool
    let isError: Bool
    let timestamp: Date

    init(_ text: String, isUser: Bool = false, isError: Bool = false) {
        self.text = text
        self.isUser = isUser
        self.isError = isError
        self.timestamp = Date()
    }
}

// MARK: - Config Bar (Embedding + LLM)

struct LLMConfigBar: View {
    @EnvironmentObject var api: MemmeAPI
    @State private var isExpanded = true
    @State private var statusMessage: String?
    @State private var statusIsError = false

    // Embedding config
    @State private var embedApiKey = ""
    @State private var embedModel = "text-embedding-3-small"
    @State private var embedBaseUrl = "https://api.openai.com"
    @State private var embedDims: String = "1536"

    // LLM config
    @State private var llmApiKey = ""
    @State private var llmModel = "gpt-4.1-nano"
    @State private var llmBaseUrl = "https://api.openai.com"

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack(spacing: 8) {
                Image(systemName: "gearshape")
                    .font(.system(size: 12))
                    .foregroundColor(Color(hex: "6C5CE7"))
                if !isExpanded && api.isReady {
                    Text("Embed: \(api.embedModel ?? "mock")")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.white.opacity(0.6))
                    if !api.llmApiKey.isEmpty {
                        Text("| LLM: \(api.llmModel)")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundColor(.white.opacity(0.6))
                    }
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 11))
                        .foregroundColor(Color(hex: "00b894"))
                } else {
                    Text(api.l10n.configuration)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(.white.opacity(0.6))
                }
                Spacer()
                Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                    .font(.system(size: 10))
                    .foregroundColor(.white.opacity(0.4))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .contentShape(Rectangle())
            .onTapGesture {
                withAnimation(.easeInOut(duration: 0.2)) { isExpanded.toggle() }
            }

            if isExpanded {
                VStack(spacing: 14) {
                    // — Embedding Section —
                    VStack(alignment: .leading, spacing: 8) {
                        Text(api.l10n.embedding)
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(Color(hex: "c8b6ff"))

                        configField(api.l10n.apiKey, text: $embedApiKey, secure: true)

                        HStack(spacing: 8) {
                            configField(api.l10n.model, text: $embedModel)
                            configField(api.l10n.dims, text: $embedDims)
                                .frame(width: 60)
                        }

                        configField(api.l10n.baseUrl, text: $embedBaseUrl)

                        Button(action: applyEmbedding) {
                            Label(api.isReady && api.embedModel != nil ? api.l10n.reconnectEmbedding : api.l10n.connectEmbedding,
                                  systemImage: "arrow.triangle.2.circlepath")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(MemmeButtonStyle(color: Color(hex: "00b894")))
                        .disabled(embedApiKey.isEmpty)
                    }

                    Divider().background(Color.white.opacity(0.1))

                    // — LLM Section —
                    VStack(alignment: .leading, spacing: 8) {
                        Text(api.l10n.llmForCompact)
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(Color(hex: "c8b6ff"))

                        configField(api.l10n.apiKey, text: $llmApiKey, secure: true)

                        HStack(spacing: 8) {
                            configField(api.l10n.model, text: $llmModel)
                            configField(api.l10n.baseUrl, text: $llmBaseUrl)
                        }

                        Button(action: applyLLM) {
                            Label(api.l10n.setLLM, systemImage: "checkmark.circle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(MemmeButtonStyle(color: Color(hex: "6C5CE7")))
                        .disabled(llmApiKey.isEmpty)
                    }

                    // — Status —
                    if let msg = statusMessage {
                        Text(msg)
                            .font(.system(size: 11))
                            .foregroundColor(statusIsError ? Color(hex: "e17055") : Color(hex: "00b894"))
                            .transition(.opacity)
                    }

                    Text(api.l10n.configHint)
                        .font(.system(size: 10))
                        .foregroundColor(.white.opacity(0.25))
                        .lineLimit(3)
                }
                .padding(.horizontal, 14)
                .padding(.bottom, 12)
            }
        }
        .background(Color(hex: "1a1a2e"))
        .cornerRadius(10)
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.white.opacity(0.06), lineWidth: 1)
        )
        .onAppear {
            llmApiKey = api.llmApiKey
            llmModel = api.llmModel
            llmBaseUrl = api.llmBaseUrl
            if !api.llmApiKey.isEmpty && api.isReady {
                isExpanded = false
            }
        }
    }

    @ViewBuilder
    private func configField(_ placeholder: String, text: Binding<String>, secure: Bool = false) -> some View {
        Group {
            if secure {
                SecureField(placeholder, text: text)
            } else {
                TextField(placeholder, text: text)
            }
        }
        .textFieldStyle(.plain)
        .font(.system(size: 12))
        .padding(7)
        .background(Color(hex: "16162a"))
        .cornerRadius(6)
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color.white.opacity(0.08), lineWidth: 1)
        )
    }

    private func normalizeBaseUrl(_ url: String) -> String {
        var u = url.trimmingCharacters(in: .whitespacesAndNewlines)
        // 去掉末尾斜杠
        while u.hasSuffix("/") { u = String(u.dropLast()) }
        // 去掉用户可能多填的 endpoint 路径
        for suffix in ["/embeddings", "/chat/completions", "/completions"] {
            if u.hasSuffix(suffix) { u = String(u.dropLast(suffix.count)); break }
        }
        while u.hasSuffix("/") { u = String(u.dropLast()) }
        // 确保以 /v1 结尾
        if !u.hasSuffix("/v1") { u += "/v1" }
        return u
    }

    private func applyEmbedding() {
        let dims = UInt32(embedDims) ?? 1536
        let url = normalizeBaseUrl(embedBaseUrl)
        embedBaseUrl = url
        api.initWithApiKey(
            apiKey: embedApiKey,
            baseUrl: url,
            model: embedModel.isEmpty ? nil : embedModel,
            dims: dims
        )
        if api.isReady {
            showStatus("Embedding connected: \(embedModel)", isError: false)
        } else {
            showStatus(api.errorMessage ?? "Failed to connect", isError: true)
        }
    }

    private func applyLLM() {
        let url = normalizeBaseUrl(llmBaseUrl)
        llmBaseUrl = url
        api.llmApiKey = llmApiKey
        api.llmModel = llmModel
        api.llmBaseUrl = url
        showStatus("LLM set: \(llmModel)", isError: false)
    }

    private func showStatus(_ msg: String, isError: Bool) {
        statusIsError = isError
        withAnimation { statusMessage = msg }
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
            withAnimation { statusMessage = nil }
            if !isError {
                withAnimation(.easeInOut(duration: 0.2)) { isExpanded = false }
            }
        }
    }
}

// MARK: - Chat Input View

struct ChatInputView: View {
    @EnvironmentObject var api: MemmeAPI
    @State private var inputText = ""
    @State private var messages: [UIChatMessage] = []
    @State private var isLoading = false
    @State private var welcomeShown = false
    @State private var showTestData = false
    @State private var loadedConvIds: Set<Int> = []

    var body: some View {
        VStack(spacing: 0) {
            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(messages) { msg in
                            MessageBubble(message: msg)
                                .id(msg.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: messages.count) { _ in
                    if let last = messages.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }

            // Test Data Panel
            if showTestData {
                TestDataPanel(
                    loadedIds: $loadedConvIds,
                    isLoading: $isLoading,
                    onLoad: { conv in loadTestConversation(conv) }
                )
            }

            Divider()

            // Input
            VStack(spacing: 10) {
                TextEditor(text: $inputText)
                    .font(.system(size: 14))
                    .frame(minHeight: 60, maxHeight: 100)
                    .padding(8)
                    .background(Color(hex: "16162a"))
                    .cornerRadius(10)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(Color.white.opacity(0.1), lineWidth: 1)
                    )

                HStack(spacing: 10) {
                    Label(api.l10n.push, systemImage: "arrow.up.circle.fill")
                        .frame(maxWidth: .infinity)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(.white)
                        .padding(.vertical, 10)
                        .background(inputText.isEmpty || isLoading ? Color(hex: "6C5CE7").opacity(0.3) : Color(hex: "6C5CE7"))
                        .cornerRadius(10)
                        .contentShape(Rectangle())
                        .onTapGesture { if !inputText.isEmpty && !isLoading { push() } }

                    HStack {
                        if isLoading {
                            ProgressView()
                                .scaleEffect(0.7)
                                .tint(.white)
                        }
                        Label(api.l10n.pushAndCompact, systemImage: "sparkles")
                    }
                    .frame(maxWidth: .infinity)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.white)
                    .padding(.vertical, 10)
                    .background(inputText.isEmpty || isLoading ? Color(hex: "c8b6ff").opacity(0.3) : Color(hex: "c8b6ff"))
                    .cornerRadius(10)
                    .contentShape(Rectangle())
                    .onTapGesture { if !inputText.isEmpty && !isLoading { pushAndCompact() } }

                    // One-Click Test
                    Text(api.l10n.oneClickTest)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(isLoading ? .white.opacity(0.3) : Color(hex: "00b894"))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 10)
                        .background(isLoading ? Color(hex: "1a1a2e").opacity(0.5) : Color(hex: "00b894").opacity(0.15))
                        .cornerRadius(10)
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(Color(hex: "00b894").opacity(0.3), lineWidth: 1)
                        )
                        .contentShape(Rectangle())
                        .onTapGesture { if !isLoading { runOneClickTest() } }
                }
            }
            .padding()
        }
        .background(Color(hex: "12121a"))
        .onAppear {
            if !welcomeShown {
                messages.append(UIChatMessage(api.l10n.welcome))
                welcomeShown = true
            }
        }
    }

    private func push() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        messages.append(UIChatMessage(text, isUser: true))
        inputText = ""

        Task {
            isLoading = true
            defer { isLoading = false }
            do {
                let result = try await api.addMemory(content: text)
                messages.append(UIChatMessage("\(api.l10n.memoryAdded): \"\(result.content.prefix(80))...\""))
            } catch {
                messages.append(UIChatMessage("Error: \(error.localizedDescription)", isError: true))
            }
        }
    }

    private func loadTestConversation(_ conv: TestConversation) {
        messages.append(UIChatMessage("[\(conv.localTitle(api.language))] \(api.l10n.loading)"))

        Task {
            isLoading = true
            defer { isLoading = false }
            var count = 0
            for msg in conv.messages {
                let text = "\(msg.role): \(msg.content)"
                do {
                    try await api.addMemory(content: text)
                    count += 1
                } catch {
                    messages.append(UIChatMessage("Error: \(error.localizedDescription)", isError: true))
                    return
                }
            }
            loadedConvIds.insert(conv.id)
            messages.append(UIChatMessage("\(api.l10n.pushed) \"\(conv.localTitle(api.language))\" — \(count) \(api.l10n.memories)"))
        }
    }

    private func runOneClickTest() {
        // Check embedding is configured
        guard api.embedModel != nil else {
            messages.append(UIChatMessage(api.l10n.configEmbeddingFirst, isError: true))
            return
        }

        Task {
            isLoading = true
            defer { isLoading = false }

            // Step 1: Push all test conversations
            messages.append(UIChatMessage("🚀 \(api.l10n.testStep1)"))

            let allConvs = testConversations
            var totalPushed = 0

            for conv in allConvs {
                for msg in conv.messages {
                    let text = "\(msg.role): \(msg.content)"
                    do {
                        try await api.addMemory(content: text)
                        totalPushed += 1
                    } catch {
                        messages.append(UIChatMessage("Error: \(error.localizedDescription)", isError: true))
                        return
                    }
                }
                loadedConvIds.insert(conv.id)
                messages.append(UIChatMessage("  ✓ \(conv.localTitle(api.language)) (\(conv.messages.count) msgs)"))

                // Small delay for visual feedback
                try? await Task.sleep(nanoseconds: 200_000_000)
            }

            messages.append(UIChatMessage("✅ \(api.l10n.testStep1Done): \(totalPushed) \(api.l10n.memories)"))
            try? await Task.sleep(nanoseconds: 500_000_000)

            // Step 2: Search verification
            messages.append(UIChatMessage("🔍 \(api.l10n.testStep2)"))

            let queries = [api.l10n.testSearchQuery1, api.l10n.testSearchQuery2, api.l10n.testSearchQuery3]
            for query in queries {
                do {
                    let results = try await api.search(query: query)
                    let preview = results.first.map { "\"\($0.content.prefix(60))...\"" } ?? "-"
                    messages.append(UIChatMessage("  🔎 \(api.l10n.testSearching) \"\(query)\" → \(api.l10n.testFound) \(results.count) \(api.l10n.testResults)\n  \(preview)"))
                } catch {
                    messages.append(UIChatMessage("  Search error: \(error.localizedDescription)", isError: true))
                }
                try? await Task.sleep(nanoseconds: 300_000_000)
            }

            // Trigger auto-search on SearchView with last query
            api.autoSearchQuery = queries.last

            messages.append(UIChatMessage("🎉 \(api.l10n.testComplete)"))
        }
    }

    private func pushAndCompact() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        messages.append(UIChatMessage(text, isUser: true))
        inputText = ""

        Task {
            isLoading = true
            defer { isLoading = false }
            do {
                messages.append(UIChatMessage(api.l10n.extracting))
                let result = try await api.smartAdd(text: text)
                let count = result.memories.count
                let graphInfo = result.graph.map { " | \($0.entities.count) \(api.l10n.entities), \($0.relations.count) \(api.l10n.relations)" } ?? ""
                messages.append(UIChatMessage("Compact: \(api.l10n.extracted) \(count) \(api.l10n.memories)\(graphInfo)"))
            } catch {
                messages.append(UIChatMessage("Compact error: \(error.localizedDescription)", isError: true))
            }
        }
    }
}

// MARK: - Test Data

struct TestConversation: Identifiable {
    let id: Int
    let title: String
    let titleZh: String
    let messages: [(role: String, content: String)]

    func localTitle(_ lang: AppLanguage) -> String { lang == .zh ? titleZh : title }
}

let testConversations: [TestConversation] = [
    TestConversation(
        id: 0,
        title: "Self Introduction",
        titleZh: "自我介绍",
        messages: [
            ("user", "Hi! I'm Alex Chen, I'm 28 years old. I work as a software engineer at TechCorp in Beijing."),
            ("assistant", "Nice to meet you, Alex! A software engineer at TechCorp in Beijing — sounds exciting. What kind of projects do you work on?"),
            ("user", "Mostly backend stuff with Rust and Go. I also love photography and hiking on weekends."),
            ("assistant", "Great combo! Rust and Go for work, photography and hiking for the soul. Do you have a favorite hiking trail?"),
        ]
    ),
    TestConversation(
        id: 1,
        title: "Family & Friends",
        titleZh: "家人和朋友",
        messages: [
            ("user", "My mom's name is Li Wei, she's a retired teacher. My dad Zhang Guoqiang runs a small bookstore."),
            ("assistant", "A teacher and a bookstore owner — sounds like you grew up surrounded by knowledge! Are they in Beijing too?"),
            ("user", "No, they're in Hangzhou. My best friend Sarah is here in Beijing though, she's a photographer at a magazine."),
            ("assistant", "That's nice to have Sarah nearby. Do you and Sarah go on photography trips together?"),
            ("user", "Yeah! We went to Inner Mongolia last month for astrophotography. It was amazing."),
        ]
    ),
    TestConversation(
        id: 2,
        title: "Daily Routine",
        titleZh: "日常生活",
        messages: [
            ("user", "I usually wake up at 7am and go for a run in Chaoyang Park. Then I grab coffee at Blue Bottle near my office."),
            ("assistant", "Morning runs in Chaoyang Park sound lovely! How far do you usually run?"),
            ("user", "About 5km. I'm training for the Beijing Marathon in October. My current best is 4:15."),
            ("assistant", "4:15 is solid! What's your target time for the marathon?"),
            ("user", "I want to break 4 hours. My coach thinks it's doable if I keep up the training."),
        ]
    ),
    TestConversation(
        id: 3,
        title: "Work Project",
        titleZh: "工作项目",
        messages: [
            ("user", "We just launched Project Orion at work — it's a distributed storage system written in Rust."),
            ("assistant", "Congrats on the launch! What makes Orion special compared to existing solutions?"),
            ("user", "It's optimized for edge computing. Sub-millisecond reads at the edge. My team lead is Wang Jun, he designed the core architecture."),
            ("assistant", "Edge-optimized with sub-ms reads — that's impressive. How big is the team?"),
            ("user", "8 people. We've been working on it for 14 months. The CEO personally presented it to the board."),
        ]
    ),
    TestConversation(
        id: 4,
        title: "Hobbies & Learning",
        titleZh: "兴趣爱好",
        messages: [
            ("user", "I started learning piano 6 months ago. My teacher Ms. Liu is super patient with me."),
            ("assistant", "That's wonderful! What pieces are you working on?"),
            ("user", "Chopin's Nocturne Op.9 No.2. It's hard but beautiful. I practice every evening for about 30 minutes."),
            ("assistant", "A classic choice! The Nocturne is challenging but so rewarding. Do you have a piano at home?"),
            ("user", "Yeah, a Yamaha digital piano. I also have a golden retriever named Max who always sits next to me when I play."),
        ]
    ),
    TestConversation(
        id: 5,
        title: "Travel Plans",
        titleZh: "旅行计划",
        messages: [
            ("user", "I went to Tokyo last spring with my girlfriend Emma. We stayed in Shibuya for a week."),
            ("assistant", "Tokyo is amazing! What were the highlights of the trip?"),
            ("user", "The food was incredible — we found this tiny ramen shop in Shinjuku, 8 seats only. Also visited TeamLab Borderless."),
            ("assistant", "TeamLab is such an experience! Are you planning any trips next?"),
            ("user", "We're thinking about Iceland next summer. Emma really wants to see the Northern Lights. She's doing her PhD at Tsinghua in astrophysics."),
        ]
    ),
]

struct TestDataPanel: View {
    @EnvironmentObject var api: MemmeAPI
    @Binding var loadedIds: Set<Int>
    @Binding var isLoading: Bool
    let onLoad: (TestConversation) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "flask")
                    .font(.system(size: 11))
                    .foregroundColor(Color(hex: "00b894"))
                Text(api.l10n.testConversations)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(Color(hex: "00b894"))
                Spacer()
                Text(api.l10n.tapToLoad)
                    .font(.system(size: 10))
                    .foregroundColor(.white.opacity(0.35))
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(testConversations) { conv in
                        TestConvCard(
                            conv: conv,
                            isLoaded: loadedIds.contains(conv.id),
                            isLoading: isLoading
                        ) {
                            onLoad(conv)
                        }
                    }
                }
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
            }
        }
        .background(Color(hex: "0f1a14"))
        .overlay(
            Rectangle()
                .frame(height: 1)
                .foregroundColor(Color(hex: "00b894").opacity(0.2)),
            alignment: .top
        )
    }
}

struct TestConvCard: View {
    @EnvironmentObject var api: MemmeAPI
    let conv: TestConversation
    let isLoaded: Bool
    let isLoading: Bool
    let onTap: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                if isLoaded {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 10))
                        .foregroundColor(Color(hex: "00b894"))
                }
                Text(conv.localTitle(api.language))
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(isLoaded ? Color(hex: "00b894") : .white)
            }
            Text("\(conv.messages.count) messages")
                .font(.system(size: 10))
                .foregroundColor(.white.opacity(0.4))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(isLoaded ? Color(hex: "00b894").opacity(0.1) : Color(hex: "1a1a2e"))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isLoaded ? Color(hex: "00b894").opacity(0.3) : Color.white.opacity(0.06), lineWidth: 1)
        )
        .opacity(isLoading ? 0.5 : 1)
        .contentShape(Rectangle())
        .onTapGesture {
            if !isLoaded && !isLoading { onTap() }
        }
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: UIChatMessage

    var body: some View {
        HStack {
            if message.isUser { Spacer() }
            Text(message.text)
                .font(.system(size: 13))
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(backgroundColor)
                .foregroundColor(message.isError ? Color(hex: "e17055") : .white)
                .cornerRadius(12)
            if !message.isUser { Spacer() }
        }
    }

    private var backgroundColor: Color {
        if message.isError { return Color(hex: "e17055").opacity(0.2) }
        if message.isUser { return Color(hex: "6C5CE7") }
        return Color(hex: "1a1a2e")
    }
}

// MARK: - Shared Styles

struct MemmeButtonStyle: ButtonStyle {
    let color: Color
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .semibold))
            .foregroundColor(.white)
            .padding(.vertical, 10)
            .background(color.opacity(configuration.isPressed ? 0.7 : 1))
            .cornerRadius(10)
    }
}

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let r = Double((int >> 16) & 0xFF) / 255
        let g = Double((int >> 8) & 0xFF) / 255
        let b = Double(int & 0xFF) / 255
        self.init(red: r, green: g, blue: b)
    }
}
