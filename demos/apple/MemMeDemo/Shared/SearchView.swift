import SwiftUI

struct SearchView: View {
    @EnvironmentObject var api: MemmeAPI
    @State private var searchQuery = ""
    @State private var memories: [MemoryResult] = []
    @State private var graphResult: GraphSearchResult?
    @State private var isLoading = false
    @State private var hasSearched = false
    @State private var errorMessage: String?

    var body: some View {
        VStack(spacing: 0) {
            // Search Bar
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(Color(hex: "aaaaaa"))
                TextField(api.l10n.searchMemories, text: $searchQuery)
                    .textFieldStyle(.plain)
                    .font(.system(size: 14))
                    .onSubmit { search() }
                if !searchQuery.isEmpty {
                    Button(action: { searchQuery = "" }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(Color(hex: "aaaaaa"))
                    }
                    .buttonStyle(.plain)
                }
                Button(api.l10n.search) { search() }
                    .buttonStyle(MemmeButtonStyle(color: Color(hex: "6C5CE7")))
                    .disabled(searchQuery.isEmpty)
            }
            .padding()
            .background(Color(hex: "12121a"))

            Divider()

            // Results
            if isLoading {
                Spacer()
                ProgressView()
                    .tint(Color(hex: "6C5CE7"))
                Spacer()
            } else if !hasSearched {
                Spacer()
                VStack(spacing: 12) {
                    Image(systemName: "magnifyingglass")
                        .font(.system(size: 40))
                        .foregroundColor(Color(hex: "aaaaaa"))
                    Text(api.l10n.searchToStart)
                        .foregroundColor(Color(hex: "aaaaaa"))
                        .font(.system(size: 14))
                }
                Spacer()
            } else if memories.isEmpty && graphResult == nil {
                Spacer()
                VStack(spacing: 12) {
                    Image(systemName: "magnifyingglass")
                        .font(.system(size: 40))
                        .foregroundColor(Color(hex: "aaaaaa"))
                    Text(api.l10n.noResults)
                        .foregroundColor(Color(hex: "aaaaaa"))
                }
                Spacer()
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        // Memories Section
                        if !memories.isEmpty {
                            SectionHeader(title: "Memories", count: memories.count, icon: "brain.head.profile")

                            ForEach(memories) { memory in
                                MemoryCard(memory: memory, onDelete: { deleteMemory(memory.id) })
                            }
                        }

                        // Knowledge Graph Section (inline)
                        if let graph = graphResult, !graph.entities.isEmpty {
                            SectionHeader(title: api.l10n.knowledgeGraph, count: graph.entities.count, icon: "point.3.connected.trianglepath.dotted")

                            InlineGraphView(graph: graph)
                                .frame(height: 300)
                                .background(Color(hex: "12121a"))
                                .cornerRadius(12)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(Color.white.opacity(0.06), lineWidth: 1)
                                )

                            // Relations list below graph
                            if !graph.relations.isEmpty {
                                VStack(alignment: .leading, spacing: 6) {
                                    ForEach(graph.relations) { rel in
                                        HStack(spacing: 6) {
                                            Text(rel.source)
                                                .font(.system(size: 12, weight: .semibold))
                                                .foregroundColor(Color(hex: "6C5CE7"))
                                            Image(systemName: "arrow.right")
                                                .font(.system(size: 9))
                                                .foregroundColor(.white.opacity(0.3))
                                            Text(rel.relationType)
                                                .font(.system(size: 11))
                                                .foregroundColor(.white.opacity(0.5))
                                            Image(systemName: "arrow.right")
                                                .font(.system(size: 9))
                                                .foregroundColor(.white.opacity(0.3))
                                            Text(rel.target)
                                                .font(.system(size: 12, weight: .semibold))
                                                .foregroundColor(Color(hex: "c8b6ff"))
                                        }
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 6)
                                        .background(Color(hex: "1a1a2e"))
                                        .cornerRadius(8)
                                    }
                                }
                            }
                        }
                    }
                    .padding()
                }
            }
        }
        .background(Color(hex: "0a0a0f"))
        .alert("Error", isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
            Button("OK") { errorMessage = nil }
        } message: {
            Text(errorMessage ?? "")
        }
        .onChange(of: api.autoSearchQuery) { query in
            if let q = query {
                searchQuery = q
                api.autoSearchQuery = nil
                search()
            }
        }
    }

    private func search() {
        guard !searchQuery.isEmpty else { return }
        Task {
            isLoading = true
            hasSearched = true
            defer { isLoading = false }
            do {
                let result = try await api.unifiedSearch(query: searchQuery)
                memories = result.memories
                graphResult = result.graph
            } catch {
                memories = []
                graphResult = nil
                errorMessage = error.localizedDescription
            }
        }
    }

    private func deleteMemory(_ id: String) {
        Task {
            do {
                try await api.deleteMemory(id: id)
                memories.removeAll { $0.id == id }
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}

// MARK: - Section Header

struct SectionHeader: View {
    let title: String
    let count: Int
    let icon: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 13))
                .foregroundColor(Color(hex: "6C5CE7"))
            Text(title)
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(.white.opacity(0.8))
            Text("\(count)")
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(Color(hex: "c8b6ff"))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color(hex: "6C5CE7").opacity(0.2))
                .cornerRadius(6)
            Spacer()
        }
        .padding(.top, 4)
    }
}

// MARK: - Inline Graph View (Canvas-based, embedded in SearchView)

struct InlineGraphView: View {
    let graph: GraphSearchResult
    @State private var nodes: [GraphNode] = []
    @State private var edges: [GraphEdge] = []
    @State private var timer: Timer?
    @State private var draggedNodeIdx: Int?

    var body: some View {
        GeometryReader { geo in
            Canvas { ctx, size in
                // Draw edges
                for edge in edges {
                    guard edge.sourceIdx < nodes.count, edge.targetIdx < nodes.count else { continue }
                    let s = nodes[edge.sourceIdx]
                    let t = nodes[edge.targetIdx]
                    var path = Path()
                    path.move(to: CGPoint(x: s.x, y: s.y))
                    path.addLine(to: CGPoint(x: t.x, y: t.y))
                    ctx.stroke(path, with: .color(.white.opacity(0.15)), lineWidth: 1.5)

                    let midX = (s.x + t.x) / 2
                    let midY = (s.y + t.y) / 2
                    ctx.draw(
                        Text(edge.label).font(.system(size: 9)).foregroundColor(.white.opacity(0.3)),
                        at: CGPoint(x: midX, y: midY)
                    )
                }

                // Draw nodes
                for node in nodes {
                    let r: CGFloat = 14
                    let rect = CGRect(x: node.x - r, y: node.y - r, width: r * 2, height: r * 2)
                    ctx.fill(Path(ellipseIn: rect), with: .color(node.color))
                    ctx.stroke(Path(ellipseIn: rect), with: .color(.white.opacity(0.2)), lineWidth: 1.5)

                    ctx.draw(
                        Text(node.name).font(.system(size: 11, weight: .medium)).foregroundColor(.white),
                        at: CGPoint(x: node.x, y: node.y - r - 10)
                    )
                }
            }
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        if draggedNodeIdx == nil {
                            draggedNodeIdx = nearestNode(to: value.startLocation)
                        }
                        if let idx = draggedNodeIdx {
                            nodes[idx].x = value.location.x
                            nodes[idx].y = value.location.y
                        }
                    }
                    .onEnded { _ in
                        draggedNodeIdx = nil
                    }
            )
            .onAppear {
                buildGraph()
                initPositions(size: geo.size)
                startSimulation(size: geo.size)
            }
            .onDisappear { stopSimulation() }
            .onChange(of: graph.entities.count) { _ in
                buildGraph()
                initPositions(size: geo.size)
            }
        }
    }

    private func buildGraph() {
        let newNodes = graph.entities.map { e in
            GraphNode(id: e.name, name: e.name, type: e.entityType ?? "default",
                     x: CGFloat.random(in: 100...500), y: CGFloat.random(in: 50...250))
        }
        let nameToIdx = Dictionary(uniqueKeysWithValues: newNodes.enumerated().map { ($0.element.name, $0.offset) })
        let newEdges = graph.relations.compactMap { r -> GraphEdge? in
            guard let si = nameToIdx[r.source], let ti = nameToIdx[r.target] else { return nil }
            return GraphEdge(sourceIdx: si, targetIdx: ti, label: r.relationType)
        }
        nodes = newNodes
        edges = newEdges
    }

    private func nearestNode(to point: CGPoint) -> Int? {
        var minDist: CGFloat = 30
        var best: Int?
        for (i, node) in nodes.enumerated() {
            let d = hypot(node.x - point.x, node.y - point.y)
            if d < minDist { minDist = d; best = i }
        }
        return best
    }

    private func initPositions(size: CGSize) {
        let cx = size.width / 2
        let cy = size.height / 2
        for i in nodes.indices {
            let angle = CGFloat(i) / CGFloat(max(nodes.count, 1)) * .pi * 2
            let r = min(size.width, size.height) * 0.3
            nodes[i].x = cx + cos(angle) * r + CGFloat.random(in: -20...20)
            nodes[i].y = cy + sin(angle) * r + CGFloat.random(in: -20...20)
        }
    }

    private func startSimulation(size: CGSize) {
        let centerX = size.width / 2
        let centerY = size.height / 2
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { _ in
            simulationStep(centerX: centerX, centerY: centerY)
        }
    }

    private func stopSimulation() {
        timer?.invalidate()
        timer = nil
    }

    private func simulationStep(centerX: CGFloat, centerY: CGFloat) {
        let damping: CGFloat = 0.9
        let repulsion: CGFloat = 5000
        let springLength: CGFloat = 120
        let springK: CGFloat = 0.02
        let centerPull: CGFloat = 0.005

        guard !nodes.isEmpty else { return }

        for i in nodes.indices {
            for j in (i + 1)..<nodes.count {
                let dx = nodes[i].x - nodes[j].x
                let dy = nodes[i].y - nodes[j].y
                let dist = max(hypot(dx, dy), 1)
                let force = repulsion / (dist * dist)
                let fx = (dx / dist) * force
                let fy = (dy / dist) * force
                nodes[i].vx += fx
                nodes[i].vy += fy
                nodes[j].vx -= fx
                nodes[j].vy -= fy
            }
        }

        for edge in edges {
            guard edge.sourceIdx < nodes.count, edge.targetIdx < nodes.count else { continue }
            let dx = nodes[edge.targetIdx].x - nodes[edge.sourceIdx].x
            let dy = nodes[edge.targetIdx].y - nodes[edge.sourceIdx].y
            let dist = max(hypot(dx, dy), 1)
            let force = (dist - springLength) * springK
            let fx = (dx / dist) * force
            let fy = (dy / dist) * force
            nodes[edge.sourceIdx].vx += fx
            nodes[edge.sourceIdx].vy += fy
            nodes[edge.targetIdx].vx -= fx
            nodes[edge.targetIdx].vy -= fy
        }

        for i in nodes.indices {
            nodes[i].vx -= (nodes[i].x - centerX) * centerPull
            nodes[i].vy -= (nodes[i].y - centerY) * centerPull
            nodes[i].vx *= damping
            nodes[i].vy *= damping
            nodes[i].x += nodes[i].vx
            nodes[i].y += nodes[i].vy
        }
    }
}

// MARK: - Memory Card

struct MemoryCard: View {
    let memory: MemoryResult
    let onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(memory.content)
                .font(.system(size: 13))
                .lineLimit(4)
                .foregroundColor(.white)

            HStack {
                if !memory.createdAt.isEmpty {
                    Text(formatDate(memory.createdAt))
                        .font(.system(size: 11))
                        .foregroundColor(Color(hex: "aaaaaa"))
                }
                Spacer()
                if let score = memory.score {
                    Text("\(Int(score * 100))%")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(Color(hex: "c8b6ff"))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(Color(hex: "6C5CE7").opacity(0.2))
                        .cornerRadius(10)
                }
                if let cats = memory.categories, !cats.isEmpty {
                    ForEach(cats.prefix(2), id: \.self) { cat in
                        Text(cat)
                            .font(.system(size: 10))
                            .foregroundColor(Color(hex: "74b9ff"))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color(hex: "74b9ff").opacity(0.15))
                            .cornerRadius(8)
                    }
                }
            }
        }
        .padding(14)
        .background(Color(hex: "12121a"))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.white.opacity(0.06), lineWidth: 1)
        )
        .contextMenu {
            Button(role: .destructive, action: onDelete) {
                Label("Delete", systemImage: "trash")
            }
        }
    }

    private func formatDate(_ dateStr: String) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        guard let date = formatter.date(from: dateStr) else { return dateStr }
        let rel = RelativeDateTimeFormatter()
        rel.unitsStyle = .short
        return rel.localizedString(for: date, relativeTo: Date())
    }
}
