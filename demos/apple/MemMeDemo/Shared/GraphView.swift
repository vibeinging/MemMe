import SwiftUI

struct GraphNode: Identifiable {
    let id: String
    let name: String
    let type: String
    var x: CGFloat
    var y: CGFloat
    var vx: CGFloat = 0
    var vy: CGFloat = 0

    var color: Color {
        switch type.lowercased() {
        case "person": return Color(hex: "6C5CE7")
        case "location": return Color(hex: "00b894")
        case "organization": return Color(hex: "fdcb6e")
        case "event": return Color(hex: "e17055")
        case "concept": return Color(hex: "74b9ff")
        default: return Color(hex: "a29bfe")
        }
    }
}

struct GraphEdge: Identifiable {
    let id = UUID()
    let sourceIdx: Int
    let targetIdx: Int
    let label: String
}

// NOTE: GraphView is currently unused as a standalone view (graph is shown inline via InlineGraphView in SearchView).
// Kept for the shared type definitions above and potential future use.
struct GraphView: View {
    @EnvironmentObject var api: MemmeAPI
    @State private var nodes: [GraphNode] = []
    @State private var edges: [GraphEdge] = []
    @State private var searchQuery = ""
    @State private var isLoading = false
    @State private var selectedNode: GraphNode?
    @State private var timer: Timer?
    @State private var draggedNodeIdx: Int?

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Image(systemName: "point.3.connected.trianglepath.dotted")
                    .foregroundColor(Color(hex: "888"))
                TextField("Search entities...", text: $searchQuery)
                    .textFieldStyle(.plain)
                    .onSubmit { loadGraph() }
                Button("Search") { loadGraph() }
                    .buttonStyle(MemmeButtonStyle(color: Color(hex: "6C5CE7")))
            }
            .padding()
            .background(Color(hex: "12121a"))

            Divider()

            if isLoading {
                Spacer()
                ProgressView().tint(Color(hex: "6C5CE7"))
                Spacer()
            } else if nodes.isEmpty {
                Spacer()
                VStack(spacing: 12) {
                    Image(systemName: "point.3.connected.trianglepath.dotted")
                        .font(.system(size: 40))
                        .foregroundColor(Color(hex: "888"))
                    Text("No graph data")
                        .foregroundColor(Color(hex: "888"))
                }
                Spacer()
            } else {
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

                            // Edge label
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

                            // Label
                            ctx.draw(
                                Text(node.name).font(.system(size: 11, weight: .medium)).foregroundColor(.white),
                                at: CGPoint(x: node.x, y: node.y - r - 10)
                            )
                        }
                    }
                    .background(Color(hex: "12121a"))
                    .cornerRadius(12)
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
                        initPositions(size: geo.size)
                        startSimulation(size: geo.size)
                    }
                    .onDisappear { stopSimulation() }
                }
            }
        }
        .background(Color(hex: "0a0a0f"))
        .onAppear { loadGraph() }
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

        // Repulsion between all pairs
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

        // Spring forces for edges
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

        // Center pull (uses dynamic center from geometry)
        for i in nodes.indices {
            nodes[i].vx -= (nodes[i].x - centerX) * centerPull
            nodes[i].vy -= (nodes[i].y - centerY) * centerPull
            nodes[i].vx *= damping
            nodes[i].vy *= damping
            nodes[i].x += nodes[i].vx
            nodes[i].y += nodes[i].vy
        }
    }

    private func loadGraph() {
        Task {
            isLoading = true
            defer { isLoading = false }
            do {
                let query = searchQuery.isEmpty ? "all" : searchQuery
                let result = try await api.searchGraph(query: query)
                buildGraph(from: result)
            } catch {
                nodes = []
                edges = []
            }
        }
    }

    private func buildGraph(from result: GraphSearchResult) {
        let newNodes = result.entities.map { e in
            GraphNode(id: e.name, name: e.name, type: e.entityType ?? "default",
                     x: CGFloat.random(in: 100...700), y: CGFloat.random(in: 100...500))
        }
        let nameToIdx = Dictionary(uniqueKeysWithValues: newNodes.enumerated().map { ($0.element.name, $0.offset) })
        let newEdges = result.relations.compactMap { r -> GraphEdge? in
            guard let si = nameToIdx[r.source], let ti = nameToIdx[r.target] else { return nil }
            return GraphEdge(sourceIdx: si, targetIdx: ti, label: r.relationType)
        }
        nodes = newNodes
        edges = newEdges
    }
}
