import SwiftUI

struct StatsView: View {
    @EnvironmentObject var api: MemmeAPI
    @State private var stats: UserStats?
    @State private var topEntities: [EntityStat] = []
    @State private var isLoading = false
    @State private var errorMessage: String?

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Summary Cards
                HStack(spacing: 12) {
                    StatCard(value: Int(stats?.totalMemories ?? 0), label: "Memories", icon: "brain.head.profile")
                    StatCard(value: Int(stats?.totalEntities ?? 0), label: "Entities", icon: "person.3.fill")
                    StatCard(value: Int(stats?.totalRelationships ?? 0), label: "Relations", icon: "point.3.connected.trianglepath.dotted")
                }

                // Top Entities
                VStack(alignment: .leading, spacing: 12) {
                    Text(api.l10n.topEntities)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(Color(hex: "aaaaaa"))

                    if topEntities.isEmpty {
                        HStack {
                            Spacer()
                            VStack(spacing: 8) {
                                Image(systemName: "chart.bar.fill")
                                    .font(.system(size: 30))
                                    .foregroundColor(Color(hex: "aaaaaa"))
                                Text(api.l10n.noEntities)
                                    .font(.system(size: 13))
                                    .foregroundColor(Color(hex: "aaaaaa"))
                            }
                            .padding(.vertical, 30)
                            Spacer()
                        }
                    } else {
                        ForEach(topEntities) { entity in
                            EntityRow(entity: entity)
                        }
                    }
                }
                .padding()
                .background(Color(hex: "12121a"))
                .cornerRadius(12)
            }
            .padding()
        }
        .background(Color(hex: "0a0a0f"))
        .overlay {
            if isLoading {
                ProgressView()
                    .tint(Color(hex: "6C5CE7"))
            }
        }
        .alert("Error", isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
            Button("OK") { errorMessage = nil }
        } message: {
            Text(errorMessage ?? "")
        }
        .onAppear { loadStats() }
        .refreshable { loadStats() }
    }

    private func loadStats() {
        Task {
            isLoading = true
            defer { isLoading = false }
            do {
                stats = try await api.userStats()
                topEntities = try await api.topEntities()
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}

struct StatCard: View {
    let value: Int
    let label: String
    let icon: String

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 20))
                .foregroundColor(Color(hex: "6C5CE7"))
            Text("\(value)")
                .font(.system(size: 28, weight: .bold))
                .foregroundColor(Color(hex: "c8b6ff"))
            Text(label)
                .font(.system(size: 12))
                .foregroundColor(Color(hex: "aaaaaa"))
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 16)
        .background(Color(hex: "12121a"))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.white.opacity(0.06), lineWidth: 1)
        )
    }
}

struct EntityRow: View {
    let entity: EntityStat

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(entity.name)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.white)
                if let type = entity.entityType {
                    Text(type)
                        .font(.system(size: 11))
                        .foregroundColor(Color(hex: "c8b6ff"))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(Color(hex: "6C5CE7").opacity(0.2))
                        .cornerRadius(8)
                }
            }
            Spacer()
            Text("\(entity.relationshipCount) links")
                .font(.system(size: 12))
                .foregroundColor(Color(hex: "aaaaaa"))
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(Color(hex: "1a1a2e"))
        .cornerRadius(10)
    }
}
