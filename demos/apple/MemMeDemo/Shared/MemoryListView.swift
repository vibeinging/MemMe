import SwiftUI

struct MemoryListView: View {
    @EnvironmentObject var api: MemmeAPI
    @State private var memories: [MemoryResult] = []
    @State private var searchQuery = ""
    @State private var isLoading = false
    @State private var errorMessage: String?

    var body: some View {
        VStack(spacing: 0) {
            // Search Bar
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(Color(hex: "888"))
                TextField("Search memories...", text: $searchQuery)
                    .textFieldStyle(.plain)
                    .onSubmit { search() }
                if !searchQuery.isEmpty {
                    Button(action: { searchQuery = ""; loadAll() }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(Color(hex: "888"))
                    }
                    .buttonStyle(.plain)
                }
                Button("Search") { search() }
                    .buttonStyle(MemmeButtonStyle(color: Color(hex: "6C5CE7")))
                    .disabled(searchQuery.isEmpty)
            }
            .padding()
            .background(Color(hex: "12121a"))

            Divider()

            // Memory List
            if isLoading {
                Spacer()
                ProgressView()
                    .tint(Color(hex: "6C5CE7"))
                Spacer()
            } else if memories.isEmpty {
                Spacer()
                VStack(spacing: 12) {
                    Image(systemName: "brain.head.profile")
                        .font(.system(size: 40))
                        .foregroundColor(Color(hex: "888"))
                    Text("No memories yet")
                        .foregroundColor(Color(hex: "888"))
                }
                Spacer()
            } else {
                ScrollView {
                    LazyVStack(spacing: 10) {
                        ForEach(memories) { memory in
                            MemoryCard(memory: memory) {
                                deleteMemory(memory.id)
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
        .onAppear { loadAll() }
        .refreshable { loadAll() }
    }

    private func loadAll() {
        Task {
            isLoading = true
            defer { isLoading = false }
            do {
                memories = try await api.listMemories()
            } catch {
                memories = []
                errorMessage = error.localizedDescription
            }
        }
    }

    private func search() {
        guard !searchQuery.isEmpty else { loadAll(); return }
        Task {
            isLoading = true
            defer { isLoading = false }
            do {
                memories = try await api.hybridSearch(query: searchQuery)
            } catch {
                memories = []
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
