import SwiftUI

@main
struct MemMeDemoApp: App {
    @StateObject private var api = MemmeAPI()

    var body: some Scene {
        #if os(iOS)
        WindowGroup {
            TabView {
                NavigationStack {
                    VStack(spacing: 0) {
                        LLMConfigBar()
                            .padding(.horizontal, 12)
                            .padding(.top, 8)
                        ChatInputView()
                    }
                    .navigationTitle("Chat")
                    .navigationBarTitleDisplayMode(.inline)
                }
                .tabItem { Label(api.l10n.lang == .zh ? "聊天" : "Chat", systemImage: "bubble.left.fill") }

                NavigationStack {
                    SearchView()
                        .navigationTitle(api.l10n.search)
                        .navigationBarTitleDisplayMode(.inline)
                }
                .tabItem { Label(api.l10n.search, systemImage: "magnifyingglass") }

                NavigationStack {
                    StatsView()
                        .navigationTitle(api.l10n.stats)
                        .navigationBarTitleDisplayMode(.inline)
                }
                .tabItem { Label(api.l10n.stats, systemImage: "chart.bar.fill") }

                NavigationStack {
                    BenchmarkView()
                }
                .tabItem { Label("Bench", systemImage: "speedometer") }
            }
            .environmentObject(api)
            .preferredColorScheme(.dark)
            .tint(Color(hex: "6C5CE7"))
        }
        #else
        WindowGroup {
            NavigationSplitView {
                VStack(spacing: 0) {
                    HStack {
                        Text("MemMe")
                            .font(.title2.bold())
                            .foregroundColor(Color(hex: "c8b6ff"))
                        Spacer()
                        Button(action: {
                            api.language = api.language == .en ? .zh : .en
                        }) {
                            Text(api.language.label)
                                .font(.system(size: 13, weight: .medium))
                                .foregroundColor(.white.opacity(0.7))
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color(hex: "6C5CE7").opacity(0.3))
                                .cornerRadius(6)
                        }
                        .buttonStyle(.plain)
                        .help(api.language == .en ? "切换到中文" : "Switch to English")
                    }
                    .padding(.horizontal, 14)
                    .padding(.top, 12)
                    .padding(.bottom, 8)

                    LLMConfigBar()
                        .padding(.horizontal, 12)
                        .padding(.bottom, 8)

                    ChatInputView()
                }
                .frame(minWidth: 320)
                .background(Color(hex: "12121a"))
                .navigationSplitViewColumnWidth(min: 320, ideal: 380, max: 500)
            } detail: {
                TabView {
                    SearchView()
                        .tabItem { Label(api.l10n.search, systemImage: "magnifyingglass") }
                    StatsView()
                        .tabItem { Label(api.l10n.stats, systemImage: "chart.bar.fill") }
                    BenchmarkView()
                        .tabItem { Label("Bench", systemImage: "speedometer") }
                }
                .background(Color(hex: "0a0a0f"))
            }
            .environmentObject(api)
            .preferredColorScheme(.dark)
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1100, height: 700)
        #endif
    }
}
