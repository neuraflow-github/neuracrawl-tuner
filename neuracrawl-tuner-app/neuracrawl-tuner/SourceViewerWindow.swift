//
//  ContentView.swift
//  neuracrawl-tuner
//
//  Created by Julius Huck on 11.11.25.
//

import SwiftUI
import WebKit

struct SourceViewerWindow: View {
    let folderUrl: URL
    
    @State private var versions: [Version] = []
    @State private var selectedVersion: Version?
    @State private var currentDisplayPackIndex: Int = 0
    @State private var selectedTab: String = "Raw HTML"
    @State private var isLoadingWeb: Bool = false
    @State private var isLoadingData: Bool = false
    @State private var errorMessage: String?
    @State private var hasLoadedOnce: Bool = false
    
    func loadAllData() {
        isLoadingData = true
        errorMessage = nil
        
        Task {
            do {
                let loadedVersions = try await loadVersionsWithData()
                
                await MainActor.run {
                    versions = loadedVersions
                    
                    if selectedVersion == nil, let first = loadedVersions.first {
                        selectedVersion = first
                        currentDisplayPackIndex = 0
                    }
                    
                    isLoadingData = false
                    hasLoadedOnce = true
                }
            } catch {
                await MainActor.run {
                    errorMessage = "Failed to load data: \(error.localizedDescription)"
                    isLoadingData = false
                    hasLoadedOnce = true
                }
            }
        }
    }
    
    func loadVersionsWithData() async throws -> [Version] {
        let fileManager = FileManager.default
        
        guard let contents = try? fileManager.contentsOfDirectory(at: folderUrl, includingPropertiesForKeys: nil) else {
            throw NSError(domain: "SourceViewer", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to read directory contents."])
        }
        
        let versionDirs = contents
            .filter { $0.hasDirectoryPath }
            .filter { $0.lastPathComponent.starts(with: "v_") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        
        guard !versionDirs.isEmpty else {
            let foundDirs = contents
                .filter { $0.hasDirectoryPath }
                .map { $0.lastPathComponent }
                .joined(separator: ", ")
            
            throw NSError(domain: "SourceViewer", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "No version folders found. Expected folders starting with 'v_'.",
                NSLocalizedRecoverySuggestionErrorKey: foundDirs.isEmpty ? "Folder is empty." : "Found directories: \(foundDirs)"
            ])
        }
        
        var loadedVersions: [Version] = []
        
        for versionDir in versionDirs {
            let displayPacks = try loadDisplayPacks(from: versionDir)
            let version = Version(name: versionDir.lastPathComponent, displayPacks: displayPacks)
            loadedVersions.append(version)
        }
        
        return loadedVersions
    }
    
    func loadDisplayPacks(from versionUrl: URL) throws -> [DisplayPack] {
        let fileManager = FileManager.default
        
        guard let contents = try? fileManager.contentsOfDirectory(at: versionUrl, includingPropertiesForKeys: nil) else {
            return []
        }
        
        let numberDirs = contents
            .filter { $0.hasDirectoryPath }
            .filter { $0.lastPathComponent.first?.isNumber == true }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        
        var packs: [DisplayPack] = []
        
        for dir in numberDirs {
            let urlPath = dir.appendingPathComponent("05_url.txt")
            let rawHtmlPath = dir.appendingPathComponent("10_raw_html.html")
            let cleanedHTMLPath = dir.appendingPathComponent("20_cleaned_html.html")
            let rawMarkdownPath = dir.appendingPathComponent("30_raw_markdown.md")
            let feedbackPath = dir.appendingPathComponent("40_feedback.txt")
            
            guard let urlString = try? String(contentsOf: urlPath, encoding: .utf8),
                  let url = URL(string: urlString.trimmingCharacters(in: .whitespacesAndNewlines)),
                  let rawHtml = try? String(contentsOf: rawHtmlPath, encoding: .utf8),
                  let cleanedHTML = try? String(contentsOf: cleanedHTMLPath, encoding: .utf8),
                  let rawMarkdown = try? String(contentsOf: rawMarkdownPath, encoding: .utf8),
                  let feedback = try? String(contentsOf: feedbackPath, encoding: .utf8)
            else {
                print("Skipping incomplete data in \(dir.lastPathComponent)")
                continue
            }
            
            packs.append(DisplayPack(
                url: url,
                rawHtml: rawHtml,
                cleanedHTML: cleanedHTML,
                rawMarkdown: rawMarkdown,
                feedback: feedback,
                name: dir.lastPathComponent
            ))
        }
        
        return packs
    }

    var body: some View {
        Group {
            if !hasLoadedOnce {
                VStack {
                    ProgressView()
                    Text("Loading data...")
                        .font(.headline)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let error = errorMessage {
                VStack {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.system(size: 48))
                        .foregroundColor(.orange)
                    Text("Error")
                        .font(.headline)
                    Text(error)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                    Button("Retry") {
                        loadAllData()
                    }
                    .buttonStyle(.borderedProminent)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if versions.isEmpty {
                VStack(spacing: 20) {
                    Image(systemName: "folder.badge.questionmark")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary)
                    Text("No Versions Found")
                        .font(.headline)
                    Text("No version folders starting with 'v_' found in the selected directory.")
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                    Button("Reload") {
                        loadAllData()
                    }
                    .buttonStyle(.borderedProminent)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                GeometryReader { geometry in
                    HStack(spacing: 0) {
                        // Sidebar
                        VStack(spacing: 0) {
                            HStack {
                                Text("Versions")
                                    .font(.headline)
                                    .padding(.horizontal)
                                Spacer()
                                Button {
                                    loadAllData()
                                } label: {
                                    Image(systemName: "arrow.clockwise")
                                }
                                .buttonStyle(.borderless)
                                .disabled(isLoadingData)
                            }
                            .padding()
                            .background(Color(nsColor: .controlBackgroundColor))
                            
                            if isLoadingData {
                                VStack {
                                    ProgressView()
                                    Text("Reloading...")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                            } else {
                                List(versions, selection: $selectedVersion) { version in
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(version.name)
                                            .fontDesign(.monospaced)
                                            .font(.body)
                                        Text("\(version.displayPacks.count) items")
                                            .font(.caption)
                                            .foregroundColor(version.displayPacks.isEmpty ? .orange : .secondary)
                                    }
                                    .tag(version)
                                }
                                .listStyle(.sidebar)
                            }
                        }
                        .onChange(of: selectedVersion) { oldValue, newValue in
                            if newValue != nil {
                                currentDisplayPackIndex = 0
                            }
                        }
                        .frame(width: 200)
                        .background(Color(nsColor: .controlBackgroundColor))
                        
                        // Main content area
                        if let currentVersion = selectedVersion {
                            if currentVersion.displayPacks.isEmpty {
                                VStack(spacing: 20) {
                                    Image(systemName: "doc.text.magnifyingglass")
                                        .font(.system(size: 48))
                                        .foregroundColor(.secondary)
                                    Text("No Items in Version")
                                        .font(.headline)
                                    Text("Version '\(currentVersion.name)' contains no display packs.")
                                        .foregroundColor(.secondary)
                                        .multilineTextAlignment(.center)
                                        .padding(.horizontal)
                                }
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                            } else {
                                VStack(spacing: 0) {
                                    HSplitView {
                                        // Tabs and text editor
                                        TabView(selection: $selectedTab) {
                                            TextEditor(text: .constant(currentVersion.displayPacks[currentDisplayPackIndex].rawHtml))
                                                .font(.system(.body, design: .monospaced))
                                                .tabItem { Text("Raw HTML") }
                                                .tag("Raw HTML")
                                                .padding()
                                            
                                            TextEditor(text: .constant(currentVersion.displayPacks[currentDisplayPackIndex].cleanedHTML))
                                                .font(.system(.body, design: .monospaced))
                                                .tabItem { Text("Cleaned HTML") }
                                                .tag("Cleaned HTML")
                                                .padding()
                                            
                                            TextEditor(text: .constant(currentVersion.displayPacks[currentDisplayPackIndex].rawMarkdown))
                                                .font(.system(.body, design: .monospaced))
                                                .tabItem { Text("Raw Markdown") }
                                                .tag("Raw Markdown")
                                                .padding()
                                            
                                            TextEditor(text: .constant(currentVersion.displayPacks[currentDisplayPackIndex].feedback))
                                                .font(.system(.body, design: .monospaced))
                                                .tabItem { Text("Feedback") }
                                                .tag("Feedback")
                                                .padding()
                                        }
                                        .padding()
                                        .frame(minWidth: 300)
                                        
                                        // Web view
                                        ZStack {
                                            WebView(url: currentVersion.displayPacks[currentDisplayPackIndex].url, isLoading: $isLoadingWeb)
                                                .frame(minWidth: 200)
                                            
                                            if isLoadingWeb {
                                                Color.black.opacity(0.3)
                                                ProgressView()
                                            }
                                        }
                                    }
                                    
                                    HStack {
                                        Text(currentVersion.displayPacks[currentDisplayPackIndex].name)
                                            .fontDesign(.monospaced)
                                        
                                        Spacer()
                                        
                                        // Direct page selector
                                        Picker("", selection: $currentDisplayPackIndex) {
                                            ForEach(currentVersion.displayPacks.indices, id: \.self) { index in
                                                Text("\(index)").tag(index)
                                            }
                                        }
                                        .pickerStyle(.menu)
                                        .frame(width: 60)
                                        
                                        Button {
                                            currentDisplayPackIndex = currentDisplayPackIndex - 1
                                        } label: {
                                            Label("Back", systemImage: "arrow.backward")
                                        }
                                        .disabled(currentDisplayPackIndex == 0)
                                        .keyboardShortcut(.leftArrow, modifiers: [.command])
                                        
                                        Text("\(currentDisplayPackIndex)/\(currentVersion.displayPacks.count - 1)")
                                            .fontDesign(.monospaced)
                                            .padding(.horizontal, 8)
                                        
                                        Button {
                                            currentDisplayPackIndex = currentDisplayPackIndex + 1
                                        } label: {
                                            HStack {
                                                Text("Next")
                                                Image(systemName: "arrow.forward")
                                            }
                                        }
                                        .disabled(currentDisplayPackIndex == currentVersion.displayPacks.count - 1)
                                        .keyboardShortcut(.rightArrow, modifiers: [.command])
                                    }
                                    .padding()
                                    .background(Color(nsColor: .controlBackgroundColor))
                                }
                                .frame(minWidth: 300)
                            }
                        } else {
                            VStack {
                                Image(systemName: "sidebar.left")
                                    .font(.system(size: 48))
                                    .foregroundColor(.secondary)
                                Text("Select a Version")
                                    .font(.headline)
                                Text("Choose a version from the sidebar to view its contents.")
                                    .foregroundColor(.secondary)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal)
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
        .task {
            if !hasLoadedOnce {
                loadAllData()
            }
        }
    }
}

#Preview {
    SourceViewerWindow(folderUrl: URL(fileURLWithPath: "/Users/juliushuck/Projects/Research/neuracrawl-tuner/projects/siegburg/20_exclusion_css_selectors"))
        .frame(width: 960, height: 640)
}
