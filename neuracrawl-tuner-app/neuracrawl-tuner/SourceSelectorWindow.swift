//
//  SourceSelectorWindow.swift
//  neuracrawl-tuner
//
//  Created by Julius Huck on 20.11.25.
//

import SwiftUI
import WebKit

struct SourceSelectorWindow: View {
    @Environment(\.openWindow) private var openWindow
    
    func openFolderPicker() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.prompt = "Select Folder"
        
        panel.begin { response in
            if response == .OK, let url = panel.url {
                openWindow(id: "project", value: url)
            }
        }
    }
    
    var body: some View {
        VStack {
            if let appIcon = NSApp.applicationIconImage {
                Image(nsImage: appIcon)
                    .resizable()
                    .frame(width: 128, height: 128)
                    .padding()
            }
            Button {
                openFolderPicker()
            } label: {
                Label("Open Source", systemImage: "folder.badge.plus")
                    .font(.title2)
                    .fontWeight(.semibold)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 12)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .padding()
    }
}

#Preview {
    SourceSelectorWindow().frame(width: 960, height: 640)
}
