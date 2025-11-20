//
//  neuracrawl_tunerApp.swift
//  neuracrawl-tuner
//
//  Created by Julius Huck on 11.11.25.
//

import SwiftUI

@main
struct neuracrawl_tunerApp: App {
    var body: some Scene {
        WindowGroup {
            SourceSelectorWindow().navigationTitle("neuracrawl tuner").fixedSize()
        }
        .windowResizability(.contentSize)
        
        WindowGroup(id: "project", for: URL.self) { $folderURL in
            if let url = folderURL {
                SourceViewerWindow(folderUrl: url).navigationTitle(url.lastPathComponent)
            }
        }
        .defaultSize(width: 960, height: 640)
    }
}
