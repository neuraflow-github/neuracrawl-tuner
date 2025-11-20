//
//  DisplayPack.swift
//  neuracrawl-tuner
//
//  Created by Julius Huck on 18.11.25.
//

import Foundation

struct DisplayPack: Identifiable, Hashable, Equatable {
    let id = UUID()
    let url: URL
    let rawHTML: String
    let cleanedHTML: String
    let rawMarkdown: String
    let cleanedMarkdown: String
    let feedback: String
    let name: String
    
    static func == (lhs: DisplayPack, rhs: DisplayPack) -> Bool {
        return lhs.id == rhs.id
    }
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

struct Version: Identifiable, Hashable, Equatable {
    let id: String
    let name: String
    let displayPacks: [DisplayPack]
    
    init(name: String, displayPacks: [DisplayPack]) {
        self.id = name
        self.name = name
        self.displayPacks = displayPacks
    }
    
    static func == (lhs: Version, rhs: Version) -> Bool {
        return lhs.id == rhs.id
    }
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}
