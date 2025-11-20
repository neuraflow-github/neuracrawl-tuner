import SwiftUI
import WebKit

public struct WebView: NSViewRepresentable {
    private let url: URL?
    private let configuration: (WKWebView) -> Void
    @Binding var isLoading: Bool
    
    public init(url: URL, isLoading: Binding<Bool> = .constant(false)) {
        self.url = url
        self._isLoading = isLoading
        self.configuration = { _ in }
    }
    
    public init(url: URL? = nil, isLoading: Binding<Bool> = .constant(false), configuration: @escaping (WKWebView) -> Void = { _ in }) {
        self.url = url
        self._isLoading = isLoading
        self.configuration = configuration
    }
    
    public func makeNSView(context: Context) -> WKWebView {
        let view = WKWebView()
        view.navigationDelegate = context.coordinator
        configuration(view)
        context.coordinator.isLoadingBinding = _isLoading
        
        guard let url = url else { return view }
        context.coordinator.currentUrl = url
        DispatchQueue.main.async {
            self.isLoading = true
        }
        view.load(URLRequest(url: url))
        return view
    }
    
    public func updateNSView(_ nsView: WKWebView, context: Context) {
        guard let url = url else { return }
        context.coordinator.isLoadingBinding = _isLoading
        
        if context.coordinator.currentUrl != url {
            context.coordinator.currentUrl = url
            DispatchQueue.main.async {
                self.isLoading = true
            }
            nsView.load(URLRequest(url: url))
        }
    }
    
    public func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    public class Coordinator: NSObject, WKNavigationDelegate {
        var currentUrl: URL?
        var isLoadingBinding: Binding<Bool>?
        
        public func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
            DispatchQueue.main.async {
                self.isLoadingBinding?.wrappedValue = true
            }
        }
        
        public func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            DispatchQueue.main.async {
                self.isLoadingBinding?.wrappedValue = false
            }
        }
        
        public func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            DispatchQueue.main.async {
                self.isLoadingBinding?.wrappedValue = false
            }
        }
        
        public func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            DispatchQueue.main.async {
                self.isLoadingBinding?.wrappedValue = false
            }
        }
    }
}
