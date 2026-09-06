import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraController()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            switch camera.state {
            case .running:
                CameraPreviewView(session: camera.session)
                    .ignoresSafeArea()
            case .denied:
                message("Camera access is off. Turn it on in Settings to use Vision Assist.")
            case .failed(let reason):
                message("The camera could not start.\n\(reason)")
            case .idle:
                ProgressView()
            }
        }
        .task { await camera.start() }
        .onDisappear { camera.stop() }
    }

    private func message(_ text: String) -> some View {
        Text(text)
            .font(.headline)
            .foregroundStyle(.white)
            .multilineTextAlignment(.center)
            .padding()
    }
}
