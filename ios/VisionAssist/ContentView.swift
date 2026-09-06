import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraController()
    @StateObject private var segmentation = SegmentationViewModel()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            switch camera.state {
            case .running:
                CameraPreviewView(session: camera.session)
                    .overlay(MaskOverlayView(mask: segmentation.mask))
                    .overlay(alignment: .top) { readout }
                    .ignoresSafeArea()
            case .denied:
                message("Camera access is off. Turn it on in Settings to use Vision Assist.")
            case .failed(let reason):
                message("The camera could not start.\n\(reason)")
            case .idle:
                ProgressView()
            }
        }
        .task {
            camera.onFrame = { [weak segmentation] frame in
                segmentation?.process(frame)
            }
            await camera.start()
        }
        .onDisappear { camera.stop() }
    }

    private var readout: some View {
        Text(segmentation.failure ?? String(format: "%.0f ms", segmentation.milliseconds))
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(.white)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.black.opacity(0.6), in: Capsule())
            .padding(.top, 60)
    }

    private func message(_ text: String) -> some View {
        Text(text)
            .font(.headline)
            .foregroundStyle(.white)
            .multilineTextAlignment(.center)
            .padding()
    }
}
