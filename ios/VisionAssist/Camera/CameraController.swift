// AVCaptureSession is thread-safe for start/stop but is not annotated
// Sendable, so passing it to the session queue warns. The @preconcurrency
// import is the supported way to silence a framework annotation gap.
@preconcurrency import AVFoundation

/// Owns the capture session and its lifecycle.
///
/// The session is configured and started off the main thread — `startRunning()`
/// blocks, and doing it on the main thread stutters the UI on launch.
@MainActor
final class CameraController: ObservableObject {

    enum State: Equatable {
        case idle
        case running
        case denied
        case failed(String)
    }

    @Published private(set) var state: State = .idle

    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "com.willasbery.visionassist.session")
    private var isConfigured = false

    func start() async {
        guard await isAuthorised() else {
            state = .denied
            return
        }

        if !isConfigured {
            do {
                try configure()
                isConfigured = true
            } catch {
                state = .failed(String(describing: error))
                return
            }
        }

        let session = self.session
        queue.async {
            if !session.isRunning { session.startRunning() }
        }
        state = .running
    }

    func stop() {
        let session = self.session
        queue.async {
            if session.isRunning { session.stopRunning() }
        }
        state = .idle
    }

    private func isAuthorised() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
        default:
            return false
        }
    }

    private enum ConfigurationError: Error {
        case noCamera
        case cannotAddInput
    }

    private func configure() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .hd1280x720

        guard let camera = AVCaptureDevice.default(
            .builtInWideAngleCamera, for: .video, position: .back
        ) else {
            throw ConfigurationError.noCamera
        }

        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input) else {
            throw ConfigurationError.cannotAddInput
        }
        session.addInput(input)
    }
}
