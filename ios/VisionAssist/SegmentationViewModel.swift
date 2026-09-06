import CoreVideo
import Foundation
import Masking

/// Runs segmentation on incoming frames and publishes the latest result.
@MainActor
final class SegmentationViewModel: ObservableObject {

    @Published private(set) var mask: Mask?
    @Published private(set) var milliseconds: Double = 0
    @Published private(set) var failure: String?

    /// Assigned once during init and only read afterwards, and `FrameStream`
    /// guarantees one `process` call at a time — so reading it off the main
    /// actor is safe, which the compiler cannot infer on its own.
    nonisolated(unsafe) private let segmenter: PavementSegmenter?

    init() {
        do {
            segmenter = try YOLOSegmenter()
        } catch {
            segmenter = nil
            failure = String(describing: error)
        }
    }

    /// Called on the video queue.
    nonisolated func process(_ frame: CVPixelBuffer) {
        guard let segmenter else { return }

        let start = CFAbsoluteTimeGetCurrent()
        let result = try? segmenter.mask(from: frame)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

        Task { @MainActor in
            self.mask = result
            // Smoothed, so the number is readable rather than flickering.
            self.milliseconds = self.milliseconds == 0
                ? elapsed
                : self.milliseconds * 0.9 + elapsed * 0.1
        }
    }
}
