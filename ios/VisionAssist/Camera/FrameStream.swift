@preconcurrency import AVFoundation
import Foundation
import os

/// Receives camera frames and forwards them one at a time.
///
/// Anything arriving while the handler is still working is dropped. The
/// handler runs on the video queue, not the main thread.
final class FrameStream: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    private let handler: (CVPixelBuffer) -> Void
    private let isBusy = OSAllocatedUnfairLock(initialState: false)

    init(handler: @escaping (CVPixelBuffer) -> Void) {
        self.handler = handler
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let shouldRun = isBusy.withLock { busy -> Bool in
            if busy { return false }
            busy = true
            return true
        }
        guard shouldRun else { return }

        handler(buffer)
        isBusy.withLock { $0 = false }
    }
}
