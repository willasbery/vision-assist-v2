import CoreVideo
import Masking

/// Produces a pavement mask from a camera frame.
///
/// The only contract between the model and the rest of the app. Swapping
/// YOLO for a semantic segmentation model means writing another conformance
/// and nothing else.
protocol PavementSegmenter {
    /// The mask for this frame, or nil if no pavement was found.
    /// Coordinates are the mask's own space, not the frame's.
    func mask(from frame: CVPixelBuffer) throws -> Mask?
}
