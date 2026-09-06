import Foundation
import Masking

/// Where the real image sits inside a letterboxed square model input.
///
/// Vision's `.scaleFit` scales the source to fit the square input and centres
/// it, padding the two remaining sides. The model's mask therefore covers the
/// padding as well, and only the content region corresponds to something the
/// camera saw.
///
/// Letterboxing rather than stretching matters a great deal here: squashing a
/// portrait camera frame into a square distorts it far beyond anything in the
/// training set, and measured confidence collapses (0.008 versus 0.845 on the
/// same frame).
public struct LetterboxGeometry: Sendable, Equatable {

    public let contentRect: MaskRect

    public init(sourceWidth: Int, sourceHeight: Int, inputSize: Int, maskSize: Int) {
        precondition(sourceWidth > 0 && sourceHeight > 0, "source must have positive extent")

        let scale = Double(inputSize) / Double(max(sourceWidth, sourceHeight))
        let scaledWidth = Double(sourceWidth) * scale
        let scaledHeight = Double(sourceHeight) * scale

        // Padding is split evenly, then expressed in mask cells.
        let cellsPerInputPixel = Double(maskSize) / Double(inputSize)
        let x = Int((Double(inputSize) - scaledWidth) / 2 * cellsPerInputPixel)
        let y = Int((Double(inputSize) - scaledHeight) / 2 * cellsPerInputPixel)

        contentRect = MaskRect(
            x: x,
            y: y,
            width: max(1, maskSize - 2 * x),
            height: max(1, maskSize - 2 * y)
        )
    }
}
