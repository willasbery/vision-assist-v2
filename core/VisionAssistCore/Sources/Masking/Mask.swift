import Foundation

/// A binary mask in its own coordinate space.
///
/// This is the entire contract between the segmentation model and everything
/// downstream. Nothing here knows how the mask was produced, which is what
/// makes swapping the model a contained change.
public struct Mask: Sendable, Equatable {
    public let width: Int
    public let height: Int

    /// Row-major, one byte per cell, 0 or 1.
    public let values: [UInt8]

    public init(width: Int, height: Int, values: [UInt8]) {
        precondition(width > 0 && height > 0, "mask must have positive extent")
        precondition(values.count == width * height,
                     "expected \(width * height) values, got \(values.count)")
        self.width = width
        self.height = height
        self.values = values
    }

    public subscript(x: Int, y: Int) -> Bool {
        precondition(x >= 0 && x < width && y >= 0 && y < height, "out of bounds")
        return values[y * width + x] != 0
    }

    /// How many cells are set. Used to pick between candidate masks.
    public var filledCount: Int {
        values.reduce(into: 0) { $0 += Int($1) }
    }

    public var isEmpty: Bool { filledCount == 0 }

    /// The sub-region `rect` as a mask in its own right.
    ///
    /// Used to discard letterbox padding, so the returned mask lines up with
    /// the source image rather than with the model's square input.
    public func cropped(to rect: MaskRect) -> Mask {
        precondition(rect.x >= 0 && rect.y >= 0, "crop origin must be non-negative")
        precondition(rect.x + rect.width <= width && rect.y + rect.height <= height,
                     "crop must lie within the mask")

        var cropped = [UInt8]()
        cropped.reserveCapacity(rect.width * rect.height)
        for row in rect.y..<(rect.y + rect.height) {
            let start = row * width + rect.x
            cropped.append(contentsOf: values[start..<(start + rect.width)])
        }
        return Mask(width: rect.width, height: rect.height, values: cropped)
    }
}
