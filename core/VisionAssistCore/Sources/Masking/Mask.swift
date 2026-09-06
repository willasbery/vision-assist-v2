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
}
