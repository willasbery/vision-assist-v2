import Foundation

/// A box in the model's 640x640 input space.
public struct BoundingBox: Sendable, Equatable {
    public let x1: Float
    public let y1: Float
    public let x2: Float
    public let y2: Float

    public init(x1: Float, y1: Float, x2: Float, y2: Float) {
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    }

    public var area: Float {
        max(0, x2 - x1) * max(0, y2 - y1)
    }
}

/// One row of the NMS-resolved detection tensor.
public struct YOLODetection: Sendable, Equatable {
    public let box: BoundingBox
    public let confidence: Float

    /// The 32 mask coefficients, combined with the prototypes to form a mask.
    public let coefficients: [Float]

    public init(box: BoundingBox, confidence: Float, coefficients: [Float]) {
        self.box = box
        self.confidence = confidence
        self.coefficients = coefficients
    }
}
