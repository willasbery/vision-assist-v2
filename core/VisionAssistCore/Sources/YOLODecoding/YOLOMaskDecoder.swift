import Accelerate
import Foundation
import Masking

/// Turns the two raw Core ML output tensors into a single binary mask.
///
/// Expects the `nms=True` export, whose detection tensor is already resolved:
///   detections  [rows, columns]  columns = 4 box + 1 confidence + 1 class + 32 coefficients
///   prototypes  [32, 160, 160]
///
/// Unused rows are zero-padded, so rows are filtered by confidence rather than
/// counted.
public struct YOLOMaskDecoder: Sendable {

    public struct Configuration: Sendable {
        public var confidenceThreshold: Float = 0.5
        public var maskThreshold: Float = 0.5
        public var coefficientCount = 32
        public var prototypeSize = 160
        public var inputSize = 640

        public init() {}
    }

    public let configuration: Configuration

    public init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
    }

    private var confidenceColumn: Int { 4 }
    private var coefficientOffset: Int { 6 }

    /// Parse every row whose confidence clears the threshold.
    public func detections(from raw: [Float], rows: Int, columns: Int) -> [YOLODetection] {
        precondition(raw.count >= rows * columns, "detection buffer is too small")
        precondition(columns >= coefficientOffset + configuration.coefficientCount,
                     "detection rows are too narrow")

        var parsed: [YOLODetection] = []
        for row in 0..<rows {
            let base = row * columns
            let confidence = raw[base + confidenceColumn]
            guard confidence > configuration.confidenceThreshold else { continue }

            parsed.append(YOLODetection(
                box: BoundingBox(x1: raw[base], y1: raw[base + 1],
                                 x2: raw[base + 2], y2: raw[base + 3]),
                confidence: confidence,
                coefficients: Array(raw[(base + coefficientOffset)
                                        ..< (base + coefficientOffset + configuration.coefficientCount)])
            ))
        }
        return parsed
    }

    /// Combine one detection's coefficients with the prototypes.
    public func mask(for detection: YOLODetection, prototypes: [Float]) -> Mask {
        let size = configuration.prototypeSize
        let pixels = size * size
        let count = configuration.coefficientCount
        precondition(prototypes.count >= pixels * count, "prototype buffer is too small")

        // (1 x 32) * (32 x 25600) -> (1 x 25600)
        var logits = [Float](repeating: 0, count: pixels)
        detection.coefficients.withUnsafeBufferPointer { coefficients in
            prototypes.withUnsafeBufferPointer { prototypes in
                vDSP_mmul(coefficients.baseAddress!, 1,
                          prototypes.baseAddress!, 1,
                          &logits, 1,
                          1, vDSP_Length(pixels), vDSP_Length(count))
            }
        }

        // sigmoid(x) > t  <=>  x > ln(t / (1 - t))
        let t = configuration.maskThreshold
        let cutoff = log(t / (1 - t))

        // The box is in input space; the prototypes are 1/scale of that.
        let scale = Float(configuration.inputSize / size)
        let minX = max(0, Int(detection.box.x1 / scale))
        let minY = max(0, Int(detection.box.y1 / scale))
        let maxX = min(size, Int((detection.box.x2 / scale).rounded(.up)))
        let maxY = min(size, Int((detection.box.y2 / scale).rounded(.up)))

        var values = [UInt8](repeating: 0, count: pixels)
        guard minX < maxX, minY < maxY else {
            return Mask(width: size, height: size, values: values)
        }

        for y in minY..<maxY {
            let row = y * size
            for x in minX..<maxX where logits[row + x] > cutoff {
                values[row + x] = 1
            }
        }

        return Mask(width: size, height: size, values: values)
    }

    /// The mask for the largest detection, or nil if nothing clears the
    /// threshold. Largest rather than most confident, because that is what the
    /// pipeline downstream cares about.
    public func decode(detections raw: [Float], rows: Int, columns: Int,
                       prototypes: [Float]) -> Mask? {
        let candidates = detections(from: raw, rows: rows, columns: columns)
        guard let largest = candidates.max(by: { $0.box.area < $1.box.area }) else {
            return nil
        }
        return mask(for: largest, prototypes: prototypes)
    }
}
