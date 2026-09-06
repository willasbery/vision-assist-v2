import XCTest
@testable import Masking
@testable import YOLODecoding

/// Verifies the decoder against a real Core ML inference captured by
/// `tools/dump_decoder_fixture.py`. The expected mask in that fixture was
/// itself checked against ultralytics' own output at IoU 0.9941.
final class YOLOMaskDecoderTests: XCTestCase {

    struct Manifest: Decodable {
        struct Detections: Decodable { let rows: Int; let columns: Int }
        struct Prototypes: Decodable { let count: Int; let size: Int }
        struct ExpectedMask: Decodable { let size: Int; let filled: Int }
        struct Box: Decodable { let x1: Float; let y1: Float; let x2: Float; let y2: Float }

        let detections: Detections
        let prototypes: Prototypes
        let expectedMask: ExpectedMask
        let expectedBox: Box
        let expectedConfidence: Float

        enum CodingKeys: String, CodingKey {
            case detections, prototypes
            case expectedMask = "expected_mask"
            case expectedBox = "expected_box"
            case expectedConfidence = "expected_confidence"
        }
    }

    /// core/VisionAssistCore/Tests/YOLODecodingTests/<file> -> repo root
    static let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // YOLODecodingTests
        .deletingLastPathComponent()   // Tests
        .deletingLastPathComponent()   // VisionAssistCore
        .deletingLastPathComponent()   // core
        .deletingLastPathComponent()   // repo root
        .appendingPathComponent("fixtures/decoder")

    private func floats(_ name: String) throws -> [Float] {
        let data = try Data(contentsOf: Self.fixtures.appendingPathComponent(name))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func manifest() throws -> Manifest {
        let data = try Data(contentsOf: Self.fixtures.appendingPathComponent("manifest.json"))
        return try JSONDecoder().decode(Manifest.self, from: data)
    }

    func testParsesTheDetectionRow() throws {
        let manifest = try manifest()
        let decoder = YOLOMaskDecoder()

        let detections = decoder.detections(
            from: try floats("detections.bin"),
            rows: manifest.detections.rows,
            columns: manifest.detections.columns
        )

        XCTAssertEqual(detections.count, 1, "fixture has one detection above threshold")
        let detection = try XCTUnwrap(detections.first)
        XCTAssertEqual(detection.confidence, manifest.expectedConfidence, accuracy: 1e-4)
        XCTAssertEqual(detection.box.x1, manifest.expectedBox.x1, accuracy: 1e-3)
        XCTAssertEqual(detection.box.y2, manifest.expectedBox.y2, accuracy: 1e-3)
        XCTAssertEqual(detection.coefficients.count, 32)
    }

    func testProducesTheExpectedMask() throws {
        let manifest = try manifest()
        let decoder = YOLOMaskDecoder()

        let mask = try XCTUnwrap(decoder.decode(
            detections: try floats("detections.bin"),
            rows: manifest.detections.rows,
            columns: manifest.detections.columns,
            prototypes: try floats("prototypes.bin")
        ))

        XCTAssertEqual(mask.width, manifest.expectedMask.size)
        XCTAssertEqual(mask.height, manifest.expectedMask.size)
        XCTAssertEqual(mask.filledCount, manifest.expectedMask.filled)

        let expected = try Data(contentsOf: Self.fixtures.appendingPathComponent("expected_mask.bin"))
        XCTAssertEqual(mask.values, Array(expected), "mask differs from the captured inference")
    }

    func testReturnsNilWhenNothingClearsTheThreshold() throws {
        let manifest = try manifest()
        let decoder = YOLOMaskDecoder()

        let mask = decoder.decode(
            detections: [Float](repeating: 0, count: manifest.detections.rows * manifest.detections.columns),
            rows: manifest.detections.rows,
            columns: manifest.detections.columns,
            prototypes: try floats("prototypes.bin")
        )

        XCTAssertNil(mask)
    }
}
