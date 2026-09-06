import XCTest
@testable import Masking
@testable import YOLODecoding

/// Vision's `.scaleFit` letterboxes: the source is scaled to fit the square
/// input and centred, leaving padding on two sides. Only the content region of
/// the resulting mask corresponds to anything the camera actually saw.
final class LetterboxGeometryTests: XCTestCase {

    func testPortraitSourceIsPaddedLeftAndRight() {
        // 720x1280 scaled to fit 640 -> 360x640, so 140px of padding each side.
        // In the 160-wide mask that is 35 either side, leaving 90 columns.
        let geometry = LetterboxGeometry(sourceWidth: 720, sourceHeight: 1280,
                                         inputSize: 640, maskSize: 160)

        XCTAssertEqual(geometry.contentRect, MaskRect(x: 35, y: 0, width: 90, height: 160))
    }

    func testLandscapeSourceIsPaddedTopAndBottom() {
        let geometry = LetterboxGeometry(sourceWidth: 1280, sourceHeight: 720,
                                         inputSize: 640, maskSize: 160)

        XCTAssertEqual(geometry.contentRect, MaskRect(x: 0, y: 35, width: 160, height: 90))
    }

    func testSquareSourceNeedsNoPadding() {
        let geometry = LetterboxGeometry(sourceWidth: 500, sourceHeight: 500,
                                         inputSize: 640, maskSize: 160)

        XCTAssertEqual(geometry.contentRect, MaskRect(x: 0, y: 0, width: 160, height: 160))
    }
}
