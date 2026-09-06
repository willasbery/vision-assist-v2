import XCTest
@testable import Masking

final class MaskCropTests: XCTestCase {
    func testCropExtractsTheRequestedRegion() {
        // 4x3, with a distinctive middle column.
        let mask = Mask(width: 4, height: 3, values: [0, 1, 0, 0,
                                                      0, 1, 1, 0,
                                                      0, 0, 1, 0])

        let cropped = mask.cropped(to: MaskRect(x: 1, y: 0, width: 2, height: 3))

        XCTAssertEqual(cropped.width, 2)
        XCTAssertEqual(cropped.height, 3)
        XCTAssertEqual(cropped.values, [1, 0,
                                        1, 1,
                                        0, 1])
    }

    func testCroppingToTheFullExtentIsANoOp() {
        let mask = Mask(width: 2, height: 2, values: [1, 0, 0, 1])
        XCTAssertEqual(mask.cropped(to: MaskRect(x: 0, y: 0, width: 2, height: 2)), mask)
    }
}
