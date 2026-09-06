import XCTest
@testable import Masking

final class MaskTests: XCTestCase {
    func testSubscriptReadsRowMajor() {
        // 3 wide, 2 tall, with only the middle of the bottom row set.
        let mask = Mask(width: 3, height: 2, values: [0, 0, 0,
                                                      0, 1, 0])
        XCTAssertTrue(mask[1, 1])
        XCTAssertFalse(mask[0, 0])
        XCTAssertFalse(mask[1, 0])
    }

    func testFilledCount() {
        XCTAssertEqual(Mask(width: 2, height: 2, values: [1, 0, 1, 1]).filledCount, 3)
        XCTAssertTrue(Mask(width: 2, height: 2, values: [0, 0, 0, 0]).isEmpty)
    }
}
