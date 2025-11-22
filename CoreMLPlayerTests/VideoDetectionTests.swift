import XCTest
@testable import CoreML_Player

final class VideoDetectionTests: XCTestCase {
    func testRepeatIntervalRespectsFrameRateAndLatency() {
        let sut = VideoDetection()
        sut.videoInfo = (isPlayable: true, frameRate: 60, duration: .zero, size: .zero)
        let interval = sut.getRepeatInterval(false)
        XCTAssertEqual(interval, 1 / 60)
    }

    func testRepeatIntervalTrimsLastDetectionTime() {
        let sut = VideoDetection()
        sut.videoInfo = (isPlayable: true, frameRate: 30, duration: .zero, size: .zero)
        sut.lastDetectionTime = 0.01
        let interval = sut.getRepeatInterval()
        XCTAssertLessThan(interval, 1 / 30)
        XCTAssertGreaterThan(interval, 0)
    }
}
