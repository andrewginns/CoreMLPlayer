import XCTest
@testable import CoreML_Player

final class VideoDetectionTests: XCTestCase {
    func testRepeatIntervalRespectsFrameRateAndLatency() {
        let sut = VideoDetection()
        sut.setVideoInfoForTesting((isPlayable: true, frameRate: 60, duration: .zero, size: .zero))
        let interval = sut.getRepeatInterval(false)
        XCTAssertEqual(interval, 1 / 60)
    }

    func testRepeatIntervalTrimsLastDetectionTime() {
        let sut = VideoDetection()
        sut.setVideoInfoForTesting((isPlayable: true, frameRate: 30, duration: .zero, size: .zero))
        sut.setLastDetectionTimeForTesting(0.01)
        let interval = sut.getRepeatInterval()
        XCTAssertLessThan(interval, 1 / 30)
        XCTAssertGreaterThan(interval, 0)
    }

    func testRepeatIntervalClampsToDetectionBudget() {
        let sut = VideoDetection()
        sut.setVideoInfoForTesting((isPlayable: true, frameRate: 30, duration: .zero, size: .zero))
        sut.setLastDetectionTimeForTesting(1.0) // detection took 1s previously

        let interval = sut.getRepeatInterval()
        XCTAssertGreaterThanOrEqual(interval, 0.5)
        XCTAssertLessThanOrEqual(interval, 1.1)
    }

    func testRepeatIntervalClampsToMinimumWhenLastDetectionIsHigh() {
        let sut = VideoDetection()
        sut.setVideoInfoForTesting((isPlayable: true, frameRate: 30, duration: .zero, size: .zero))
        sut.setLastDetectionTimeForTesting(1.0) // 1s > frame interval

        let interval = sut.getRepeatInterval()
        XCTAssertEqual(interval, 0.5, accuracy: 0.001) // clamped to half the prior detection time
    }

    func testDisappearingClearsDetectionStats() {
        let sut = VideoDetection()
        DetectionStats.shared.addMultiple([Stats(key: "FPS", value: "10")])
        XCTAssertFalse(DetectionStats.shared.items.isEmpty)

        let expectation = expectation(description: "Detection stats cleared after disappearing")
        sut.disappearing()

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            if DetectionStats.shared.items.isEmpty {
                expectation.fulfill()
            } else {
                XCTFail("Detection stats were not cleared")
            }
        }

        wait(for: [expectation], timeout: 0.5)
    }
}
