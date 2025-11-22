import XCTest
import Vision
import CoreML
import CoreVideo
import CoreGraphics
@testable import CoreML_Player

/// Tests covering timing, metrics, and scheduling behaviors in VideoDetection.
final class VideoDetectionMetricsTests: XCTestCase {
    // MARK: - Performance / scheduling / stats
    func testDetectionLatencyFeedsRepeatIntervalAndStats() throws {
        let sut = try makeStubVideoDetection(detectionTimeMs: 12, detectionFPS: "84", objects: 3)
        sut.setVideoInfoForTesting((isPlayable: true, frameRate: 30, duration: .zero, size: CGSize(width: 1920, height: 1080)))
        DetectionStats.shared.items = []

        // Warm-up run (stats intentionally skipped)
        sut.detectObjectsInFrame()

        let exp = expectation(description: "second detection completes")
        sut.detectObjectsInFrame { exp.fulfill() }
        wait(for: [exp], timeout: 1.0)

        // Stats should reflect the stubbed detection result
        let detTime = DetectionStats.shared.items.first(where: { $0.key == "Det. Time" })?.value
        XCTAssertEqual(detTime, "12 ms")
        let detObjects = DetectionStats.shared.items.first(where: { $0.key == "Det. Objects" })?.value
        XCTAssertEqual(detObjects, "3")

        // Repeat interval should subtract last detection time but stay above the clamp (0.02s)
        let expected = (1.0 / 30.0) - 0.012
        XCTAssertEqual(sut.getRepeatInterval(), expected, accuracy: 0.002)
    }

    func testFrameObjectsAndStatsClearOnDisappearing() throws {
        let sut = try makeStubVideoDetection()
        sut.setVideoInfoForTesting((isPlayable: true, frameRate: 24, duration: .zero, size: CGSize(width: 640, height: 360)))
        DetectionStats.shared.items = []

        // Warm-up
        sut.detectObjectsInFrame()
        let exp = expectation(description: "post-warmup detection completes")
        sut.detectObjectsInFrame { exp.fulfill() }
        wait(for: [exp], timeout: 1.0)
        XCTAssertFalse(sut.frameObjects.isEmpty)
        XCTAssertFalse(DetectionStats.shared.items.isEmpty)

        sut.disappearing()

        XCTAssertTrue(sut.frameObjects.isEmpty)

        let cleared = expectation(description: "stats cleared")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            if DetectionStats.shared.items.isEmpty {
                cleared.fulfill()
            }
        }
        wait(for: [cleared], timeout: 0.5)
    }

    func testWarmupFrameExcludedFromStatsAndChart() throws {
        let sut = try makeStubVideoDetection()
        sut.setVideoInfoForTesting((isPlayable: true, frameRate: 30, duration: .zero, size: CGSize(width: 1280, height: 720)))
        DetectionStats.shared.items = []

        // First call should be warm-up and not push stats.
        sut.detectObjectsInFrame()
        XCTAssertTrue(DetectionStats.shared.items.isEmpty)

        // Second call should push stats.
        let exp = expectation(description: "second detection")
        sut.detectObjectsInFrame { exp.fulfill() }
        wait(for: [exp], timeout: 1.0)
        XCTAssertFalse(DetectionStats.shared.items.isEmpty)
        XCTAssertTrue(sut.metricsForTesting().warmupCompleted)
    }

    func testDroppedFramesCountedAfterWarmup() throws {
        final class NilFirstPixelBufferVD: StubVideoDetection {
            private var first = true
            override func getPixelBuffer() -> CVPixelBuffer? {
                if first {
                    first = false
                    return nil
                }
                return super.getPixelBuffer()
            }
        }

        let sut = NilFirstPixelBufferVD(stubDetectionTimeMs: 10, stubDetectionFPS: "100", stubObjects: 1)
        sut.setModel(try VNCoreMLModel(for: MLModel(contentsOf: compiledModelURL())))
        sut.setVideoInfoForTesting((isPlayable: true, frameRate: 30, duration: .zero, size: CGSize(width: 320, height: 240)))
        DetectionStats.shared.items = []

        sut.detectObjectsInFrame() // warmup + dropped frame, should not count
        XCTAssertEqual(sut.metricsForTesting().droppedFrames, 1)
        XCTAssertTrue(DetectionStats.shared.items.isEmpty)

        // Second call becomes warm-up (because first bailed early)
        sut.detectObjectsInFrame()
        let exp = expectation(description: "post-warmup detection")
        sut.detectObjectsInFrame { exp.fulfill() }
        wait(for: [exp], timeout: 1.0)

        let dropStat = DetectionStats.shared.items.first(where: { $0.key == "Dropped Frames" })
        XCTAssertEqual(dropStat?.value, "1")
    }

    func testDetectionLatencyWithinBudgetHelper() {
        let sut = VideoDetection()
        sut.setLastDetectionTimeForTesting(0.030) // 30ms
        XCTAssertTrue(sut.isWithinLatencyBudget())
        sut.setLastDetectionTimeForTesting(0.080)
        XCTAssertFalse(sut.isWithinLatencyBudget(budgetMs: 50))
    }

    // MARK: - Helpers
    private func compiledModelURL() throws -> URL {
        let bundle = Bundle(for: type(of: self))
        if let compiled = bundle.url(forResource: "YOLOv3Tiny", withExtension: "mlmodelc") {
            return compiled
        }
        guard let raw = bundle.url(forResource: "YOLOv3Tiny", withExtension: "mlmodel") else {
            XCTFail("Missing YOLOv3Tiny model in test bundle")
            throw XCTSkip("Model fixture unavailable")
        }
        return try MLModel.compileModel(at: raw)
    }

    private func makeStubVideoDetection(
        detectionTimeMs: Double = 12,
        detectionFPS: String = "90",
        objects: Int = 2
    ) throws -> StubVideoDetection {
        let model = try VNCoreMLModel(for: MLModel(contentsOf: compiledModelURL()))
        let sut = StubVideoDetection(
            stubDetectionTimeMs: detectionTimeMs,
            stubDetectionFPS: detectionFPS,
            stubObjects: objects
        )
        sut.setModel(model)
        return sut
    }
}
