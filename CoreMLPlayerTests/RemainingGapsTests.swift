import XCTest
import CoreML
import CoreGraphics
import Vision
import AVFoundation
@testable import CoreML_Player

/// Tests that cover the remaining items in TEST_PLAN.md without adding new fixtures.
final class RemainingGapsTests: XCTestCase {
    // MARK: - Helpers
    private func rawModelURL() throws -> URL {
        let bundle = Bundle(for: type(of: self))
        if let raw = bundle.url(forResource: "YOLOv3Tiny", withExtension: "mlmodel") {
            return raw
        }
        throw XCTSkip("Raw YOLOv3Tiny.mlmodel not bundled; optimization tests skipped.")
    }

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

    // MARK: - Performance / scheduling / stats
    func testDetectionLatencyFeedsRepeatIntervalAndStats() throws {
        let sut = try makeStubVideoDetection(detectionTimeMs: 12, detectionFPS: "84", objects: 3)
        sut.setVideoInfoForTesting((isPlayable: true, frameRate: 30, duration: .zero, size: CGSize(width: 1920, height: 1080)))
        DetectionStats.shared.items = []

        let exp = expectation(description: "detection completes")
        sut.detectObjectsInFrame {
            exp.fulfill()
        }
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

        let exp = expectation(description: "detection completes")
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

        let exp = expectation(description: "post-warmup detection")
        sut.detectObjectsInFrame { exp.fulfill() }
        wait(for: [exp], timeout: 1.0)

        let dropStat = DetectionStats.shared.items.first(where: { $0.key == "Dropped Frames" })
        XCTAssertEqual(dropStat?.value, "1")
    }

    // MARK: - Geometry / overlay math
    func testOverlayRectMatchesLetterboxedMath() {
        let base = Base()
        let normalized = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.25)
        let rect = base.rectForNormalizedRect(normalizedRect: normalized, width: 200, height: 100)
        XCTAssertEqual(rect.width, 100, accuracy: 0.1)
        XCTAssertEqual(rect.height, 25, accuracy: 0.1)
        XCTAssertEqual(rect.origin.y, 50, accuracy: 0.1)
    }

    // MARK: - Performance budget guard
    func testDetectionLatencyWithinBudgetHelper() {
        let sut = VideoDetection()
        sut.setLastDetectionTimeForTesting(0.030) // 30ms
        XCTAssertTrue(sut.isWithinLatencyBudget())
        sut.setLastDetectionTimeForTesting(0.080)
        XCTAssertFalse(sut.isWithinLatencyBudget(budgetMs: 50))
    }

    // MARK: - Pixel format attributes
    func testPixelBufferAttributesFallbackWhenIdealFormatMissing() {
        let sut = VideoDetection()
        sut.setIdealFormatForTesting(nil)
        XCTAssertNil(sut.getPixelBufferAttributesForTesting())
    }

    // MARK: - Optimization path
    func testOptimizeOnLoadProducesNonLargerCopyAndValidModel() throws {
        let source = try rawModelURL()
        let sourceSize = try fileSize(at: source)

        let sut = CoreMLModel()
        sut.optimizeOnLoad = true
        sut.computeUnits = .cpuOnly // deterministic

        sut.loadTheModel(url: source)

        let exp = expectation(description: "model loads")
        DispatchQueue.main.asyncAfter(deadline: .now() + 6.0) {
            if sut.isValid {
                exp.fulfill()
            }
        }
        wait(for: [exp], timeout: 10.0)

        let optimizedURL = source.deletingPathExtension().appendingPathExtension("optimized.mlmodel")
        XCTAssertTrue(FileManager.default.fileExists(atPath: optimizedURL.path))

        let optimizedSize = try fileSize(at: optimizedURL)
        XCTAssertLessThanOrEqual(optimizedSize, sourceSize, "Optimized copy should not exceed original size")
        XCTAssertTrue(sut.wasOptimized)
        XCTAssertTrue(sut.isValid)
    }

    // MARK: - CI / test plan presence
    func testXCTestPlanListsCoreMLPlayerTestsTarget() throws {
        let testFile = URL(fileURLWithPath: #filePath)
        let repoRoot = testFile.deletingLastPathComponent().deletingLastPathComponent()
        let planURL = repoRoot.appendingPathComponent("CoreML Player.xctestplan")

        guard FileManager.default.fileExists(atPath: planURL.path) else {
            throw XCTSkip("xctestplan not found at expected path: \(planURL.path)")
        }

        let data = try Data(contentsOf: planURL)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let testTargets = json?["testTargets"] as? [[String: Any]]
        let target = testTargets?.first?["target"] as? [String: Any]

        XCTAssertEqual(target?["name"] as? String, "CoreMLPlayerTests")
    }

    // MARK: - Multi-function selection plumbing
    func testSelectedFunctionPropagatesToRequests() throws {
        let sut = try makeStubVideoDetection()
        CoreMLModel.sharedSelectedFunction = "fn_a"
        let exp = expectation(description: "detection")
        sut.detectObjectsInFrame { exp.fulfill() }
        wait(for: [exp], timeout: 1.0)
        XCTAssertEqual(Base.sharedLastFunctionName, "fn_a")
    }

    // MARK: - Stateful reset semantics
    func testStateCounterResetsOnModelChangeAndDisappearing() throws {
        let sut = try makeStubVideoDetection()
        sut.setVideoOrientationForTesting(.up)
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, 8, 8, kCVPixelFormatType_32BGRA, nil, &pb)
        guard let buffer = pb else { return XCTFail("buffer missing") }

        _ = sut.detectPixelBufferForTesting(buffer)
        _ = sut.detectPixelBufferForTesting(buffer)
        XCTAssertEqual(sut.metricsForTesting().stateFrameCounter, 2)

        sut.setModel(nil)
        XCTAssertEqual(sut.metricsForTesting().stateFrameCounter, 0)

        sut.setModel(try VNCoreMLModel(for: MLModel(contentsOf: compiledModelURL())))
        _ = sut.detectPixelBufferForTesting(buffer)
        sut.disappearing()
        XCTAssertEqual(sut.metricsForTesting().stateFrameCounter, 0)
    }

    // MARK: - File size helper
    private func fileSize(at url: URL) throws -> Int64 {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs[.size] as? NSNumber)?.int64Value ?? 0
    }
}

// MARK: - Stub video detection (avoids real Vision work)
private final class StubVideoDetection: VideoDetection {
    private let stubDetectionTimeMs: Double
    private let stubDetectionFPS: String
    private let stubObjects: Int
    private var cachedPixelBuffer: CVPixelBuffer?

    init(stubDetectionTimeMs: Double, stubDetectionFPS: String, stubObjects: Int) {
        self.stubDetectionTimeMs = stubDetectionTimeMs
        self.stubDetectionFPS = stubDetectionFPS
        self.stubObjects = stubObjects
        super.init()
    }

    override func getPixelBuffer() -> CVPixelBuffer? {
        if let cachedPixelBuffer {
            return cachedPixelBuffer
        }
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, 4, 4, kCVPixelFormatType_32BGRA, nil, &pb)
        cachedPixelBuffer = pb
        return pb
    }

    override func performObjectDetection(requestHandler: VNImageRequestHandler, vnModel: VNCoreMLModel, functionName: String? = nil, cropAndScale: VNImageCropAndScaleOption = .scaleFill) -> detectionOutput {
        Base.sharedLastFunctionName = functionName
        let object = DetectedObject(
            id: UUID(),
            label: "stub",
            confidence: "1.0",
            otherLabels: [],
            width: 0.1,
            height: 0.1,
            x: 0,
            y: 0
        )
        let objects = Array(repeating: object, count: stubObjects)
        let detTime = String(format: "%.0f ms", stubDetectionTimeMs)
        return (objects, detTime, stubDetectionFPS)
    }
}
