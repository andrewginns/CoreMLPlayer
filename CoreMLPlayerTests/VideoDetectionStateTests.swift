import XCTest
import Vision
import CoreML
import CoreVideo
import CoreGraphics
@testable import CoreML_Player

/// Tests that focus on VideoDetection state, orientation, and stateful model plumbing.
final class VideoDetectionStateTests: XCTestCase {
    func testStateTokenPersistsAcrossFrames() throws {
        let vd = VideoDetection()
        let vnModel = try VNCoreMLModel(for: MLModel(contentsOf: compiledModelURL()))
        vd.setModel(vnModel)
        vd.setVideoOrientationForTesting(.up)
        vd.setIdealFormatForTesting((width: 32, height: 32, type: kCVPixelFormatType_32BGRA))

        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, 32, 32, kCVPixelFormatType_32BGRA, nil, &pixelBuffer)
        guard let pb = pixelBuffer else { return XCTFail("Failed to create pixel buffer") }

        let first = vd.detectPixelBufferForTesting(pb)
        let second = vd.detectPixelBufferForTesting(pb)

        XCTAssertEqual(second.stateFrameCounter, first.stateFrameCounter + 1)
    }

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

    func testStatefulModelStateAdvancesAcrossFrames() throws {
        let compiledURL = try FixtureBuilder.ensureStatefulModel()
        let vnModel = try VNCoreMLModel(for: MLModel(contentsOf: compiledURL))
        let sut = VideoDetection()
        sut.setModel(vnModel)
        sut.setVideoOrientationForTesting(.up)
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, 1, 1, kCVPixelFormatType_32BGRA, nil, &pb)
        guard let buffer = pb else { return XCTFail("buffer missing") }

        _ = sut.detectPixelBufferForTesting(buffer)
        guard let first = Base.sharedLastStateValues?["state_out"]?.multiArrayValue else {
            throw XCTSkip("state outputs not available on this platform")
        }
        _ = sut.detectPixelBufferForTesting(buffer)
        guard let second = Base.sharedLastStateValues?["state_out"]?.multiArrayValue else {
            throw XCTSkip("state outputs not available on this platform")
        }

        XCTAssertLessThan(first[0].doubleValue, second[0].doubleValue)
    }

    func testVideoOrientationPersistsDuringDetection() throws {
        let model = try VNCoreMLModel(for: MLModel(contentsOf: compiledModelURL()))
        let sut = VideoDetection()
        sut.setModel(model)
        sut.setVideoOrientationForTesting(.left)
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, 4, 4, kCVPixelFormatType_32BGRA, nil, &pb)
        guard let buffer = pb else { return XCTFail("pixel buffer missing") }

        _ = sut.detectPixelBufferForTesting(buffer)

        XCTAssertEqual(VideoDetection.sharedLastVideoOrientation, .left)
        let rect = Base().rectForNormalizedRect(normalizedRect: CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.25), width: 200, height: 100)
        XCTAssertEqual(rect.origin.y, 50, accuracy: 0.1) // letterbox math still holds
    }

    // MARK: - Helpers
    private func compiledModelURL() throws -> URL {
        let bundle = Bundle(for: type(of: self))
        if let compiledURL = bundle.url(forResource: "YOLOv3Tiny", withExtension: "mlmodelc") {
            return compiledURL
        }
        guard let rawURL = bundle.url(forResource: "YOLOv3Tiny", withExtension: "mlmodel") else {
            throw XCTSkip("YOLOv3Tiny model not present in test bundle")
        }
        return try MLModel.compileModel(at: rawURL)
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
