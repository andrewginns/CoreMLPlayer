import XCTest
import CoreML
import Vision
@testable import CoreML_Player

/// Exercises real multi-function and stateful model flows when fixtures can be generated.
final class MultiFunctionAndStatefulTests: XCTestCase {
    func testMultiFunctionModelSelectionExecutesChosenFunction() throws {
        let (compiledURL, functions) = try FixtureBuilder.ensureMultiFunctionModel()
        let mlModel = try MLModel(contentsOf: compiledURL)
        guard functions.count >= 2 else {
            throw XCTSkip("Insufficient functions in generated model")
        }

        let fn = functions[1] // choose second function (plus_one)
        CoreMLModel.sharedSelectedFunction = fn

        let vnModel = try VNCoreMLModel(for: mlModel)
        let handler = VNImageRequestHandler(cgImage: CGImage.mockSquare, options: [:])
        let base = Base()

        let result = base.performObjectDetection(requestHandler: handler, vnModel: vnModel, functionName: fn)

        // We don't care about numeric outputs, only that the function name is plumbed and captured.
        XCTAssertEqual(Base.sharedLastFunctionName, fn)
        XCTAssertNotNil(result.detectionTime)
    }

    func testStatefulModelPersistsAcrossCalls() throws {
        let compiledURL = try FixtureBuilder.ensureStatefulModel()
        let mlModel = try MLModel(contentsOf: compiledURL)
        let vnModel = try VNCoreMLModel(for: mlModel)

        // Build two input buffers; we reuse the same buffer to simulate sequential frames.
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, 1, 1, kCVPixelFormatType_32BGRA, nil, &pb)
        guard let buffer = pb else { return XCTFail("Failed to create pixel buffer") }

        let vd = VideoDetection()
        vd.setModel(vnModel)
        vd.setVideoOrientationForTesting(.up)

        // First detection warms up state; second should increment the state counter (tracked internally).
        _ = vd.detectPixelBufferForTesting(buffer)
        let second = vd.detectPixelBufferForTesting(buffer)
        XCTAssertGreaterThan(second.stateFrameCounter, 1)
    }
}
