import XCTest
import CoreML
import Vision
@testable import CoreML_Player

/// Tests covering multi-function model selection and stateful model behavior.
final class MultiFunctionModelTests: XCTestCase {
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

    func testSelectedFunctionPropagatesToRequests() throws {
        let sut = try makeStubVideoDetection()
        CoreMLModel.sharedSelectedFunction = "fn_a"
        let exp = expectation(description: "detection")
        sut.detectObjectsInFrame { exp.fulfill() }
        wait(for: [exp], timeout: 1.0)
        XCTAssertEqual(Base.sharedLastFunctionName, "fn_a")
    }

    func testAutoloadRestoresSelectedFunction() throws {
        let (compiledURL, functions) = try FixtureBuilder.ensureMultiFunctionModel()
        guard functions.count > 1 else { throw XCTSkip("insufficient functions") }

        let sut = CoreMLModel()
        sut.autoloadSelection = .reloadCompiled
        sut.compiledModelURL = compiledURL
        sut.storedSelectedFunctionName = functions[1]
        sut.selectedFunction = nil

        let exp = expectation(description: "autoload loads model")
        sut.autoload()
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            if sut.isValid { exp.fulfill() }
        }
        wait(for: [exp], timeout: 6.0)

        XCTAssertEqual(sut.selectedFunction, functions[1])
        if let model = sut.model {
            _ = Base().performObjectDetection(cgImage: CGImage.mockSquare, orientation: .up, vnModel: model, functionName: sut.selectedFunction)
            XCTAssertEqual(Base.sharedLastFunctionName, functions[1])
        }
        sut.unSelectModel()
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
