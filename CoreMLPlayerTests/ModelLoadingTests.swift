import XCTest
@testable import CoreML_Player
import CoreML
import Vision

final class ModelLoadingTests: XCTestCase {
    private func compiledModelURL() throws -> URL {
        if let compiledURL = Bundle(for: type(of: self)).url(forResource: "YOLOv3Tiny", withExtension: "mlmodelc") {
            return compiledURL
        }

        guard let rawURL = Bundle(for: type(of: self)).url(forResource: "YOLOv3Tiny", withExtension: "mlmodel") else {
            XCTFail("Missing YOLOv3Tiny model in test bundle")
            throw XCTSkip("Model fixture unavailable")
        }

        return try MLModel.compileModel(at: rawURL)
    }

    func testModelCompilationAndConfiguration() throws {
        let compiledURL = try compiledModelURL()
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        configuration.allowLowPrecisionAccumulationOnGPU = true

        let mlModel = try MLModel(contentsOf: compiledURL, configuration: configuration)

        let sut = CoreMLModel()
        XCTAssertNoThrow(try sut.checkModelIO(modelDescription: mlModel.modelDescription))
        XCTAssertEqual(configuration.computeUnits, .cpuOnly)
        XCTAssertTrue(configuration.allowLowPrecisionAccumulationOnGPU)
    }

    func testModelWarmupRequestSucceeds() throws {
        let compiledURL = try compiledModelURL()
        let mlModel = try MLModel(contentsOf: compiledURL)
        let vnModel = try VNCoreMLModel(for: mlModel)

        let handler = VNImageRequestHandler(cgImage: CGImage.mockSquare, options: [:])
        let request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .centerCrop

        // Running the warm-up request can print a benign warning about missing
        // `precisionRecallCurves` on non-updatable models; the model still
        // executes correctly, so we keep the request and only assert success.

        let expectation = expectation(description: "Warmup completes")
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
                expectation.fulfill()
            } catch {
                XCTFail("Warmup failed: \(error)")
            }
        }

        wait(for: [expectation], timeout: 5.0)
    }
}
