import XCTest
@testable import CoreML_Player
import CoreML

final class ModelLoadingTests: XCTestCase {
    func testModelCompilationAndConfiguration() throws {
        guard let modelURL = Bundle(for: type(of: self)).url(forResource: "YOLOv3Tiny", withExtension: "mlmodel") else {
            XCTFail("Missing YOLOv3Tiny.mlmodel in test bundle")
            return
        }

        let compiledURL = try MLModel.compileModel(at: modelURL)
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
        guard let modelURL = Bundle(for: type(of: self)).url(forResource: "YOLOv3Tiny", withExtension: "mlmodel") else {
            XCTFail("Missing YOLOv3Tiny.mlmodel in test bundle")
            return
        }

        let compiledURL = try MLModel.compileModel(at: modelURL)
        let mlModel = try MLModel(contentsOf: compiledURL)
        let vnModel = try VNCoreMLModel(for: mlModel)

        let handler = VNImageRequestHandler(cgImage: CGImage.mockSquare, options: [:])
        let request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .centerCrop

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
