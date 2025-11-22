import XCTest
import CoreML
@testable import CoreML_Player

private func compiledYOLOURL(testCase: XCTestCase) throws -> URL {
    let bundle = Bundle(for: type(of: testCase))
    if let compiledURL = bundle.url(forResource: "YOLOv3Tiny", withExtension: "mlmodelc") {
        return compiledURL
    }
    guard let rawURL = bundle.url(forResource: "YOLOv3Tiny", withExtension: "mlmodel") else {
        XCTFail("Missing YOLOv3Tiny model in test bundle")
        throw XCTSkip("Model fixture unavailable")
    }
    return try MLModel.compileModel(at: rawURL)
}

private func allowLowPrecisionIfSupported(configuration: MLModelConfiguration, allowLowPrecision: Bool) -> MLModelConfiguration {
    let config = configuration
    config.allowLowPrecisionAccumulationOnGPU = allowLowPrecision && configuration.computeUnits != .cpuOnly
    return config
}

/// Tests focused on CoreMLModel's handling of model descriptions, IO guardrails, and configuration flags.
final class CoreMLModelFormatTests: XCTestCase {
    func testIdealFormatCapturedFromModelDescription() throws {
        let model = try MLModel(contentsOf: compiledYOLOURL(testCase: self))
        let sut = CoreMLModel()
        sut.setModelDescriptionInfo(model.modelDescription)

        guard let ideal = sut.idealFormat else {
            return XCTFail("idealFormat was not populated from model description")
        }

        XCTAssertGreaterThan(ideal.width, 0)
        XCTAssertGreaterThan(ideal.height, 0)
        XCTAssertNotEqual(ideal.type, 0)
    }

    func testModelIOValidationUsesFeatureDescriptionsPositive() throws {
        let mlModel = try MLModel(contentsOf: compiledYOLOURL(testCase: self))
        let base = Base()
        XCTAssertNoThrow(try base.checkModelIO(modelDescription: mlModel.modelDescription))
    }

    func testLowPrecisionDisabledWhenCPUOnly() {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        var applied = allowLowPrecisionIfSupported(configuration: config, allowLowPrecision: true)
        XCTAssertFalse(applied.allowLowPrecisionAccumulationOnGPU)

        let configGPU = MLModelConfiguration()
        configGPU.computeUnits = .cpuAndGPU
        applied = allowLowPrecisionIfSupported(configuration: configGPU, allowLowPrecision: true)
        XCTAssertTrue(applied.allowLowPrecisionAccumulationOnGPU)
    }
}
