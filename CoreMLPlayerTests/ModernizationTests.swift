import XCTest
import Vision
import CoreML
@testable import CoreML_Player

final class ModernizationTests: XCTestCase {
    private func compiledModelURL() throws -> URL {
        let bundle = Bundle(for: type(of: self))
        if let compiledURL = bundle.url(forResource: "YOLOv3Tiny", withExtension: "mlmodelc") {
            return compiledURL
        }
        guard let rawURL = bundle.url(forResource: "YOLOv3Tiny", withExtension: "mlmodel") else {
            XCTFail("Missing YOLOv3Tiny model in test bundle")
            throw XCTSkip("Model fixture unavailable")
        }
        return try MLModel.compileModel(at: rawURL)
    }

    // MARK: - Base / detection helpers
    func testDetectImageObjectsReturnsEmptyWhenInputsMissing() {
        let sut = Base()
        let output = sut.detectImageObjects(image: nil, model: nil)
        XCTAssertTrue(output.objects.isEmpty)
        XCTAssertEqual(output.detectionTime, "")
        XCTAssertEqual(output.detectionFPS, "")
    }

    func testClassificationConversionPreservesLabelsAndMarksClassification() {
        let sut = Base()
        let labelA = VNClassificationObservation(identifier: "cat", confidence: 0.8)
        let labelB = VNClassificationObservation(identifier: "dog", confidence: 0.6)
        let duration: Duration = .milliseconds(10) // 100 FPS

        let result = sut.asDetectedObjects(
            visionObservationResults: [labelA, labelB],
            detectionTime: duration
        )

        guard let object = result.objects.first else {
            return XCTFail("Expected one classification object")
        }

        XCTAssertTrue(object.isClassification)
        XCTAssertEqual(object.otherLabels.count, 2)
        XCTAssertEqual(object.width, 0.9, accuracy: 0.001)   // synthetic box used for classification
        XCTAssertEqual(object.height, 0.85, accuracy: 0.001)
        XCTAssertEqual(result.detectionFPS, "100")
        XCTAssertTrue(result.detectionTime.contains("ms"))
    }

    // MARK: - Model description / ideal format
    func testIdealFormatCapturedFromModelDescription() throws {
        let model = try MLModel(contentsOf: compiledModelURL())
        let sut = CoreMLModel()
        sut.setModelDescriptionInfo(model.modelDescription)

        guard let ideal = sut.idealFormat else {
            return XCTFail("idealFormat was not populated from model description")
        }

        XCTAssertGreaterThan(ideal.width, 0)
        XCTAssertGreaterThan(ideal.height, 0)
        XCTAssertNotEqual(ideal.type, 0)
    }

    // MARK: - Video lifecycle / scheduling
    func testRepeatIntervalClampsToMinimumWhenLastDetectionIsHigh() {
        let sut = VideoDetection()
        sut.setVideoInfoForTesting((isPlayable: true, frameRate: 30, duration: .zero, size: .zero))
        sut.setLastDetectionTimeForTesting(1.0) // 1s > frame interval

        let interval = sut.getRepeatInterval()
        XCTAssertEqual(interval, 0.02, accuracy: 0.0001) // clamped minimum
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
