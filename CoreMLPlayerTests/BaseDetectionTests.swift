import XCTest
import Vision
import CoreML
import CoreVideo
import AppKit
@testable import CoreML_Player

/// Tests for lightweight image detection helpers that don't require full model execution.
final class BaseDetectionTests: XCTestCase {
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

    func testImageOrientationIsCapturedAndUsed() throws {
        let model = try VNCoreMLModel(for: MLModel(contentsOf: compiledYOLOURL()))
        let base = Base()
        let portraitImage = Self.makeImage(width: 40, height: 80)
        let imageFile = ImageFile(name: "portrait", type: "png", url: URL(fileURLWithPath: "/tmp/portrait.png"))
        ImageFile.nsImageOverrideForTests = portraitImage
        defer { ImageFile.nsImageOverrideForTests = nil }

        _ = base.detectImageObjects(image: imageFile, model: model)

        XCTAssertEqual(Base.sharedLastImageOrientation, .up) // default when no EXIF, but captured
    }

    func testCropAndScaleFollowsIdealFormatSquareCenterCrop() {
        CoreMLModel.sharedIdealFormat = (width: 224, height: 224, type: kCVPixelFormatType_32BGRA)
        let base = Base()
        XCTAssertEqual(base.cropOptionForIdealFormat(), .centerCrop)

        CoreMLModel.sharedIdealFormat = (width: 224, height: 112, type: kCVPixelFormatType_32BGRA)
        XCTAssertEqual(base.cropOptionForIdealFormat(), .scaleFit)
    }

    func testCropOptionMatchesIdealFormatShapes() {
        CoreMLModel.sharedIdealFormat = (width: 320, height: 160, type: kCVPixelFormatType_32BGRA)
        XCTAssertEqual(Base().cropOptionForIdealFormat(), .scaleFit)
        CoreMLModel.sharedIdealFormat = nil
    }

    func testPerformObjectDetectionCapturesErrors() {
        let base = Base()
        let bogusURL = URL(fileURLWithPath: "/tmp/does_not_exist.png")
        let handler = VNImageRequestHandler(url: bogusURL)
        let vnModel = try! VNCoreMLModel(for: MLModel(contentsOf: try! compiledYOLOURL()))

        _ = base.performObjectDetection(requestHandler: handler, vnModel: vnModel)

        XCTAssertNotNil(Base.sharedLastError)
    }

    // MARK: - Helpers
    private func compiledYOLOURL() throws -> URL {
        let bundle = Bundle(for: type(of: self))
        if let compiledURL = bundle.url(forResource: "YOLOv3Tiny", withExtension: "mlmodelc") {
            return compiledURL
        }
        guard let rawURL = bundle.url(forResource: "YOLOv3Tiny", withExtension: "mlmodel") else {
            throw XCTSkip("YOLOv3Tiny model not present in test bundle")
        }
        return try MLModel.compileModel(at: rawURL)
    }

    private static func makeImage(width: Int, height: Int) -> NSImage {
        let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: width,
            pixelsHigh: height,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: width * 4,
            bitsPerPixel: 32
        )!
        rep.size = NSSize(width: width, height: height)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        NSColor.red.setFill()
        NSBezierPath(rect: NSRect(x: 0, y: 0, width: width, height: height)).fill()
        NSGraphicsContext.restoreGraphicsState()
        let image = NSImage(size: NSSize(width: width, height: height))
        image.addRepresentation(rep)
        return image
    }
}
