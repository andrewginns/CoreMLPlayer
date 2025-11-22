import XCTest
import CoreML
import Vision
import AVFoundation
@testable import CoreML_Player

private func allowLowPrecisionIfSupported(configuration: MLModelConfiguration, allowLowPrecision: Bool) -> MLModelConfiguration {
    var config = configuration
    config.allowLowPrecisionAccumulationOnGPU = allowLowPrecision && configuration.computeUnits != .cpuOnly
    return config
}

final class ModernizationAdvancedTests: XCTestCase {
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

    // MARK: Orientation / crop / pixel format
    func testImageOrientationIsCapturedAndUsed() throws {
        let model = try VNCoreMLModel(for: MLModel(contentsOf: compiledModelURL()))
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

    func testPixelBufferAttributesPreferIdealFormat() {
        let vd = VideoDetection()
        vd.setIdealFormatForTesting((width: 320, height: 240, type: kCVPixelFormatType_32BGRA))
        let attrs = vd.getPixelBufferAttributesForTesting()
        XCTAssertEqual(attrs?[kCVPixelBufferPixelFormatTypeKey as String] as? OSType, kCVPixelFormatType_32BGRA)
    }

    // MARK: IO validation
    func testModelIOValidationUsesFeatureDescriptionsPositive() throws {
        let mlModel = try MLModel(contentsOf: compiledModelURL())
        let base = Base()
        XCTAssertNoThrow(try base.checkModelIO(modelDescription: mlModel.modelDescription))
    }

    // MARK: Config guardrails
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

    // MARK: Error surfacing
    func testPerformObjectDetectionCapturesErrors() {
        let base = Base()
        let bogusURL = URL(fileURLWithPath: "/tmp/does_not_exist.png")
        let handler = VNImageRequestHandler(url: bogusURL)
        let vnModel = try! VNCoreMLModel(for: MLModel(contentsOf: try! compiledModelURL()))

        _ = base.performObjectDetection(requestHandler: handler, vnModel: vnModel)

        XCTAssertNotNil(Base.sharedLastError)
    }

    // MARK: Stateful signal
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

    // MARK: Optimization toggle
    func testOptimizeToggleMarksModelAsOptimized() throws {
        let sut = CoreMLModel()
        sut.optimizeOnLoad = true
        let bundle = Bundle(for: type(of: self))
        let url: URL
        if let raw = bundle.url(forResource: "YOLOv3Tiny", withExtension: "mlmodel") {
            url = raw
        } else {
            url = try compiledModelURL()
        }
        sut.loadTheModel(url: url)

        let expectation = expectation(description: "model optimized")
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            if sut.wasOptimized && sut.isValid {
                expectation.fulfill()
            }
        }
        wait(for: [expectation], timeout: 4.0)

        XCTAssertTrue(sut.wasOptimized, "optimizeOnLoad should mark the model as optimized")
        XCTAssertTrue(sut.isValid, "model should be valid after load")
    }

    // MARK: Helpers
    private static func makeImage(width: Int, height: Int) -> NSImage {
        let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: width * 4, bitsPerPixel: 32)!
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
