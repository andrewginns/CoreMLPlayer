import XCTest
@testable import CoreML_Player
import Vision
import CoreGraphics
import AppKit

final class VisionPipelineTests: XCTestCase {
    func testDetectedObjectsConversion() {
        let boundingBox = CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4)
        let label = VNClassificationObservation(identifier: "object", confidence: 0.75)
        let observation = VNRecognizedObjectObservation(boundingBox: boundingBox)
        observation.setValue([label], forKey: "labels")
        let duration: Duration = .milliseconds(40)

        let sut = Base()
        let output = sut.asDetectedObjects(visionObservationResults: [observation], detectionTime: duration)

        XCTAssertEqual(output.objects.first?.label, "object")
        XCTAssertEqual(output.objects.first?.width, boundingBox.width)
        XCTAssertEqual(output.detectionTime.contains("ms"), true)
        XCTAssertEqual(output.detectionFPS, "25")
    }

    func testRectForNormalizedRectFlipsYAxis() {
        let sut = Base()
        let normalized = CGRect(x: 0, y: 0, width: 0.5, height: 0.5)
        let rect = sut.rectForNormalizedRect(normalizedRect: normalized, width: 200, height: 100)

        XCTAssertEqual(rect.origin.y, 50)
        XCTAssertEqual(rect.size.width, 100)
    }

    func testOverlayRectMatchesLetterboxedMath() {
        let base = Base()
        let normalized = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.25)
        let rect = base.rectForNormalizedRect(normalizedRect: normalized, width: 200, height: 100)
        XCTAssertEqual(rect.width, 100, accuracy: 0.1)
        XCTAssertEqual(rect.height, 25, accuracy: 0.1)
        XCTAssertEqual(rect.origin.y, 50, accuracy: 0.1)
    }
}

extension VNClassificationObservation {
    convenience init(identifier: String, confidence: VNConfidence) {
        self.init()
        setValue(identifier, forKey: "identifier")
        setValue(confidence, forKey: "confidence")
    }
}

extension CGImage {
    static var mockSquare: CGImage {
        let context = CGContext(data: nil, width: 32, height: 32, bitsPerComponent: 8, bytesPerRow: 32 * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(NSColor.red.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: 32, height: 32))
        return context.makeImage()!
    }
}
