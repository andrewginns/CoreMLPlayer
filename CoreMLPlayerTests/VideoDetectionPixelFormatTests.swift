import XCTest
import CoreVideo
@testable import CoreML_Player

/// Pixel format selection and fallbacks for video pipeline.
final class VideoDetectionPixelFormatTests: XCTestCase {
    func testPixelBufferAttributesPreferIdealFormat() {
        let vd = VideoDetection()
        vd.setIdealFormatForTesting((width: 320, height: 240, type: kCVPixelFormatType_32BGRA))
        let attrs = vd.getPixelBufferAttributesForTesting()
        XCTAssertEqual(attrs?[kCVPixelBufferPixelFormatTypeKey as String] as? OSType, kCVPixelFormatType_32BGRA)
    }

    func testPixelBufferAttributesFallbackWhenIdealFormatMissing() {
        let sut = VideoDetection()
        sut.setIdealFormatForTesting(nil)
        XCTAssertNil(sut.getPixelBufferAttributesForTesting())
    }

    func testPixelFormatFallbackReturnsOverrideBuffer() {
        let sut = VideoDetection()
        sut.setIdealFormatForTesting((width: 320, height: 240, type: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange))
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, 2, 2, kCVPixelFormatType_32BGRA, nil, &pb)
        VideoDetection.testFallbackPixelBuffer = pb

        let fallback = sut.forcePixelBufferFallbackForTesting()

        XCTAssertNotNil(fallback)
        XCTAssertNil(sut.getPixelBufferAttributesForTesting())
        VideoDetection.testFallbackPixelBuffer = nil
    }
}
