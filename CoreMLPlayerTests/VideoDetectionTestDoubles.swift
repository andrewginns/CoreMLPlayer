import Foundation
import CoreVideo
import Vision
@testable import CoreML_Player

/// Lightweight stub to avoid real Vision execution when testing scheduling and state.
class StubVideoDetection: VideoDetection {
    private let stubDetectionTimeMs: Double
    private let stubDetectionFPS: String
    private let stubObjects: Int
    private var cachedPixelBuffer: CVPixelBuffer?

    init(stubDetectionTimeMs: Double, stubDetectionFPS: String, stubObjects: Int) {
        self.stubDetectionTimeMs = stubDetectionTimeMs
        self.stubDetectionFPS = stubDetectionFPS
        self.stubObjects = stubObjects
        super.init()
    }

    override func getPixelBuffer() -> CVPixelBuffer? {
        if let cachedPixelBuffer {
            return cachedPixelBuffer
        }
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, 4, 4, kCVPixelFormatType_32BGRA, nil, &pb)
        cachedPixelBuffer = pb
        return pb
    }

    override func performObjectDetection(
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation,
        vnModel: VNCoreMLModel,
        functionName: String? = nil,
        cropAndScale: VNImageCropAndScaleOption = .scaleFill
    ) -> detectionOutput {
        Base.sharedLastFunctionName = functionName
        return stubbedOutput()
    }

    override func performObjectDetection(
        requestHandler: VNImageRequestHandler,
        vnModel: VNCoreMLModel,
        functionName: String? = nil,
        cropAndScale: VNImageCropAndScaleOption = .scaleFill
    ) -> detectionOutput {
        Base.sharedLastFunctionName = functionName
        return stubbedOutput()
    }

    private func stubbedOutput() -> detectionOutput {
        let object = DetectedObject(
            id: UUID(),
            label: "stub",
            confidence: "1.0",
            otherLabels: [],
            width: 0.1,
            height: 0.1,
            x: 0,
            y: 0
        )
        let objects = Array(repeating: object, count: stubObjects)
        let detTime = String(format: "%.0f ms", stubDetectionTimeMs)
        return (objects, detTime, stubDetectionFPS)
    }
}
