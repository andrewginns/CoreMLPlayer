//
//  Base.swift
//  CoreML Player
//
//  Created by NA on 1/22/23.
//

import SwiftUI
import UniformTypeIdentifiers
import Vision
import ImageIO

class Base {
    typealias detectionOutput = (objects: [DetectedObject], detectionTime: String, detectionFPS: String)
    let emptyDetection: detectionOutput = ([], "", "")
    
    func selectFiles(contentTypes: [UTType], multipleSelection: Bool = true) -> [URL]? {
        let picker = NSOpenPanel()
        picker.allowsMultipleSelection = multipleSelection
        picker.allowedContentTypes = contentTypes
        picker.canChooseDirectories = false
        picker.canCreateDirectories = false
        
        if picker.runModal() == .OK {
            return picker.urls
        }
        
        return nil
    }
    
    // Old-style Alert is less work on Mac
    func showAlert(title: String, message: String? = nil) {
        let alert = NSAlert()
        alert.messageText = title
        if let message {
            alert.informativeText = message
        }
        alert.runModal()
    }
    
    func detectImageObjects(image: ImageFile?, model: VNCoreMLModel?) -> detectionOutput {
        guard let vnModel = model,
              let nsImage = image?.getNSImage(),
              let cgImage = nsImage.cgImageForCurrentRepresentation
        else {
            return emptyDetection
        }

        let orientation = nsImage.cgImagePropertyOrientation ?? .up
        #if DEBUG
        Base.sharedLastImageOrientation = orientation
        #endif

        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
        let cropOption = cropOptionForIdealFormat()
        return performObjectDetection(requestHandler: handler, vnModel: vnModel, functionName: CoreMLModel.sharedSelectedFunction, cropAndScale: cropOption)
    }
    
    func performObjectDetection(requestHandler: VNImageRequestHandler, vnModel: VNCoreMLModel, functionName: String? = nil, cropAndScale: VNImageCropAndScaleOption = .scaleFill) -> detectionOutput {
        var observationResults: [VNObservation]?
        let request = VNCoreMLRequest(model: vnModel) { (request, error) in
            observationResults = request.results
        }
        request.imageCropAndScaleOption = cropAndScale
        #if DEBUG
        Base.sharedLastFunctionName = functionName
        #endif

        let detectionTime = ContinuousClock().measure {
            do {
                try requestHandler.perform([request])
            } catch {
                #if DEBUG
                Base.sharedLastError = error
                #endif
            }
        }
        
        return asDetectedObjects(visionObservationResults: observationResults, detectionTime: detectionTime)
    }
    
    func asDetectedObjects(visionObservationResults: [VNObservation]?, detectionTime: Duration) -> detectionOutput {
        let classificationObservations = visionObservationResults as? [VNClassificationObservation]
        let objectObservations = visionObservationResults as? [VNRecognizedObjectObservation]

        var detectedObjects: [DetectedObject] = []
        
        let msTime = detectionTime.formatted(.units(allowed: [.seconds, .milliseconds], width: .narrow))
        let detectionFPS = String(format: "%.0f", Duration.seconds(1) / detectionTime)
        
        var labels: [(label: String, confidence: String)] = []
        
        // TODO: Implement more model types, and improve classificationObservations
        
        if let objectObservations // VNRecognizedObjectObservation
        {
            for obj in objectObservations {
                labels = []
                for l in obj.labels {
                    labels.append((label: l.identifier, confidence: String(format: "%.4f", l.confidence)))
                }
                
                let newObject = DetectedObject(
                    id: obj.uuid,
                    label: obj.labels.first?.identifier ?? "",
                    confidence: String(format: "%.3f", obj.confidence),
                    otherLabels: labels,
                    width: obj.boundingBox.width,
                    height: obj.boundingBox.height,
                    x: obj.boundingBox.origin.x,
                    y: obj.boundingBox.origin.y
                )
                
                detectedObjects.append(newObject)
            }
        }
        else if let classificationObservations, let mainObject = classificationObservations.first // VNClassificationObservation
        {
            // For now:
            for c in classificationObservations {
                labels.append((label: c.identifier, confidence: String(format: "%.4f", c.confidence)))
            }
            let label = "\(mainObject.identifier) (\(mainObject.confidence))"
            let newObject = DetectedObject(
                id: mainObject.uuid,
                label: label, //mainObject.identifier,
                confidence: String(format: "%.3f", mainObject.confidence),
                otherLabels: labels,
                width: 0.9,
                height: 0.85,
                x: 0.05,
                y: 0.05,
                isClassification: true
            )
            detectedObjects.append(newObject)
            #if DEBUG
            print("Classification Observation:")
            print(classificationObservations)
            #endif
        }
        else
        {
            #if DEBUG
            print("No objects found.")
            #endif
        }
        
        return (objects: detectedObjects, detectionTime: msTime, detectionFPS: detectionFPS)
    }
    
    func checkModelIO(modelDescription: MLModelDescription) throws {
        let inputs = modelDescription.inputDescriptionsByName.values
        let outputs = modelDescription.outputDescriptionsByName.values

        let hasImageInput = inputs.contains { $0.type == .image }
        if !hasImageInput {
            DispatchQueue.main.async {
                self.showAlert(title: "This model does not accept Images as an input, and at the moment is not supported.")
            }
            throw MLModelError(.io)
        }

        let supportsOutput = outputs.contains { desc in
            switch desc.type {
            case .multiArray, .dictionary, .string:
                return true
            default:
                return false
            }
        }

        if !supportsOutput {
            DispatchQueue.main.async {
                self.showAlert(title: "This model is not of type Object Detection or Classification, and at the moment is not supported.")
            }
            throw MLModelError(.io)
        }
    }

    /// Derive crop-and-scale based on the ideal format if available (square ⇒ centerCrop, otherwise scaleFit)
    func cropOptionForIdealFormat() -> VNImageCropAndScaleOption {
        if let format = CoreMLModel.sharedIdealFormat {
            return format.width == format.height ? .centerCrop : .scaleFit
        }
        return .scaleFill
    }
    
    func prepareObjectForSwiftUI(object: DetectedObject, geometry: GeometryProxy) -> CGRect {
        let objectRect = CGRect(x: object.x, y: object.y, width: object.width, height: object.height)
        
        return rectForNormalizedRect(normalizedRect: objectRect, width: Int(geometry.size.width), height: Int(geometry.size.height))
    }
    
    func rectForNormalizedRect(normalizedRect: CGRect, width: Int, height: Int) -> CGRect {
        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -CGFloat(height))
        return VNImageRectForNormalizedRect(normalizedRect, width, height).applying(transform)
    }
}

extension NSImage {
    // Without this NSImage returns size in points not pixels
    var actualSize: NSSize {
        guard representations.count > 0 else { return .zero }
        return NSSize(width: representations[0].pixelsWide, height: representations[0].pixelsHigh)
    }

    /// Current CGImage for the representation, if available.
    var cgImageForCurrentRepresentation: CGImage? {
        return cgImage(forProposedRect: nil, context: nil, hints: nil)
    }

    /// EXIF orientation mapping for Vision handlers.
    var cgImagePropertyOrientation: CGImagePropertyOrientation? {
        guard let tiffData = self.tiffRepresentation,
              let source = CGImageSourceCreateWithData(tiffData as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let raw = properties[kCGImagePropertyOrientation] as? UInt32,
              let orientation = CGImagePropertyOrientation(rawValue: raw) else {
            return nil
        }
        return orientation
    }
}

extension VNRecognizedObjectObservation: Identifiable {
    public var id: UUID {
        return self.uuid
    }
    static func ==(lhs: VNRecognizedObjectObservation, rhs: VNRecognizedObjectObservation) -> Bool {
        return lhs.uuid == rhs.uuid
    }
}

#if DEBUG
extension Base {
    /// Last used image orientation (testing only).
    static var sharedLastImageOrientation: CGImagePropertyOrientation?
    /// Last Vision error encountered (testing only).
    static var sharedLastError: Error?
    /// Last function name requested on a VNCoreMLRequest (testing only).
    static var sharedLastFunctionName: String?
}
#endif
