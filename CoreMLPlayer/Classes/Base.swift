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
import CoreML

class Base {
    typealias detectionOutput = (objects: [DetectedObject], detectionTime: String, detectionFPS: String)
    let emptyDetection: detectionOutput = ([], "", "")
    /// Serial queue for all model inferences to avoid concurrent contention and to centralize error handling.
    static let inferenceQueue = DispatchQueue(label: "com.coremlplayer.inference")
    /// Map underlying MLModels so we can reach model descriptions/state when only a VNCoreMLModel is in hand.
    private static let underlyingModelMap = NSMapTable<VNCoreMLModel, MLModel>(keyOptions: .weakMemory, valueOptions: .strongMemory)
    static func register(mlModel: MLModel, for vnModel: VNCoreMLModel) {
        underlyingModelMap.setObject(mlModel, forKey: vnModel)
    }
    private func underlyingModel(for vnModel: VNCoreMLModel) -> MLModel? {
        if let mapped = Base.underlyingModelMap.object(forKey: vnModel) {
            return mapped
        }
        if let ml = vnModel.value(forKey: "model") as? MLModel {
            Base.underlyingModelMap.setObject(ml, forKey: vnModel)
            return ml
        }
        return nil
    }

    /// Reusable per-model context; weakly keyed to avoid leaks when models are swapped.
    private static let requestCache = NSMapTable<VNCoreMLModel, ModelContext>(keyOptions: .weakMemory, valueOptions: .strongMemory)

    private final class ModelContext {
        let model: VNCoreMLModel
        let mlModel: MLModel?
        let sequenceHandler = VNSequenceRequestHandler()
        let isStateful: Bool
        let inputDescriptions: [String: MLFeatureDescription]
        let outputDescriptions: [String: MLFeatureDescription]
        let stateInputNames: [String]
        let stateOutputNames: [String]
        let imageInputName: String?
        var mlStateStorage: Any?
        @available(macOS 15.0, *)
        var mlState: MLState? {
            get { mlStateStorage as? MLState }
            set { mlStateStorage = newValue }
        }
        var manualState: MLFeatureProvider?

        init(model: VNCoreMLModel, underlying: MLModel?) {
            self.model = model
            self.mlModel = underlying
            if let description = underlying?.modelDescription {
                self.inputDescriptions = description.inputDescriptionsByName
                self.outputDescriptions = description.outputDescriptionsByName
            } else {
                self.inputDescriptions = [:]
                self.outputDescriptions = [:]
            }
            self.stateInputNames = Self.names(containing: "state", in: inputDescriptions)
            self.stateOutputNames = Self.names(containing: "state", in: outputDescriptions)
            self.isStateful = !stateInputNames.isEmpty || !stateOutputNames.isEmpty
            self.imageInputName = inputDescriptions.first(where: { $0.value.type == .image })?.key

            #if DEBUG
            let multiArrayInputs = inputDescriptions.keys.filter { inputDescriptions[$0]?.type == .multiArray }
            let multiArrayOutputs = outputDescriptions.keys.filter { outputDescriptions[$0]?.type == .multiArray }
            if isStateful {
                print("[CoreMLPlayer] Stateful model detected. State inputs: \(stateInputNames); outputs: \(stateOutputNames)")
            } else if !multiArrayInputs.isEmpty || !multiArrayOutputs.isEmpty {
                print("[CoreMLPlayer] No explicit 'state' keys found. Multi-array inputs: \(multiArrayInputs); outputs: \(multiArrayOutputs)")
            }
            #endif
        }

        func resetState() {
            mlStateStorage = nil
            manualState = nil
        }

        private static func names(containing needle: String, in dict: [String: MLFeatureDescription]) -> [String] {
            return dict.keys.filter { $0.lowercased().contains(needle) }
        }
    }

    private func context(for model: VNCoreMLModel) -> ModelContext {
        if let cached = Base.requestCache.object(forKey: model) {
            return cached
        }
        let fresh = ModelContext(model: model, underlying: underlyingModel(for: model))
        Base.requestCache.setObject(fresh, forKey: model)
        return fresh
    }
    
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

        let cropOption = cropOptionForIdealFormat()
        return performObjectDetection(cgImage: cgImage, orientation: orientation, vnModel: vnModel, functionName: CoreMLModel.sharedSelectedFunction, cropAndScale: cropOption)
    }
    
    func performObjectDetection(requestHandler: VNImageRequestHandler, vnModel: VNCoreMLModel, functionName: String? = nil, cropAndScale: VNImageCropAndScaleOption = .scaleFill) -> detectionOutput {
        let ctx = context(for: vnModel)
        if ctx.isStateful {
            return performStatefulDetection(input: .handler(requestHandler), context: ctx, cropAndScale: cropAndScale, functionName: functionName)
        }
        return performVisionDetection(with: ctx, cropAndScale: cropAndScale, functionName: functionName) { request in
            try requestHandler.perform([request])
        }
    }

    /// Pixel-buffer based detection path (preferred for video/stateful use cases).
    func performObjectDetection(pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation, vnModel: VNCoreMLModel, functionName: String? = nil, cropAndScale: VNImageCropAndScaleOption = .scaleFill) -> detectionOutput {
        let ctx = context(for: vnModel)
        if ctx.isStateful {
            return performStatefulDetection(input: .pixelBuffer(pixelBuffer, orientation), context: ctx, cropAndScale: cropAndScale, functionName: functionName)
        }
        return performVisionDetection(with: ctx, cropAndScale: cropAndScale, functionName: functionName) { request in
            try ctx.sequenceHandler.perform([request], on: pixelBuffer, orientation: orientation)
        }
    }

    /// CGImage-based detection path used by the image gallery.
    func performObjectDetection(cgImage: CGImage, orientation: CGImagePropertyOrientation, vnModel: VNCoreMLModel, functionName: String? = nil, cropAndScale: VNImageCropAndScaleOption = .scaleFill) -> detectionOutput {
        let ctx = context(for: vnModel)
        #if DEBUG
        Base.sharedLastImageOrientation = orientation
        #endif
        if ctx.isStateful {
            return performStatefulDetection(input: .cgImage(cgImage, orientation), context: ctx, cropAndScale: cropAndScale, functionName: functionName)
        }
        return performVisionDetection(with: ctx, cropAndScale: cropAndScale, functionName: functionName) { request in
            try ctx.sequenceHandler.perform([request], on: cgImage, orientation: orientation)
        }
    }

    /// Vision-backed inference path (non-stateful).
    private func performVisionDetection(with context: ModelContext, cropAndScale: VNImageCropAndScaleOption, functionName: String?, operation: (VNCoreMLRequest) throws -> Void) -> detectionOutput {
        var observationResults: [VNObservation]?
        let request = VNCoreMLRequest(model: context.model) { request, _ in
            observationResults = request.results
        }
        request.preferBackgroundProcessing = true
        request.imageCropAndScaleOption = cropAndScale

        #if DEBUG
        Base.sharedLastFunctionName = functionName
        #endif

        let detectionTime = ContinuousClock().measure {
            do {
                try Base.inferenceQueue.sync {
                    try operation(request)
                }
            } catch {
                #if DEBUG
                Base.sharedLastError = error
                #endif
            }
        }

        return asDetectedObjects(visionObservationResults: observationResults, detectionTime: detectionTime)
    }
    
    private enum DetectionInput {
        case pixelBuffer(CVPixelBuffer, CGImagePropertyOrientation)
        case cgImage(CGImage, CGImagePropertyOrientation)
        case handler(VNImageRequestHandler)
    }

    /// Core ML stateful inference path used when the model declares state inputs/outputs.
    private func performStatefulDetection(input: DetectionInput, context: ModelContext, cropAndScale: VNImageCropAndScaleOption, functionName: String?) -> detectionOutput {
        #if DEBUG
        Base.sharedLastFunctionName = functionName
        #endif

        var outputProvider: MLFeatureProvider?

        let detectionTime = ContinuousClock().measure {
            Base.inferenceQueue.sync {
                guard let mlModel = context.mlModel,
                      let features = makeFeatureProvider(for: context, input: input) else { return }
                do {
                    if #available(macOS 15.0, *) {
                        if context.mlState == nil {
                            context.mlState = mlModel.makeState()
                        }
                        if let state = context.mlState {
                            outputProvider = try mlModel.prediction(from: features, using: state)
                        } else {
                            outputProvider = try mlModel.prediction(from: features)
                        }
                    } else {
                        outputProvider = try mlModel.prediction(from: features)
                        if let provider = outputProvider {
                            context.manualState = extractState(from: provider, outputNames: context.stateOutputNames)
                        }
                    }
                } catch {
                    #if DEBUG
                    Base.sharedLastError = error
                    #endif
                }
            }
        }

        let seconds = Double(detectionTime.components.seconds) + (Double(detectionTime.components.attoseconds) / 1_000_000_000_000_000_000)
        let msTime = String(format: "%.0f ms", seconds * 1000)
        let detectionFPS = seconds > 0 ? String(format: "%.0f", 1.0 / seconds) : "0"

        let objects = outputProvider.flatMap { detectedObjects(from: $0) } ?? []

        #if DEBUG
        if let provider = outputProvider {
            var states: [String: MLFeatureValue] = [:]
            for name in context.stateOutputNames {
                if provider.featureNames.contains(name),
                   let value = provider.featureValue(for: name) {
                    states[name] = value
                }
            }
            Base.sharedLastStateValues = states.isEmpty ? nil : states
        }
        #endif

        if let provider = outputProvider,
           let extracted = extractState(from: provider, outputNames: context.stateOutputNames) {
            context.manualState = extracted
        }

        return (objects, msTime, detectionFPS)
    }

    /// Build an MLFeatureProvider for the current input, including any carried state.
    private func makeFeatureProvider(for context: ModelContext, input: DetectionInput) -> MLFeatureProvider? {
        var dict: [String: MLFeatureValue] = [:]
        var pixelBuffer: CVPixelBuffer?
        var cgImage: CGImage?

        switch input {
        case .pixelBuffer(let pb, _):
            pixelBuffer = pb
        case .cgImage(let cg, _):
            cgImage = cg
        case .handler:
            break
        }

        for (name, desc) in context.inputDescriptions {
            switch desc.type {
            case .image:
                if let pb = pixelBuffer {
                    dict[name] = MLFeatureValue(pixelBuffer: pb)
                } else if let cg = cgImage, let constraint = desc.imageConstraint {
                    dict[name] = try? MLFeatureValue(cgImage: cg, constraint: constraint, options: [:])
                }
            case .multiArray:
                // State inputs reuse prior state if available, otherwise zeros.
                if name.lowercased().contains("state") {
                    if let manual = context.manualState?.featureValue(for: name) {
                        dict[name] = manual
                        continue
                    }
                    if #available(macOS 15.0, *), let state = context.mlState {
                        var captured: MLFeatureValue?
                        state.withMultiArray(for: name) { buffer in
                            captured = MLFeatureValue(multiArray: buffer)
                        }
                        if let captured {
                            dict[name] = captured
                            continue
                        }
                    }
                }

                if let arr = multiArray(for: desc, fill: 1.0) {
                    dict[name] = MLFeatureValue(multiArray: arr)
                }
            default:
                continue
            }
        }

        guard !dict.isEmpty else { return nil }
        return try? MLDictionaryFeatureProvider(dictionary: dict)
    }

    private func multiArray(for desc: MLFeatureDescription, fill value: Double) -> MLMultiArray? {
        guard let shape = desc.multiArrayConstraint?.shape else { return nil }
        let dataType = desc.multiArrayConstraint?.dataType ?? .double
        guard let array = try? MLMultiArray(shape: shape, dataType: dataType) else { return nil }
        switch dataType {
        case .float32:
            let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
            for i in 0..<array.count {
                ptr[i] = Float32(value)
            }
        case .int32:
            let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
            for i in 0..<array.count {
                ptr[i] = Int32(value)
            }
        default:
            let ptr = array.dataPointer.bindMemory(to: Double.self, capacity: array.count)
            for i in 0..<array.count {
                ptr[i] = value
            }
        }
        return array
    }

    private func extractState(from provider: MLFeatureProvider, outputNames: [String]) -> MLFeatureProvider? {
        var dict: [String: MLFeatureValue] = [:]
        for name in outputNames {
            if let value = provider.featureValue(for: name) {
                dict[name] = value
            }
        }
        guard !dict.isEmpty else { return nil }
        return try? MLDictionaryFeatureProvider(dictionary: dict)
    }

    private func detectedObjects(from provider: MLFeatureProvider) -> [DetectedObject] {
        // Attempt to treat dictionary outputs as classification probabilities; otherwise fall back to empty.
        if let dictFeatureName = provider.featureNames.first(where: { provider.featureValue(for: $0)?.type == .dictionary }),
           let dict = provider.featureValue(for: dictFeatureName)?.dictionaryValue as? [String: NSNumber],
           let best = dict.max(by: { $0.value.doubleValue < $1.value.doubleValue }) {
            let object = DetectedObject(
                id: UUID(),
                label: best.key,
                confidence: String(format: "%.3f", best.value.doubleValue),
                otherLabels: dict.map { (label: $0.key, confidence: String(format: "%.4f", $0.value.doubleValue)) },
                width: 0.9,
                height: 0.85,
                x: 0.05,
                y: 0.05,
                isClassification: true
            )
            return [object]
        }

        return []
    }

    func asDetectedObjects(visionObservationResults: [VNObservation]?, detectionTime: Duration) -> detectionOutput {
        let classificationObservations = visionObservationResults as? [VNClassificationObservation]
        let objectObservations = visionObservationResults as? [VNRecognizedObjectObservation]

        var detectedObjects: [DetectedObject] = []
        let seconds = Double(detectionTime.components.seconds) + (Double(detectionTime.components.attoseconds) / 1_000_000_000_000_000_000)
        let msTime = String(format: "%.0f ms", seconds * 1000)
        let detectionFPS = seconds > 0 ? String(format: "%.0f", 1.0 / seconds) : "0"
        
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

        let hasImageInput = inputs.contains { $0.type == .image && $0.imageConstraint != nil }
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
        // If a model explicitly set a crop preference without idealFormat, honor it.
        if let stored = CoreMLModel.sharedCropAndScale, CoreMLModel.sharedIdealFormat == nil {
            return stored
        }
        if let format = CoreMLModel.sharedIdealFormat {
            return format.width == format.height ? .centerCrop : .scaleFit
        }
        // fallback to any stored preference or default
        if let stored = CoreMLModel.sharedCropAndScale {
            return stored
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

extension VNRecognizedObjectObservation: @retroactive Identifiable {
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
    /// Last observed state outputs (testing only, stateful models).
    static var sharedLastStateValues: [String: MLFeatureValue]?
}
#endif
