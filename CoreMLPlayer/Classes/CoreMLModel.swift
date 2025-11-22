//
//  CoreMLModel.swift
//  CoreMLPlayer
//
//  Created by NA on 1/21/23.
//

import SwiftUI
import CoreML
import Vision
import CoreVideo

class CoreMLModel: Base, ObservableObject {
    /// Shared ideal format for crop/pixel decisions across app; updated when a model loads.
    static var sharedIdealFormat: (width: Int, height: Int, type: OSType)?
    /// Shared selection for multi-function models (test hook).
    static var sharedSelectedFunction: String?
    /// Shared crop/scaling preference derived from the model input constraints.
    static var sharedCropAndScale: VNImageCropAndScaleOption?

    enum ModelKind: String, CaseIterable, Identifiable {
        case detector
        case classifier
        case embedding
        case unknown

        var id: String { rawValue }
    }

    @Published var isValid = false
    @Published var isLoading = false
    @Published var name: String?
    @Published var availableFunctions: [String] = []
    @Published var selectedFunction: String? {
        didSet {
            CoreMLModel.sharedSelectedFunction = selectedFunction
            storedSelectedFunctionName = selectedFunction
            if isValid, selectedFunction != oldValue {
                reconfigure()
            }
        }
    }
    @Published var modelKind: ModelKind = .unknown {
        didSet { storedModelKind = modelKind.rawValue }
    }
    @Published var cropAndScaleOption: VNImageCropAndScaleOption = .scaleFill {
        didSet { CoreMLModel.sharedCropAndScale = cropAndScaleOption }
    }
    @Published var supportsStatefulModel: Bool = false
    @Published var optimizationWarning: String?
    
    @AppStorage("CoreMLModel-selectedBuiltInModel") var selectedBuiltInModel: String?
    @AppStorage("CoreMLModel-autoloadSelection") var autoloadSelection: AutoloadChoices = .disabled
    @AppStorage("CoreMLModel-bookmarkData") var bookmarkData: Data?
    @AppStorage("CoreMLModel-originalModelURL") var originalModelURL: URL?
    @AppStorage("CoreMLModel-compiledModelURL") var compiledModelURL: URL?
    @AppStorage("CoreMLModel-computeUnits") var computeUnits: MLComputeUnits = .all
    @AppStorage("CoreMLModel-gpuAllowLowPrecision") var gpuAllowLowPrecision: Bool = false
    @AppStorage("CoreMLModel-allowBackgroundTasks") var allowBackgroundTasks: Bool = true
    @AppStorage("CoreMLModel-optimizeOnLoad") var optimizeOnLoad: Bool = false
    @AppStorage("CoreMLModel-selectedFunctionName") var storedSelectedFunctionName: String?
    @AppStorage("CoreMLModel-modelKind") private var storedModelKind: String = ModelKind.unknown.rawValue
    
    var model: VNCoreMLModel?
    var modelDescription: [ModelDescription] = []
    var idealFormat: (width: Int, height: Int, type: OSType)? {
        didSet {
            CoreMLModel.sharedIdealFormat = idealFormat
        }
    }
    @Published var wasOptimized: Bool = false

    override init() {
        super.init()
        modelKind = ModelKind(rawValue: storedModelKind) ?? .unknown
        CoreMLModel.sharedSelectedFunction = storedSelectedFunctionName
        CoreMLModel.sharedCropAndScale = cropAndScaleOption

        if UserDefaults.standard.object(forKey: "CoreMLModel-computeUnits") == nil {
            #if arch(x86_64)
            computeUnits = .cpuAndGPU
            #else
            computeUnits = .all
            #endif
        }

        if let storedSelectedFunctionName {
            selectedFunction = storedSelectedFunctionName
        }

        optimizationWarning = nil
    }
    
    enum AutoloadChoices: String, CaseIterable, Identifiable {
        case disabled = "Disabled"
        case reloadCompiled = "Reload compiled cache" // Available until system reboot/shutdown
        case recompile = "Compile again"
        
        var id: String { self.rawValue }
    }
    
//    var modelType: CMPModelTypes = .unacceptable
//    enum CMPModelTypes: String {
//        case imageObjectDetection = "Object Detection"
//        case imageClassification = "Classification"
//        case unacceptable = "Unacceptable"
//    }
    
    func autoload() {
        switch autoloadSelection {
        case .disabled:
            bookmarkData = nil
            selectedBuiltInModel = nil
            return
        case .reloadCompiled:
            if let selectedBuiltInModel {
                loadBuiltInModel(name: selectedBuiltInModel)
            } else if let compiledModelURL {
                if FileManager.default.fileExists(atPath: compiledModelURL.path) {
                    loadTheModel(url: compiledModelURL)
                } else {
                    fallthrough // Fall Through to recompile from Bookmark
                }
            }
        case .recompile:
            if let selectedBuiltInModel {
                loadBuiltInModel(name: selectedBuiltInModel)
            } else if let url = loadBookmark() {
                loadTheModel(url: url, useSecurityScope: true)
            }
        }
    }
    
    func loadBuiltInModel(name: String) {
        if let builtInModelURL = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            selectedBuiltInModel = name
            loadTheModel(url: builtInModelURL, useSecurityScope: false)
        } else {
            selectedBuiltInModel = nil
            showAlert(title: "Failed to load built-in model (\(name)) from app bundle!")
        }
    }
    
    func reconfigure() {
        if let compiledModelURL {
            loadTheModel(url: compiledModelURL, useSecurityScope: true)
        }
    }
    
    func bookmarkModel() {
        if loadBookmark() == originalModelURL {
            return
        } else if let modelUrl = originalModelURL, autoloadSelection != .disabled {
            saveBookmark(modelUrl)
        }
    }
    
    func loadTheModel(url: URL, useSecurityScope: Bool = false) {
        DispatchQueue.main.async {
            self.isLoading = true
        }
        DispatchQueue.global(qos: .userInitiated).async {
            var hasScope = false
            DispatchQueue.main.async {
                self.optimizationWarning = nil
            }
            if useSecurityScope {
                hasScope = url.startAccessingSecurityScopedResource()
            }

            defer {
                if hasScope {
                    url.stopAccessingSecurityScopedResource()
                }
            }

            do {
                // Identify function names before load so we can honor selection when building the configuration.
                let functions = self.functionNames(from: url)
                let selectedFn: String? = {
                    if let current = self.selectedFunction, functions.isEmpty || functions.contains(current) { return current }
                    if let stored = self.storedSelectedFunctionName, functions.isEmpty || functions.contains(stored) { return stored }
                    return functions.first
                }()

                let (sourceURL, isCompiled, optimizedFlag) = try self.prepareSourceURL(for: url)
                let compiledURL = try self.compileModelIfNeeded(sourceURL: sourceURL, isAlreadyCompiled: isCompiled)

                let config = self.makeConfiguration(selectedFunction: selectedFn)
                let mlModel = try self.loadModel(at: compiledURL, configuration: config)
                try super.checkModelIO(modelDescription: mlModel.modelDescription)

                let inferredKind = self.inferModelKind(from: mlModel.modelDescription)
                let ideal = self.extractIdealFormat(from: mlModel.modelDescription)
                let crop = self.deriveCropAndScale(from: ideal)
                let stateful = self.detectStateful(from: mlModel.modelDescription)

                let vnModel = try VNCoreMLModel(for: mlModel)
                Base.register(mlModel: mlModel, for: vnModel)
                self.performWarmupIfPossible(vnModel: vnModel, ideal: ideal, crop: crop, functionName: selectedFn)

                DispatchQueue.main.async {
                    self.wasOptimized = optimizedFlag || (self.optimizeOnLoad && isCompiled)
                    if !isCompiled && !useSecurityScope {
                        self.originalModelURL = url
                        self.bookmarkModel()
                    }
                    self.compiledModelURL = compiledURL
                    self.model = vnModel
                    self.setModelDescriptionInfo(mlModel.modelDescription)
                    self.idealFormat = ideal
                    self.cropAndScaleOption = crop
                    self.modelKind = inferredKind
                    self.supportsStatefulModel = stateful
                    self.availableFunctions = functions
                    self.selectedFunction = selectedFn
                    self.name = url.lastPathComponent
                    withAnimation {
                        self.isValid = true
                        self.isLoading = false
                    }
                    if let warning = self.optimizationWarning,
                       ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] == nil {
                        self.showAlert(title: "Optimization skipped", message: warning)
                    }
                }
            } catch {
                #if DEBUG
                print(error)
                #endif
                DispatchQueue.main.async {
                    self.unSelectModel()
                    super.showAlert(title: "Failed to compile/initiate your MLModel!")
                }
            }
        }
    }
    
    func unSelectModel() {
        autoloadSelection = .disabled
        originalModelURL = nil
        compiledModelURL = nil
        selectedBuiltInModel = nil
        modelDescription = []
        bookmarkData = nil
        model = nil
        name = nil
        selectedFunction = nil
        storedSelectedFunctionName = nil
        availableFunctions = []
        modelKind = .unknown
        cropAndScaleOption = .scaleFill
        supportsStatefulModel = false
        optimizationWarning = nil
        CoreMLModel.sharedIdealFormat = nil
        CoreMLModel.sharedCropAndScale = nil
        CoreMLModel.sharedSelectedFunction = nil
        withAnimation {
            isValid = false
            isLoading = false
        }
    }

    @MainActor
    func recordOptimizationWarning(_ message: String) {
        optimizationWarning = message
    }
    
    func getModelURLString() -> (original: (file: String, directory: String), compiled: (file: String, directory: String)) {
        var originalFile = ""
        var originalDirectory = ""
        var compiledFile = ""
        var compiledDirectory = ""
        
        if let url = originalModelURL {
            originalFile = String(url.path)
            originalDirectory = url.deletingLastPathComponent().path()
        }
        
        if let url = compiledModelURL {
            compiledFile = String(url.path)
            compiledDirectory = url.deletingLastPathComponent().path()
        }
        
        return (original: (file: originalFile, directory: originalDirectory), compiled: (file: compiledFile, directory: compiledDirectory))
    }
    
    func selectCoreMLModel() {
        let file = super.selectFiles(contentTypes: K.CoreMLModel.contentTypes, multipleSelection: false)
        
        guard let selectedFile = file?.first else { return }
        selectedBuiltInModel = nil
        loadTheModel(url: selectedFile)
    }
    
    func setModelDescriptionInfo(_ coreMLModelDescription: MLModelDescription?) { //  MLModelDescription.h
        var info: [ModelDescription] = []
        guard let description = coreMLModelDescription else {
            modelDescription = []
            return
        }
        
        var inputDescriptionItems: [ModelDescription.Item] = []
        for item in description.inputDescriptionsByName {
            inputDescriptionItems.append(ModelDescription.Item(key: item.key, value: "\(item.value)"))
            if let image = item.value.imageConstraint {
                idealFormat = (width: image.pixelsWide, height: image.pixelsHigh, type: image.pixelFormatType)
            }
        }
        info.append(ModelDescription(category: "Input Description", items: inputDescriptionItems))
        
        var outputDescriptionItems: [ModelDescription.Item] = []
        for item in description.outputDescriptionsByName {
            outputDescriptionItems.append(ModelDescription.Item(key: item.key, value: "\(item.value)"))
        }
        info.append(ModelDescription(category: "Output Description", items: outputDescriptionItems))
        
        var metaDataItems: [ModelDescription.Item] = []
        for metaData in description.metadata {
            let key = metaData.key.rawValue.replacingOccurrences(of: "Key", with: "")
            let value = String(describing: metaData.value)
            
            if key == "MLModelCreatorDefined", let creatorDefinedItems = description.metadata[MLModelMetadataKey.creatorDefinedKey] as? NSDictionary {
                for creatorDefined in creatorDefinedItems {
                    metaDataItems.append(ModelDescription.Item(key: "\(creatorDefined.key)", value: "\(creatorDefined.value)"))
                }
            } else {
                metaDataItems.append(ModelDescription.Item(key: key, value: value))
            }
        }
        info.append(ModelDescription(category: "MetaData", items: metaDataItems))

        if let predictedFeatureName = description.predictedFeatureName {
            info.append(ModelDescription(category: "Predicted Feature Name", items: [ModelDescription.Item(key: "predictedFeatureName", value: predictedFeatureName)]))
        }
        
        if let predictedProbabilitiesName = description.predictedProbabilitiesName {
            info.append(ModelDescription(category: "Predicted Probabilities Name", items: [ModelDescription.Item(key: "predictedProbabilitiesName", value: predictedProbabilitiesName)]))
        }
        
        if let classLabels = description.classLabels {
            var classLabelItems: [ModelDescription.Item] = []
            for item in classLabels {
                classLabelItems.append(ModelDescription.Item(key: "\(item)", value: ""))
            }
            info.append(ModelDescription(category: "Class Labels", items: classLabelItems))
        }

        modelDescription = info

        if let idealFormat {
            cropAndScaleOption = deriveCropAndScale(from: idealFormat)
        }
    }

    private func optimizedDestination(for url: URL) throws -> URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let optimizedDir = appSupport.appendingPathComponent("CoreMLPlayer/Optimized", isDirectory: true)
        try FileManager.default.createDirectory(at: optimizedDir, withIntermediateDirectories: true)
        let baseName = url.deletingPathExtension().lastPathComponent
        return optimizedDir.appendingPathComponent("\(baseName).optimized.mlmodel")
    }

    private func prepareSourceURL(for url: URL) throws -> (URL, Bool, Bool) {
        // If already compiled, just use as-is.
        if url.pathExtension == "mlmodelc" {
            return (url, true, false)
        }

        var sourceURL = url
        var optimized = false

        if optimizeOnLoad {
            do {
                let optimizedURL = try optimizedDestination(for: url)
                try? FileManager.default.removeItem(at: optimizedURL)

                let quantized = try optimizeModelIfPossible(source: url, destination: optimizedURL)
                if !quantized {
                    try FileManager.default.copyItem(at: url, to: optimizedURL)
                }

                if validateOptimizedCandidate(source: url, candidate: optimizedURL) {
                    sourceURL = optimizedURL
                    optimized = true
                } else {
                    try? FileManager.default.removeItem(at: optimizedURL)
                    optimized = false
                    sourceURL = url
                }
            } catch {
                #if DEBUG
                print("Optimization copy failed, falling back to original:", error)
                #endif
                optimized = false
                sourceURL = url
            }
        }
        return (sourceURL, false, optimized)
    }

    /// Attempt to quantize/palettize using coremltools when available. Returns true on success.
    @discardableResult
    func optimizeModelIfPossible(source: URL, destination: URL) throws -> Bool {
        let script = """
import sys, pathlib, traceback
src = pathlib.Path(r\"""\(source.path)\""")
dst = pathlib.Path(r\"""\(destination.path)\""")
try:
    import coremltools as ct
except ImportError:
    sys.exit(2)

try:
    ml = ct.models.MLModel(src)
    try:
        from coremltools.models.neural_network.quantization_utils import quantize_weights
        quantized = quantize_weights(ml, nbits=4, quantization_mode="linear")
    except Exception:
        try:
            quantized = ct.optimize.coreml.quantization(ml, mode="linear8")
        except Exception:
            quantized = ml
    quantized.save(dst)
except Exception:
    traceback.print_exc()
    sys.exit(3)
"""

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        proc.arguments = ["-c", script]
        let pipe = Pipe()
        proc.standardError = pipe
        proc.standardOutput = Pipe()
        try proc.run()
        proc.waitUntilExit()

        if proc.terminationStatus == 0, FileManager.default.fileExists(atPath: destination.path) {
            Task { @MainActor in self.optimizationWarning = nil }
            return true
        }

        let stderr = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        #if DEBUG
        print("coremltools optimization skipped/fell back:", stderr)
        #endif

        let warning: String
        switch proc.terminationStatus {
        case 2:
            warning = "coremltools is not installed; install it to enable Optimize on Load or turn the toggle off."
        case 3:
            warning = "coremltools failed to optimize this model; using the original copy instead."
        default:
            warning = "Optimization could not run (exit code \(proc.terminationStatus)); original model will be used."
        }

        Task { @MainActor in
            self.optimizationWarning = warning
        }
        return false
    }

    /// Ensure an optimized candidate is valid and no larger than the source.
    private func validateOptimizedCandidate(source: URL, candidate: URL) -> Bool {
        guard let sourceSize = fileSize(at: source),
              let candidateSize = fileSize(at: candidate),
              candidateSize <= sourceSize else {
            Task { @MainActor in self.optimizationWarning = "Optimized model was larger than the original; reverting to the original file." }
            return false
        }

        do {
            let compiled = try MLModel.compileModel(at: candidate)
            try? FileManager.default.removeItem(at: compiled)
            return true
        } catch {
            #if DEBUG
            print("Optimized candidate failed validation:", error)
            #endif
            Task { @MainActor in self.optimizationWarning = "Optimized model failed to compile; using the original model instead." }
            return false
        }
    }

    private func fileSize(at url: URL) -> Int64? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? NSNumber else {
            return nil
        }
        return size.int64Value
    }

    private func compileModelIfNeeded(sourceURL: URL, isAlreadyCompiled: Bool) throws -> URL {
        if isAlreadyCompiled { return sourceURL }
        return try MLModel.compileModel(at: sourceURL)
    }

    private func makeConfiguration(selectedFunction: String?) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        config.allowLowPrecisionAccumulationOnGPU = gpuAllowLowPrecision && computeUnits != .cpuOnly
        // Public symbol for allowsBackgroundTasks is not yet surfaced on macOS SDK; guard via selector to stay source-compatible.
        if config.responds(to: Selector(("setAllowsBackgroundTasks:"))) {
            config.setValue(allowBackgroundTasks, forKey: "allowsBackgroundTasks")
        }
        if let selectedFunction {
            if #available(macOS 15.0, *) {
                config.functionName = selectedFunction
            }
        }
        return config
    }

    private func loadModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
        var loadedModel: MLModel?
        var loadError: Error?

        let semaphore = DispatchSemaphore(value: 0)
        MLModel.load(contentsOf: url, configuration: configuration) { result in
            switch result {
            case .success(let model):
                loadedModel = model
            case .failure(let error):
                loadError = error
            }
            semaphore.signal()
        }
        semaphore.wait()

        if let model = loadedModel {
            return model
        }
        if let error = loadError {
            throw error
        }
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    private func extractIdealFormat(from description: MLModelDescription) -> (width: Int, height: Int, type: OSType)? {
        for item in description.inputDescriptionsByName.values {
            if let image = item.imageConstraint {
                return (width: image.pixelsWide, height: image.pixelsHigh, type: image.pixelFormatType)
            }
        }
        return nil
    }

    private func deriveCropAndScale(from ideal: (width: Int, height: Int, type: OSType)?) -> VNImageCropAndScaleOption {
        guard let ideal else { return .scaleFill }
        return ideal.width == ideal.height ? .centerCrop : .scaleFit
    }

    private func inferModelKind(from description: MLModelDescription) -> ModelKind {
        if description.predictedFeatureName != nil || description.predictedProbabilitiesName != nil {
            return .classifier
        }

        let outputs = Array(description.outputDescriptionsByName.values)
        let hasVectorOnly = outputs.allSatisfy { $0.type == .multiArray }
        if hasVectorOnly {
            return .embedding
        }

        let hasProbabilities = outputs.contains { $0.type == .dictionary || $0.type == .string }
        if hasProbabilities {
            return .classifier
        }

        let hasCoordinateLike = outputs.contains { desc in
            if let shape = desc.multiArrayConstraint?.shape, shape.count >= 3 {
                return true
            }
            return false
        }
        if hasCoordinateLike {
            return .detector
        }

        return .unknown
    }

    private func detectStateful(from description: MLModelDescription) -> Bool {
        let names = Array(description.inputDescriptionsByName.keys) + Array(description.outputDescriptionsByName.keys)
        return names.contains { $0.lowercased().contains("state") }
    }

    private func functionNames(from url: URL) -> [String] {
        let packageURL: URL
        switch url.pathExtension {
        case "mlpackage":
            packageURL = url
        case "mlmodelc", "mlmodel":
            let candidate = url.deletingPathExtension().appendingPathExtension("mlpackage")
            guard FileManager.default.fileExists(atPath: candidate.path) else { return [] }
            packageURL = candidate
        default:
            return []
        }

        let manifest = packageURL.appendingPathComponent("Manifest.json")
        guard let data = try? Data(contentsOf: manifest),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let fn = json["functions"] as? [[String: Any]] else {
            return []
        }
        return fn.compactMap { $0["name"] as? String }
    }

    private func performWarmupIfPossible(vnModel: VNCoreMLModel, ideal: (width: Int, height: Int, type: OSType)?, crop: VNImageCropAndScaleOption, functionName: String?) {
        let size = ideal.map { ($0.width, $0.height, $0.type) } ?? (224, 224, kCVPixelFormatType_32BGRA)
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, size.0, size.1, size.2, nil, &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return }

        CVPixelBufferLockBaseAddress(buffer, [])
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            let length = CVPixelBufferGetDataSize(buffer)
            // Fill with a neutral mid-gray to better resemble real input.
            memset(base, 0x7F, length)
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])

        DispatchQueue.global(qos: .utility).async {
            _ = self.performObjectDetection(pixelBuffer: buffer, orientation: .up, vnModel: vnModel, functionName: functionName, cropAndScale: crop)
        }
    }
    
    func saveBookmark(_ url: URL) {
        do {
            bookmarkData = try url.bookmarkData(
                options: [.securityScopeAllowOnlyReadAccess, .withSecurityScope],
                includingResourceValuesForKeys: nil,
                relativeTo: nil
            )
        } catch {
            #if DEBUG
            print("Failed to save bookmark data for \(url)", error)
            #endif
        }
    }
    
    func loadBookmark() -> URL? {
        guard let data = bookmarkData else { return nil }
        
        do {
            var isStale = false
            let url = try URL(
                resolvingBookmarkData: data,
                options: .withSecurityScope,
                relativeTo: nil,
                bookmarkDataIsStale: &isStale
            )
            if isStale {
                saveBookmark(url)
            }
            return url
        } catch {
            #if DEBUG
            print("Error resolving bookmark:", error)
            #endif
            return nil
        }
    }
}
