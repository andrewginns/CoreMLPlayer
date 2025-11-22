//
//  VideoDetection.swift
//  CoreML Player
//
//  Created by NA on 1/30/23.
//

import CoreML
import AVKit
import Vision
import Combine
import ImageIO

class VideoDetection: Base, ObservableObject {
    @Published var playMode = PlayModes.normal {
        didSet {
            playing = false
        }
    }
    @Published var playing = false {
        didSet {
            playManager()
        }
    }
    @Published private(set) var frameObjects: [DetectedObject] = []
    @Published private(set) var videoInfo: (isPlayable: Bool, frameRate: Double, duration: CMTime, size: CGSize) = (false, 30, .zero, .zero)
    @Published private(set) var player: AVPlayer?
    @Published private(set) var canStart = false
    @Published private(set) var errorMessage: String?
    
    private var model: VNCoreMLModel?
    private var fpsCounter = 0
    private var fpsDisplay = 0
    private var chartDuration: Duration = .zero
    private var playerOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: nil)
    private var videoOutputAttributes: [String: Any]?
    private var timeTracker = DispatchTime.now()
    private var lastDetectionTime: Double = 0
    private var videoHasEnded = false
    private var idealFormat: (width: Int, height: Int, type: OSType)?
    private var videoOrientation: CGImagePropertyOrientation = .up
    private var stateFrameCounter: Int = 0
    private var droppedFrames: Int = 0
    private var warmupCompleted = false
    
    var videoURL: URL? {
        didSet {
            let url = videoURL
            Task { [weak self, url] in
                await self?.prepareToPlay(videoURL: url)
            }
        }
    }
    
    private var avPlayerItemStatus: AVPlayerItem.Status = .unknown {
        didSet {
            if avPlayerItemStatus == .readyToPlay {
                canStart = true
                #if DEBUG
                print("PlayerItem is readyToPlay")
                #endif
            } else {
                canStart = false
            }
        }
    }
    
    private var avPlayerTimeControlStatus: AVPlayer.TimeControlStatus = .paused {
        didSet {
            if avPlayerTimeControlStatus != oldValue, playMode != .maxFPS {
                DispatchQueue.main.async {
                    switch self.avPlayerTimeControlStatus {
                    case .playing:
                        self.playing = true
                    default:
                        self.playing = false
                    }
                }
            }
        }
    }
    
    enum PlayModes {
        case maxFPS
        case normal
    }
    
    func disappearing() {
        playing = false
        frameObjects = []
        stateFrameCounter = 0
        droppedFrames = 0
        warmupCompleted = false
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            DetectionStats.shared.items = []
        }
    }
    
    func setModel(_ vnModel: VNCoreMLModel?) {
        model = vnModel
        stateFrameCounter = 0
        droppedFrames = 0
        warmupCompleted = false
    }

    func setIdealFormat(_ format: (width: Int, height: Int, type: OSType)?) {
        idealFormat = format
        configureVideoOutputIfNeeded(attachedTo: player?.currentItem)
    }
    
    func playManager() {
        if let playerItem = player?.currentItem, playerItem.currentTime() >= playerItem.duration {
            player?.seek(to: CMTime.zero)
            DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + 0.1) {
                self.detectObjectsInFrame()
            }
            videoHasEnded = true
            frameObjects = []
        }
        
        if playing {
            if(videoHasEnded) {
                FPSChart.shared.reset()
            }
            videoHasEnded = false
            switch playMode {
            case .maxFPS:
                startMaxFPSDetection()
            default:
                startNormalDetection()
                player?.play()
            }
        } else {
            player?.pause()
        }
    }
    
    func seek(steps: Int) {
        guard let playerItem = player?.currentItem else {
            return
        }
        
        playerItem.step(byCount: steps)
        DispatchQueue.global(qos: .userInitiated).async {
            self.detectObjectsInFrame()
        }
    }
    
    func getRepeatInterval(_ reduceLastDetectionTime: Bool = true) -> Double {
        let nominalFrameInterval = videoInfo.frameRate > 0 ? (1.0 / videoInfo.frameRate) : (1.0 / 30.0)

        guard reduceLastDetectionTime else {
            return nominalFrameInterval
        }

        let minInterval = max(0.02, lastDetectionTime * 0.5)
        let maxInterval = max(nominalFrameInterval + lastDetectionTime, nominalFrameInterval * 2)
        let adjusted = nominalFrameInterval - lastDetectionTime

        return min(max(adjusted, minInterval), maxInterval)
    }
    
    /// Simple guard to verify detection time stays within a budget (defaults to 50ms).
    func isWithinLatencyBudget(budgetMs: Double = 50) -> Bool {
        return (lastDetectionTime * 1000) <= budgetMs
    }

    private func configureVideoOutputIfNeeded(attachedTo playerItem: AVPlayerItem? = nil) {
        let oldOutput = playerOutput
        var attrs: [String: Any]? = nil
        if let idealFormat {
            attrs = [
                kCVPixelBufferPixelFormatTypeKey as String: idealFormat.type,
                kCVPixelBufferWidthKey as String: idealFormat.width,
                kCVPixelBufferHeightKey as String: idealFormat.height
            ]
        }
        videoOutputAttributes = attrs
        playerOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: attrs)

        if let item = playerItem {
            item.remove(oldOutput)
            item.add(playerOutput)
        }
    }
    
    @MainActor
    func prepareToPlay(videoURL: URL?) async {
        guard let url = videoURL,
              url.isFileURL,
              let isReachable = try? url.checkResourceIsReachable(),
              isReachable
        else {
            return
        }
        
        let asset = AVAsset(url: url)
        
        do {
            if let videoTrack = try await asset.loadTracks(withMediaType: .video).first
            {
                let (frameRate, size) = try await videoTrack.load(.nominalFrameRate, .naturalSize)
                let transform = try await videoTrack.load(.preferredTransform)
                let (isPlayable, duration) = try await asset.load(.isPlayable, .duration)
                let playerItem = AVPlayerItem(asset: asset)
                configureVideoOutputIfNeeded()
                playerItem.add(playerOutput)
                self.videoOrientation = Self.orientation(from: transform)
                
                DispatchQueue.main.async {
                    self.videoInfo.frameRate = Double(frameRate)
                    self.videoInfo.duration = duration
                    self.videoInfo.size = size
                    if isPlayable {
                        self.player = AVPlayer(playerItem: playerItem)
                        self.videoInfo.isPlayable = true
                        
                        // Set avPlayerItemStatus when playerItem.status changes, when it is readyToPlay avPlayerItemStatus will set canStart to true
                        let playerItemStatusPublisher = playerItem.publisher(for: \.status)
                        let playerItemStatusSubscriber = Subscribers.Assign(object: self, keyPath: \.avPlayerItemStatus)
                        playerItemStatusPublisher.receive(subscriber: playerItemStatusSubscriber)
                        // AVPlayer.TimeControlStatus
                        let timeControlStatusPublisher = self.player?.publisher(for: \.timeControlStatus)
                        let timeControlStatusSubscriber = Subscribers.Assign(object: self, keyPath: \.avPlayerTimeControlStatus)
                        timeControlStatusPublisher?.receive(subscriber: timeControlStatusSubscriber)
                    } else {
                        self.errorMessage = "Video item is not playable."
                    }
                }
            }
        } catch {
            self.videoInfo.isPlayable = false
            self.errorMessage = "There was an error trying to load asset."
            #if DEBUG
            print("Error: \(error)")
            #endif
        }
    }

    private static func orientation(from transform: CGAffineTransform) -> CGImagePropertyOrientation {
        // Derived from AVAssetTrack preferredTransform conventions
        switch (transform.a, transform.b, transform.c, transform.d) {
        case (0, 1, -1, 0):
            return .right
        case (0, -1, 1, 0):
            return .left
        case (1, 0, 0, 1):
            return .up
        case (-1, 0, 0, -1):
            return .down
        default:
            return .up
        }
    }
    
    func getPlayerItemIfContinuing(mode: PlayModes) -> AVPlayerItem? {
        guard let playerItem = player?.currentItem,
              playing == true,
              playMode == mode
        else {
            return nil
        }
        
        if playerItem.currentTime() >= playerItem.duration {
            DispatchQueue.main.async {
                self.playing = false
            }
            
            return nil
        }
        
        return playerItem
    }
        
    func startNormalDetection() {
        guard getPlayerItemIfContinuing(mode: .normal) != nil else {
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            self.detectObjectsInFrame() {
                DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + self.getRepeatInterval()) { [weak self] in
                    self?.startNormalDetection()
                }
            }
        }
    }
    
    func startMaxFPSDetection() {
        guard let playerItem = getPlayerItemIfContinuing(mode: .maxFPS) else {
            return
        }
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            playerItem.step(byCount: 1)
            self?.detectObjectsInFrame() {
                self?.startMaxFPSDetection()
            }
        }
    }
    
    func detectObjectsInFrame(completion: (() -> ())? = nil) {
        guard let model else { completion?(); return }
        guard let pixelBuffer = getPixelBuffer() else {
            recordDroppedFrame()
            completion?()
            return
        }
        
        // Process the frame
        #if DEBUG
        VideoDetection.sharedLastVideoOrientation = videoOrientation
        #endif
        let cropOption = cropOptionForIdealFormat()
        let detectionResult = performObjectDetection(pixelBuffer: pixelBuffer, orientation: videoOrientation, vnModel: model, functionName: CoreMLModel.sharedSelectedFunction, cropAndScale: cropOption)
        
        DispatchQueue.main.async {
            let isWarmup = self.warmupCompleted == false
            self.warmupCompleted = true
            self.frameObjects = detectionResult.objects
            self.fpsCounter += 1
            let timePassed = DispatchTime.now().uptimeNanoseconds - self.timeTracker.uptimeNanoseconds
            if timePassed >= 1_000_000_000 && !isWarmup {
                self.chartDuration += .seconds(1)
                if let detFPSDouble = Double(detectionResult.detectionFPS),
                   self.playMode == .maxFPS
                {
                    let narrowDuration = self.chartDuration.formatted(.units(allowed: [.seconds], width: .narrow))
                    FPSChart.shared.data.append(contentsOf: [
                        FPSChartData(name: "FPS", time: narrowDuration, value: Double(self.fpsCounter)),
                        FPSChartData(name: "Det. FPS", time: narrowDuration, value: detFPSDouble)
                    ])
                }
                self.timeTracker = DispatchTime.now()
                self.fpsDisplay = self.fpsCounter
                self.fpsCounter = 0
                #if DEBUG
                if self.playMode != .maxFPS {
                    print(self.fpsDisplay)
                }
                #endif
            }
            
            var stats: [Stats] = []
            
            if self.playMode == .maxFPS && !isWarmup {
                stats.append(Stats(key: "FPS", value: "\(self.fpsDisplay)"))
                stats.append(Stats(key: "Det. FPS", value: "\(detectionResult.detectionFPS)"))
            }
            
            let detTime = Double(detectionResult.detectionTime.replacingOccurrences(of: " ms", with: "")) ?? 0
            self.lastDetectionTime = detTime / 1000
            self.stateFrameCounter += 1
            
            if !isWarmup {
                stats += [
                    Stats(key: "Det. Objects", value: "\(detectionResult.objects.count)"),
                    Stats(key: "Det. Time", value: "\(detectionResult.detectionTime)"),
                    Stats(key: "Dropped Frames", value: "\(self.droppedFrames)"),
                    Stats(key: "-", value: ""), // Divider
                    Stats(key: "Width", value: "\(self.videoInfo.size.width)"),
                    Stats(key: "Height", value: "\(self.videoInfo.size.height)")
                ]
            }
            
            if !stats.isEmpty {
                DetectionStats.shared.addMultiple(stats)
            }
            
            if completion != nil {
                completion!()
            }
        }
    }
    
    func getUnixTimestampInt() -> Int {
        return Int(Date().timeIntervalSince1970)
    }
    
    func getPixelBuffer() -> CVPixelBuffer? {
        if let currentTime = player?.currentTime() {
            if let buffer = playerOutput.copyPixelBuffer(forItemTime: currentTime, itemTimeForDisplay: nil) {
                return buffer
            }

            // Fallback: if ideal format was too strict, downgrade to default BGRA
            if videoOutputAttributes != nil, let item = player?.currentItem {
                let reattach = {
                    item.remove(self.playerOutput)
                    self.playerOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: nil)
                    self.videoOutputAttributes = nil
                    item.add(self.playerOutput)
                }
                if Thread.isMainThread {
                    reattach()
                } else {
                    DispatchQueue.main.sync { reattach() }
                }
                return playerOutput.copyPixelBuffer(forItemTime: currentTime, itemTimeForDisplay: nil)
            }
        }
        
        return nil
    }
    
    private func recordDroppedFrame() {
        droppedFrames += 1
        if warmupCompleted {
            DetectionStats.shared.addMultiple([
                Stats(key: "Dropped Frames", value: "\(droppedFrames)")
            ])
        }
    }
}

#if DEBUG
extension VideoDetection {
    /// Test-only helper to allow deterministic configuration without relying on async player setup.
    func setVideoInfoForTesting(_ info: (isPlayable: Bool, frameRate: Double, duration: CMTime, size: CGSize)) {
        videoInfo = info
    }

    /// Test-only helper to inject a last detection duration when verifying scheduling behavior.
    func setLastDetectionTimeForTesting(_ value: Double) {
        lastDetectionTime = value
    }

    /// Test-only helper to set ideal format.
    func setIdealFormatForTesting(_ format: (width: Int, height: Int, type: OSType)?) {
        setIdealFormat(format)
    }

    /// Test-only helper to set orientation.
    func setVideoOrientationForTesting(_ orientation: CGImagePropertyOrientation) {
        videoOrientation = orientation
    }

    static var sharedLastVideoOrientation: CGImagePropertyOrientation?
    /// Optional override buffer to drive fallback tests.
    static var testFallbackPixelBuffer: CVPixelBuffer?

    func getPixelBufferAttributesForTesting() -> [String: Any]? {
        return videoOutputAttributes
    }

    /// Test-only helper to run detection on a supplied pixel buffer, returning the state counter.
    func detectPixelBufferForTesting(_ pixelBuffer: CVPixelBuffer) -> (objects: [DetectedObject], stateFrameCounter: Int) {
        guard let model else { return ([], stateFrameCounter) }
        VideoDetection.sharedLastVideoOrientation = videoOrientation
        let result = performObjectDetection(pixelBuffer: pixelBuffer, orientation: videoOrientation, vnModel: model, functionName: CoreMLModel.sharedSelectedFunction, cropAndScale: cropOptionForIdealFormat())
        stateFrameCounter += 1
        return (result.objects, stateFrameCounter)
    }

    /// Force the pixel-buffer fallback path without needing an AVPlayer instance.
    func forcePixelBufferFallbackForTesting() -> CVPixelBuffer? {
        guard videoOutputAttributes != nil else { return nil }
        videoOutputAttributes = nil
        playerOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: nil)
        return VideoDetection.testFallbackPixelBuffer
    }
    
    /// Expose internal counters for tests.
    func metricsForTesting() -> (droppedFrames: Int, warmupCompleted: Bool, lastDetectionTime: Double, stateFrameCounter: Int) {
        return (droppedFrames, warmupCompleted, lastDetectionTime, stateFrameCounter)
    }
}
#endif
