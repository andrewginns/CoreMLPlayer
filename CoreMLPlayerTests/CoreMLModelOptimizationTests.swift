import XCTest
import CoreML
@testable import CoreML_Player

/// Tests that exercise CoreMLModel's optimize-on-load behaviors and related fallbacks.
final class CoreMLModelOptimizationTests: XCTestCase {
    private func rawModelURL() throws -> URL {
        let bundle = Bundle(for: type(of: self))
        if let raw = bundle.url(forResource: "YOLOv3Tiny", withExtension: "mlmodel") {
            return raw
        }
        throw XCTSkip("Raw YOLOv3Tiny.mlmodel not bundled; optimization tests skipped.")
    }

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

    func testOptimizeOnLoadProducesNonLargerCopyAndValidModel() throws {
        let source = try rawModelURL()
        let sourceSize = try fileSize(at: source)

        let sut = CoreMLModel()
        sut.optimizeOnLoad = true
        sut.computeUnits = .cpuOnly // deterministic

        sut.loadTheModel(url: source)

        let exp = expectation(description: "model loads")
        DispatchQueue.main.asyncAfter(deadline: .now() + 6.0) {
            if sut.isValid {
                exp.fulfill()
            }
        }
        wait(for: [exp], timeout: 10.0)

        let optimizedURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("CoreMLPlayer/Optimized/\(source.deletingPathExtension().lastPathComponent).optimized.mlmodel")
        XCTAssertTrue(FileManager.default.fileExists(atPath: optimizedURL.path))

        let optimizedSize = try fileSize(at: optimizedURL)
        XCTAssertLessThanOrEqual(optimizedSize, sourceSize, "Optimized copy should not exceed original size")
        XCTAssertTrue(sut.wasOptimized)
        XCTAssertTrue(sut.isValid)
    }

    func testOptimizeOnLoadFallsBackWhenCandidateInvalid() throws {
        final class InvalidOptimizingModel: CoreMLModel {
            override func optimizeModelIfPossible(source: URL, destination: URL) throws -> Bool {
                // Write an invalid payload to force validation failure.
                let data = Data(repeating: 0xFF, count: 1024)
                try data.write(to: destination)
                return true
            }
        }

        let source = try rawModelURL()
        let sut = InvalidOptimizingModel()
        sut.optimizeOnLoad = true
        sut.computeUnits = .cpuOnly

        sut.loadTheModel(url: source)

        let exp = expectation(description: "model falls back after invalid optimization")
        DispatchQueue.main.asyncAfter(deadline: .now() + 6.0) {
            if sut.isValid {
                exp.fulfill()
            }
        }
        wait(for: [exp], timeout: 10.0)

        XCTAssertFalse(sut.wasOptimized)

        let optimizedURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("CoreMLPlayer/Optimized/\(source.deletingPathExtension().lastPathComponent).optimized.mlmodel")
        XCTAssertFalse(FileManager.default.fileExists(atPath: optimizedURL.path), "Invalid candidate should be removed")
    }

    func testOptimizeOnLoadSurfacesMissingToolchainWarning() throws {
        final class MissingToolModel: CoreMLModel {
            override func optimizeModelIfPossible(source: URL, destination: URL) throws -> Bool {
                Task { @MainActor in self.optimizationWarning = "coremltools is not installed; install it to enable Optimize on Load or turn the toggle off." }
                return false
            }
        }

        let source = try rawModelURL()
        let sut = MissingToolModel()
        sut.optimizeOnLoad = true
        sut.computeUnits = .cpuOnly

        sut.loadTheModel(url: source)

        let exp = expectation(description: "warning surfaced")
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            if sut.optimizationWarning != nil {
                exp.fulfill()
            }
        }
        wait(for: [exp], timeout: 5.0)

        XCTAssertEqual(sut.optimizationWarning, "coremltools is not installed; install it to enable Optimize on Load or turn the toggle off.")
    }

    func testOptimizeOnLoadShrinksModelWhenCoremltoolsPresent() throws {
        guard coremltoolsAvailable() else { throw XCTSkip("coremltools unavailable") }

        let source = try rawModelURL()
        let sourceSize = try fileSize(at: source)

        let sut = CoreMLModel()
        sut.optimizeOnLoad = true
        sut.computeUnits = .cpuOnly
        sut.loadTheModel(url: source)

        let exp = expectation(description: "model loads optimized")
        DispatchQueue.main.asyncAfter(deadline: .now() + 8.0) {
            if sut.isValid {
                exp.fulfill()
            }
        }
        wait(for: [exp], timeout: 12.0)

        let optimizedURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("CoreMLPlayer/Optimized/\(source.deletingPathExtension().lastPathComponent).optimized.mlmodel")

        let optimizedSize = try fileSize(at: optimizedURL)
        XCTAssertLessThan(optimizedSize, sourceSize, "Optimized model should be smaller after quantization")
        XCTAssertTrue(sut.wasOptimized)
        XCTAssertTrue(sut.isValid)
    }

    // MARK: - Helpers
    private func fileSize(at url: URL) throws -> Int64 {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs[.size] as? NSNumber)?.int64Value ?? 0
    }

    private func coremltoolsAvailable() -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = ["-c", "import coremltools"]
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }
}
