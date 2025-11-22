import Foundation
import XCTest
import CoreML

/// Utility to lazily generate lightweight Core ML fixtures (multi-function, stateful) using Python + coremltools when available.
/// Tests call these helpers and skip gracefully if tooling or platform support is missing.
enum FixtureBuilder {
    enum FixtureError: Error {
        case generationFailed(String)
    }

    /// Creates (or reuses) a tiny multi-function model and returns the compiled URL plus function names.
    static func ensureMultiFunctionModel() throws -> (compiledURL: URL, functionNames: [String]) {
        let tmpDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("CMPFixtures", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        let modelURL = tmpDir.appendingPathComponent("multifunction.mlpackage")
        let compiledURL = tmpDir.appendingPathComponent("multifunction.mlmodelc")

        if !FileManager.default.fileExists(atPath: compiledURL.path) {
            try generateFixture(kind: .multifunction, outputURL: modelURL)
            _ = try compileModel(at: modelURL, compiledURL: compiledURL)
        }

        // Probe function names from the package manifest (simple JSON read)
        let manifestURL = modelURL.appendingPathComponent("Manifest.json")
        let data = try Data(contentsOf: manifestURL)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let fnames = (json?["functions"] as? [[String: Any]])?.compactMap { $0["name"] as? String } ?? []
        guard !fnames.isEmpty else {
            throw XCTSkip("Multi-function fixture manifest missing function names; likely not supported on this platform.")
        }
        return (compiledURL, fnames)
    }

    /// Creates (or reuses) a tiny stateful model; returns compiled URL.
    static func ensureStatefulModel() throws -> URL {
        let tmpDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("CMPFixtures", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        let modelURL = tmpDir.appendingPathComponent("stateful.mlpackage")
        let compiledURL = tmpDir.appendingPathComponent("stateful.mlmodelc")

        if !FileManager.default.fileExists(atPath: compiledURL.path) {
            try generateFixture(kind: .stateful, outputURL: modelURL)
            _ = try compileModel(at: modelURL, compiledURL: compiledURL)
        }
        return compiledURL
    }

    private enum FixtureKind: String {
        case multifunction
        case stateful
    }

    /// Runs a tiny Python script that builds the requested model via coremltools.
    private static func generateFixture(kind: FixtureKind, outputURL: URL) throws {
        let script: String
        switch kind {
        case .multifunction:
            script = """
import sys, json, pathlib
try:
    import coremltools as ct
except ImportError:
    sys.exit(2)

out = pathlib.Path(r\"""\(outputURL.path)\""")
out.parent.mkdir(parents=True, exist_ok=True)

# Build two trivial functions (y = x and y = x + 1)
def make_model(offset):
    from coremltools.models import neural_network as nn
    from coremltools.models import datatypes
    input_features = [("x", datatypes.Array(1))]
    output_features = [("y", datatypes.Array(1))]
    builder = nn.NeuralNetworkBuilder(input_features, output_features)
    builder.add_elementwise(name="add", input_names=["x"], output_name="y", mode="ADD", alpha=offset)
    return builder.spec

specs = {"identity": make_model(0.0), "plus_one": make_model(1.0)}

try:
    ct.models.multifunction.save_multifunction(specs, str(out))
except Exception as e:
    sys.stderr.write(str(e))
    sys.exit(3)
"""
        case .stateful:
            script = """
import sys, pathlib
try:
    import coremltools as ct
except ImportError:
    sys.exit(2)

out = pathlib.Path(r\"""\(outputURL.path)\""")
out.parent.mkdir(parents=True, exist_ok=True)

# Minimal stateful counter: state_out = state_in + x; y = state_out
try:
    from coremltools.models import datatypes
    from coremltools.models.neural_network import NeuralNetworkBuilder

    input_features = [("x", datatypes.Array(1)), ("state_in", datatypes.Array(1))]
    output_features = [("y", datatypes.Array(1)), ("state_out", datatypes.Array(1))]
    builder = NeuralNetworkBuilder(input_features, output_features, has_skip_connections=False)
    builder.add_elementwise(name="acc", input_names=["x", "state_in"], output_name="y", mode="ADD")
    builder.add_copy(name="copy_state", input_name="y", output_name="state_out")
    builder.spec.description.statefulNetwork.isStateful = True
    builder.spec.description.input[1].isOptional = False
    builder.spec.description.output[1].isLossLayer = False

    mlmodel = ct.models.MLModel(builder.spec)
    mlmodel.save(str(out))
except Exception as e:
    sys.stderr.write(str(e))
    sys.exit(3)
"""
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = ["-c", script]

        let pipe = Pipe()
        process.standardError = pipe
        process.standardOutput = Pipe()
        try process.run()
        process.waitUntilExit()

        let err = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""

        if process.terminationStatus == 2 {
            throw XCTSkip("coremltools not available; skipping \(kind.rawValue) fixture generation.")
        }
        if process.terminationStatus != 0 {
            if err.contains("App Sandbox") || err.contains("xcrun: error") {
                throw XCTSkip("Fixture generation not permitted in sandbox: \(err)")
            }
            throw FixtureError.generationFailed(err)
        }
    }

    @discardableResult
    private static func compileModel(at url: URL, compiledURL: URL) throws -> URL {
        if FileManager.default.fileExists(atPath: compiledURL.path) {
            return compiledURL
        }
        let compiled = try MLModel.compileModel(at: url)
        try? FileManager.default.removeItem(at: compiledURL)
        try FileManager.default.copyItem(at: compiled, to: compiledURL)
        return compiledURL
    }
}
