import XCTest
import Foundation

/// Sanity checks for project configuration files used by CI.
final class ProjectConfigurationTests: XCTestCase {
    func testXCTestPlanListsCoreMLPlayerTestsTarget() throws {
        let testFile = URL(fileURLWithPath: #filePath)
        let repoRoot = testFile.deletingLastPathComponent().deletingLastPathComponent()
        let planURL = repoRoot.appendingPathComponent("CoreML Player.xctestplan")

        guard FileManager.default.fileExists(atPath: planURL.path) else {
            throw XCTSkip("xctestplan not found at expected path: \(planURL.path)")
        }

        let data = try Data(contentsOf: planURL)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let testTargets = json?["testTargets"] as? [[String: Any]]
        let target = testTargets?.first?["target"] as? [String: Any]

        XCTAssertEqual(target?["name"] as? String, "CoreMLPlayerTests")
    }
}
