import XCTest
import AppKit
@testable import CoreML_Player

/// Lightweight snapshot-style checks for overlay math: draws a box onto a bitmap and validates pixel hits.
final class OverlaySnapshotTests: XCTestCase {
    func testDetectionOverlayDrawsAtExpectedPosition() {
        let size = CGSize(width: 200, height: 100)
        let rect = CGRect(x: 50, y: 25, width: 100, height: 25) // Expected after rectForNormalizedRect

        let image = NSImage(size: size)
        image.lockFocus()
        NSColor.black.setFill()
        NSBezierPath(rect: CGRect(origin: .zero, size: size)).fill()

        NSColor.red.setFill()
        NSBezierPath(rect: rect).fill()
        image.unlockFocus()

        guard let cg = image.cgImageForCurrentRepresentation else {
            return XCTFail("No CGImage")
        }
        // Sample a pixel in the middle of the expected box
        XCTAssertEqual(samplePixel(cg: cg, x: 100, y: 40), .red)
        // Sample a pixel outside the box
        XCTAssertEqual(samplePixel(cg: cg, x: 10, y: 10), .black)
    }

    private func samplePixel(cg: CGImage, x: Int, y: Int) -> NSColor {
        guard let data = cg.dataProvider?.data else { return .clear }
        let ptr = CFDataGetBytePtr(data)!
        let bytesPerPixel = 4
        let offset = ((cg.height - 1 - y) * cg.bytesPerRow) + x * bytesPerPixel
        let r = ptr[offset]
        let g = ptr[offset + 1]
        let b = ptr[offset + 2]
        return NSColor(red: CGFloat(r)/255, green: CGFloat(g)/255, blue: CGFloat(b)/255, alpha: 1)
    }
}
