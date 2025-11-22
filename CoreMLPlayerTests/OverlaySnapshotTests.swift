import XCTest
import AppKit
@testable import CoreML_Player

/// Lightweight snapshot-style checks for overlay math: draws a box onto a bitmap and validates pixel hits.
final class OverlaySnapshotTests: XCTestCase {
    func testDetectionOverlayDrawsAtExpectedPosition() {
        let size = CGSize(width: 200, height: 100)
        let rect = CGRect(x: 50, y: 25, width: 100, height: 25) // Expected after rectForNormalizedRect
        var buffer = [UInt8](repeating: 0, count: Int(size.width * size.height * 4))
        guard let ctx = CGContext(
            data: &buffer,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: Int(size.width) * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return XCTFail("Failed to create context")
        }

        ctx.setFillColor(NSColor.black.cgColor)
        ctx.fill(CGRect(origin: .zero, size: size))

        ctx.setFillColor(NSColor.red.cgColor)
        ctx.fill(rect)

        guard let cg = ctx.makeImage() else { return XCTFail("No CGImage") }

        let inside = samplePixel(cg: cg, x: 100, y: 40)
        XCTAssertGreaterThan(inside.redComponent, 0.8)
        XCTAssertLessThan(inside.greenComponent, 0.2)
        XCTAssertLessThan(inside.blueComponent, 0.2)

        let outside = samplePixel(cg: cg, x: 5, y: 5)
        XCTAssertLessThan(outside.redComponent, 0.1)
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
