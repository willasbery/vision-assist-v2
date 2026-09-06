import Foundation
import Masking
import SwiftUI

/// Draws a mask as a translucent tint over whatever is behind it.
struct MaskOverlayView: View {
    let mask: Mask?
    var tint: Color = .green

    var body: some View {
        GeometryReader { geometry in
            if let mask, let image = Self.image(from: mask) {
                // The preview layer uses .resizeAspectFill, so the overlay has
                // to crop the same way or the two will not line up.
                Image(decorative: image, scale: 1)
                    .resizable()
                    .interpolation(.none)
                    .aspectRatio(CGFloat(mask.width) / CGFloat(mask.height),
                                 contentMode: .fill)
                    .frame(width: geometry.size.width, height: geometry.size.height)
                    .clipped()
                    .colorMultiply(tint)
                    .allowsHitTesting(false)
            }
        }
    }

    /// Premultiplied RGBA, white where the mask is set and fully transparent
    /// where it is not. White so that `.colorMultiply` can tint it.
    ///
    /// The pixel buffer is wrapped in `Data` and handed to `CGDataProvider`,
    /// which retains it — pointing a provider at a local buffer would leave
    /// the image referencing freed memory.
    private static func image(from mask: Mask) -> CGImage? {
        let opacity: UInt8 = 180
        var pixels = [UInt8](repeating: 0, count: mask.values.count * 4)
        for (index, value) in mask.values.enumerated() where value != 0 {
            let offset = index * 4
            pixels[offset] = opacity      // premultiplied red
            pixels[offset + 1] = opacity  // premultiplied green
            pixels[offset + 2] = opacity  // premultiplied blue
            pixels[offset + 3] = opacity  // alpha
        }

        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else {
            return nil
        }

        return CGImage(
            width: mask.width,
            height: mask.height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: mask.width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }
}
