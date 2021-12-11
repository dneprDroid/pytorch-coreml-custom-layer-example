import UIKit

extension CGImage {
    
    func scaled(size: CGSize) -> CGImage {
        guard let context = CGContext(
            data: nil,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: self.bitsPerComponent,
            bytesPerRow: self.bytesPerRow,
            space: self.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: self.bitmapInfo.rawValue
        ) else {
            return self
        }
        context.interpolationQuality = .high
        context.setShouldAntialias(true)
        context.setAllowsAntialiasing(true)
        
        context.draw(self, in: CGRect(origin: .zero, size: size))
        guard let resultImg = context.makeImage() else {
            return self
        }
        return resultImg
    }
}
