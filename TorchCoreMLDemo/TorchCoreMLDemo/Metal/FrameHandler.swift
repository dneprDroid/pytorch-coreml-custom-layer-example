import Foundation
import CoreML
import UIKit
import Vision
import Metal
import VideoToolbox

private let queue = DispatchQueue(label: "FrameHandler.queue", qos: .userInitiated)

final class FrameHandler {
    
    enum Constants {
        static let wh = 512
    }
    
    enum Errors: Error {
        case outputInvalid
    }
    
    private var model: MLModel?
    private var grid: MLMultiArray?
    private var finished = true    
    
    func setup(
        modelUrl: URL,
        gridUrl: URL,
        onError: @escaping (Error) -> Void
    ) {
        queue.async {
            do {
                
                
                let compiledUrl = try MLModel.compileModel(at: modelUrl)
                let config = MLModelConfiguration()
                config.computeUnits = .all
                self.model = try MLModel(contentsOf: compiledUrl, configuration: config)
                try? FileManager.default.removeItem(at: compiledUrl)
                
                
                self.grid = MLMultiArray.array4d(fromJson: gridUrl)
            } catch {
                onError(error)
            }
        }
    }
    
    func render(frame: CGImage, on imageView: UIImageView) {
        guard finished else { return }
        finished = false
        queue.async {
            let outImage = self.pred(frame: frame)
            DispatchQueue.main.async {
                imageView.image = outImage.flatMap { UIImage(cgImage: $0) }
                self.finished = true
            }
        }
    }
    
    private func pred(frame: CGImage) -> CGImage? {
        guard let model = self.model, let grid = self.grid else {
            return nil
        }
        do {
            let input = Input(input: frame, grid: grid)
            let result = try model.prediction(from: input)
                .featureValue(for: "output")
            guard let pixelBuffer = result?.imageBufferValue else {
                throw Errors.outputInvalid
            }
            var outImage: CGImage?
            VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &outImage)
            return outImage?.scaled(
                size: CGSize(width: frame.width, height: frame.height)
            )
        } catch {
            print("prediction error: \(error)")
        }
        return nil
    }
}

private extension FrameHandler {
    
    final class Input: MLFeatureProvider {
        
        var featureNames: Set<String> = ["input", "warp_grid"]
        
        private let input: CGImage
        private let grid: MLMultiArray
        private let options: [MLFeatureValue.ImageOption : Any]

        init(input: CGImage, grid: MLMultiArray) {
            self.input = input
            self.grid = grid
            self.options = [
                MLFeatureValue.ImageOption.cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue
            ]
        }
        
        func featureValue(for featureName: String) -> MLFeatureValue? {
            if featureName == "input" {
                let value = try? MLFeatureValue(
                    cgImage: input,
                    pixelsWide: Constants.wh,
                    pixelsHigh: Constants.wh,
                    pixelFormatType: input.pixelFormatInfo.rawValue,
                    options: self.options
                )
                return value
            }
            if featureName == "warp_grid" {
                return MLFeatureValue(multiArray: grid)
            }
            return .none
        }
    }
}
