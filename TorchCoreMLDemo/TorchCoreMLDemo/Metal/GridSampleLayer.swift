import Foundation
import CoreML

// will be loaded by CoreML engine, don't change class name 
@objc(GridSampleLayer)
final class GridSampleLayer: NSObject, MLCustomLayer {
    
    enum Error: Swift.Error {
        case metalNotSupported
        case shaderLibNotFound
        case shaderNotFound
        case cpuNotImplemented // just only GPU
        case encoderInvalid
    }
    
    private let pipelineState: MTLComputePipelineState
    
    required init(parameters: [String : Any]) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Error.metalNotSupported
        }
        guard let library = device.makeDefaultLibrary() else {
            throw Error.shaderLibNotFound
        }
        guard let function = library.makeFunction(name: "grid_sampler") else {
            throw Error.shaderNotFound
        }
        pipelineState = try device.makeComputePipelineState(function: function)
        super.init()
    }
    
    func setWeightData(_ weights: [Data]) throws { }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        return inputShapes
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        throw Error.cpuNotImplemented // see GPU implementation below
    }
    
    // MARK: GPU implementation
    func encode(commandBuffer: MTLCommandBuffer, inputs: [MTLTexture], outputs: [MTLTexture]) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Error.encoderInvalid
        }
        let input = inputs[0]
        let grid = inputs[1]
        let output = outputs[0]

        encoder.setTexture(input, index: 0)
        encoder.setTexture(grid, index: 1)
        encoder.setTexture(output, index: 2)
        
        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadGroupSize = MTLSize(width: w, height: h, depth: 1)

        let threadGroups = MTLSize(
            width:  (output.width       + threadGroupSize.width  - 1) / threadGroupSize.width,
            height: (output.height      + threadGroupSize.height - 1) / threadGroupSize.height,
            depth:  (output.arrayLength + threadGroupSize.depth  - 1) / threadGroupSize.depth
        )
        encoder.setComputePipelineState(pipelineState)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }

}
