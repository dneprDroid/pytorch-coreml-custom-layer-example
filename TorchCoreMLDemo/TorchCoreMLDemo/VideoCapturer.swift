import AVFoundation
import CoreVideo
import UIKit
import VideoToolbox

protocol VideoCapturerDelegate: AnyObject {
    func videoCapture(_ videoCapture: VideoCapturer, onCapture image: CGImage?)
}

final class VideoCapturer: NSObject {

    enum Error: Swift.Error {
        case captureSessionIsMissing
        case invalidInput
        case invalidOutput
        case unknown
    }

    weak var delegate: VideoCapturerDelegate?

    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private(set) var cameraPostion = AVCaptureDevice.Position.front
    private let sessionQueue = DispatchQueue(label: "VideoCapturer.sessionQueue")

    func start(completion completionHandler: (() -> Void)? = nil) {
        sessionQueue.async {
            if !self.captureSession.isRunning {
                self.captureSession.startRunning()
            }
            if let completionHandler = completionHandler {
                DispatchQueue.main.async {
                    completionHandler()
                }
            }
        }
    }

    func stop(completion completionHandler: (() -> Void)? = nil) {
        sessionQueue.async {
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }

            if let completionHandler = completionHandler {
                DispatchQueue.main.async {
                    completionHandler()
                }
            }
        }
    }

    func setup(completion: @escaping (Swift.Error?) -> Void) {
        sessionQueue.async {
            do {
                try self.setup()
                DispatchQueue.main.async {
                    completion(nil)
                }
            } catch {
                DispatchQueue.main.async {
                    completion(error)
                }
            }
        }
    }
}

private extension VideoCapturer {
    
    func setup() throws {
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .vga640x480

        try setupSessionInput()
        try setupSessionOutput()

        captureSession.commitConfiguration()
    }

    func setupSessionInput() throws {
        guard let captureDevice = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: AVMediaType.video,
            position: cameraPostion
        ) else {
                throw Error.invalidInput
        }

        captureSession.inputs.forEach { input in
            captureSession.removeInput(input)
        }

        guard let videoInput = try? AVCaptureDeviceInput(device: captureDevice) else {
            throw Error.invalidInput
        }
        guard captureSession.canAddInput(videoInput) else {
            throw Error.invalidInput
        }
        captureSession.addInput(videoInput)
    }

    func setupSessionOutput() throws {
        captureSession.outputs.forEach { output in
            captureSession.removeOutput(output)
        }
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)

        guard captureSession.canAddOutput(videoOutput) else {
            throw Error.invalidOutput
        }
        captureSession.addOutput(videoOutput)

        if let connection = videoOutput.connection(with: .video),
            connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
            connection.isVideoMirrored = cameraPostion == .front

            if connection.videoOrientation == .landscapeLeft {
                connection.videoOrientation = .landscapeRight
            } else if connection.videoOrientation == .landscapeRight {
                connection.videoOrientation = .landscapeLeft
            }
        }
    }
}

extension VideoCapturer: AVCaptureVideoDataOutputSampleBufferDelegate {

     func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
     ) {
        guard
             let delegate = delegate,
             let pixelBuffer = sampleBuffer.imageBuffer,
             CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == kCVReturnSuccess
        else {
             return
        }

        var image: CGImage?
        VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &image)

        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)

        DispatchQueue.main.sync {
            delegate.videoCapture(self, onCapture: image)
        }
    }
}
