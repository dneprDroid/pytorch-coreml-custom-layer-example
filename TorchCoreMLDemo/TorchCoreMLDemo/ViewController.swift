import UIKit

final class ViewController: UIViewController {
    
    enum Errors: Swift.Error {
        case modelNotFound
    }

    @IBOutlet private weak var imageView: UIImageView!
    
    private let capturer = VideoCapturer()
    private let frameHandler = FrameHandler()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupModel()
        setupCapture()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        UIApplication.shared.isIdleTimerDisabled = true
        capturer.start(completion: .none)
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        UIApplication.shared.isIdleTimerDisabled = false
        capturer.stop(completion: .none)
    }
}

private extension ViewController {
    
    func setupCapture() {
        capturer.delegate = self
        capturer.setup { [weak self] error in
            guard let error = error else { return }
            self?.present(error: error)
        }
    }
    
    func setupModel() {
        guard
            let modelUrl = Bundle.main.url(
                forResource: "model.pb",
                withExtension: .none
            ),
            let gridUrl = Bundle.main.url(
                forResource: "grid.json",
                withExtension: .none
            )
        else {
            self.present(error: Errors.modelNotFound)
            return
        }
        frameHandler.setup(modelUrl: modelUrl, gridUrl: gridUrl) { [weak self] error in
            self?.present(error: error)
        }
    }
    
    func present(error: Error) {
        DispatchQueue.main.async {
            let alert = UIAlertController(
                title: "Error",
                message: error.localizedDescription,
                preferredStyle: .alert
            )
            let okAction = UIAlertAction(
                title: "OK",
                style: .default,
                handler: { _ in }
            )
            alert.addAction(okAction)
            self.present(alert, animated: true)
        }
    }
}

extension ViewController: VideoCapturerDelegate {
    
    func videoCapture(_ videoCapture: VideoCapturer, onCapture image: CGImage?) {
        guard let image = image else { return }
        frameHandler.render(frame: image, on: imageView)
    }    
}
