import CoreML

extension MLMultiArray {
        
    typealias NdArray4d = [[[[Float32]]]]
    
    static func array4d(fromJson url: URL) -> MLMultiArray? {
        guard let data = try? Data(contentsOf: url) else {
            return nil
        }
        guard let ndarray = try? JSONDecoder().decode(NdArray4d.self, from: data) else {
            return nil
        }
        let shape = [
            ndarray.count,
            ndarray.first?.count ?? 0,
            ndarray.first?.first?.count ?? 0,
            ndarray.first?.first?.first?.count ?? 0
        ]
        let numberShape = shape.map { NSNumber(value: $0) }
        guard let marray = try? MLMultiArray(shape: numberShape, dataType: .float32) else {
            return nil
        }
        for n in 0..<shape[0] {
            for c in 0..<shape[1] {
                for x in 0..<shape[2] {
                    for y in 0..<shape[3] {
                        let index = [n, c, x, y].map { NSNumber(value: $0) }
                        marray[index] = NSNumber(value: ndarray[n][c][x][y])
                    }
                }
            }
        }
        return marray
    }
}

