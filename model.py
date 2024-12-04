import numpy as np


def ParCol(input_data, kernel_size, stride):
    """
    ParCol: 이미지를 작은 패치로 나누어 2D 행렬로 변환 (im2col 기법).
    """
    C, H, W = input_data.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1

    col = np.zeros((C * kernel_size * kernel_size, out_h * out_w))

    for y in range(out_h):
        for x in range(out_w):
            patch = input_data[:, y * stride:y * stride + kernel_size, x * stride:x * stride + kernel_size]
            col[:, y * out_w + x] = patch.flatten()

    return col, out_h, out_w


class Kernel:
    def __init__(self, input_depth, kernel_size, learning_rate):
        self.input_depth = input_depth
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate

        # Xavier Initialization
        self.weights = np.random.randn(kernel_size, kernel_size, input_depth) * 0.1
        self.bias = 0

    def forward(self, input_data, ParCol_matrix, out_h, out_w):
        """
        Forward pass: Convolution as matrix multiplication.
        """
        kernel_flattened = self.weights.reshape(-1, 1)
        conv_result = np.dot(kernel_flattened.T, ParCol_matrix) + self.bias
        return conv_result.reshape(out_h, out_w)

    def backward(self, ParCol_matrix, output_gradient):
        """
        Backward pass: Compute gradients for weights and bias.
        """
        grad_weights = np.dot(output_gradient.flatten(), ParCol_matrix.T)
        grad_bias = np.sum(output_gradient)
        input_gradient = np.dot(self.weights.reshape(-1, 1), output_gradient.flatten())
        input_gradient = input_gradient.reshape(self.kernel_size, self.kernel_size, self.input_depth)

        # Update weights and bias
        self.weights -= self.learning_rate * grad_weights.reshape(self.kernel_size, self.kernel_size, self.input_depth)
        self.bias -= self.learning_rate * grad_bias

        return input_gradient

    def update(self):
        """
        Reset gradients (handled after each batch).
        """
        self.weights.fill(0)
        self.bias = 0


class ConvolutionalLayer:
    def __init__(self, input_width, input_height, input_depth, kernel_size, kernel_depth, stride, learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.input_depth = input_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.learning_rate = learning_rate

        self.output_width = (input_width - kernel_size) // stride + 1
        self.output_height = (input_height - kernel_size) // stride + 1
        self.output_depth = kernel_depth

        # Initialize multiple kernels
        self.kernels = [Kernel(input_depth, kernel_size, learning_rate) for _ in range(kernel_depth)]

    def forward(self, input_data):
        """
        Forward pass through the convolutional layer.
        """
        self.input_data = input_data
        self.ParCol_matrix, self.out_h, self.out_w = ParCol(input_data, self.kernel_size, self.stride)
        self.output = np.zeros((self.output_depth, self.out_h, self.out_w))

        for d in range(self.output_depth):
            self.output[d] = self.kernels[d].forward(input_data, self.ParCol_matrix, self.out_h, self.out_w)

        return self.output

    def backward(self, output_gradient):
        """
        Backward pass through the convolutional layer.
        """
        input_gradient = np.zeros_like(self.input_data)

        for d in range(self.output_depth):
            input_gradient += self.kernels[d].backward(self.ParCol_matrix, output_gradient[d])

        return input_gradient
    
    def update(self):
        """
        각 Kernel의 update 메서드를 호출하여 가중치를 업데이트합니다.
        """
        for kernel in self.kernels:
            kernel.update()

class MaxPoolingLayer:
    def __init__(self, input_width, input_height, input_depth, pool_size):
        self.input_width = input_width
        self.input_height = input_height
        self.input_depth = input_depth
        self.pool_size = pool_size
        self.stride = pool_size

        self.output_width = (input_width - pool_size) // self.stride + 1
        self.output_height = (input_height - pool_size) // self.stride + 1
        self.output_depth = input_depth

    def forward(self, input_data):
        """
        Forward pass through the max pooling layer.
        """
        self.input_data = input_data
        # 디버깅: 입력 데이터 크기 확인
        print(f"Input to MaxPooling: {input_data.shape}")
        self.output = np.zeros((self.output_depth, self.output_height, self.output_width))

        for d in range(self.input_depth):
            for i in range(0, self.input_height - self.pool_size + 1, self.stride):
                for j in range(0, self.input_width - self.pool_size + 1, self.stride):
                    self.output[d, i // self.pool_size, j // self.pool_size] = np.max(
                        input_data[d, i:i + self.pool_size, j:j + self.pool_size]
                    )
        print(f"Output shape: {self.output.shape}")
        return self.output

    def backward(self, output_gradient):
        """
        Backward pass through the max pooling layer.
        """
        input_gradient = np.zeros_like(self.input_data)

        for d in range(self.input_depth):
            for i in range(0, self.input_height, self.pool_size):
                for j in range(0, self.input_width, self.pool_size):
                    patch = self.input_data[d, i:i + self.pool_size, j:j + self.pool_size]
                    max_index = np.argmax(patch)
                    max_coords = np.unravel_index(max_index, patch.shape)
                # 디버깅용 출력 추가
                print(f"Input gradient shape: {input_gradient.shape}")
                print(f"Output gradient shape: {output_gradient.shape}")
                print(f"Patch shape: {patch.shape}, max_coords: {max_coords}")
                print(f"i: {i}, j: {j}, max_coords: {max_coords}")

                # 범위 체크 및 값 할당
                input_i = i + max_coords[0]
                input_j = j + max_coords[1]
                if input_i < self.input_height and input_j < self.input_width:
                    input_gradient[d, input_i, input_j] = output_gradient[d, i // self.pool_size, j // self.pool_size]

        return input_gradient


class FullyConnectedLayer:
    def __init__(self, input_size, output_size, learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.bias = np.zeros((output_size, 1))

    def forward(self, input_data):
        self.input_data = input_data.flatten().reshape(-1, 1)
        return np.dot(self.weights, input_data) + self.bias

    def backward(self, output_gradient):
        output_gradient = output_gradient.reshape(self.output_size, 1)  # (output_size, 1)로 reshape
        weights_gradient = np.dot(output_gradient, self.input_data.T)  # (output_size, input_size)
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)

        # 가중치와 편향 업데이트
        self.weights -= self.learning_rate * weights_gradient
        self.bias -= self.learning_rate * bias_gradient

        input_gradient = np.dot(self.weights.T, output_gradient)  # (input_size, 1)
        return input_gradient  # Flatten하여 반환



class ReLULayer:
    def forward(self, input_data):
        self.input_data = input_data
        return np.maximum(0, input_data)

    def backward(self, output_gradient):
        return output_gradient * (self.input_data > 0)


class CNNModel:
    def __init__(self, learning_rate=0.01):
        self.conv1 = ConvolutionalLayer(28, 28, 1, 3, 6, 1, learning_rate)
        self.relu1 = ReLULayer()
        self.pool1 = MaxPoolingLayer(26, 26, 6, 2)

        self.conv2 = ConvolutionalLayer(13, 13, 6, 3, 16, 1, learning_rate)
        self.relu2 = ReLULayer()
        self.pool2 = MaxPoolingLayer(11, 11, 16, 2)

        self.fc = FullyConnectedLayer(5 * 5 * 16, 10, learning_rate)

    def forward(self, input_data):
        out = self.conv1.forward(input_data)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)

        out = out.flatten().reshape(-1, 1)
        out = self.fc.forward(out)
        return out

    def backward(self, output_gradient):
        # output_gradient의 크기가 (output_size, 1)인지 확인
        #output_gradient = output_gradient.reshape(self.fc.output_size, 1)  # (10, 1)
        grad = self.fc.backward(output_gradient)
        grad = grad.reshape(16, 5, 5)

        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)

        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        self.conv1.backward(grad)

    def update(self):
        self.conv1.update()
        self.conv2.update()
        self.fc.update()
