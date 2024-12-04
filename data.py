import numpy as np

def image_data(filename):
    with open(filename, 'rb') as f:
        _ = int.from_bytes(f.read(4), byteorder='big')
        num_images = int.from_bytes(f.read(4), byteorder='big')
        num_rows = int.from_bytes(f.read(4), byteorder='big')
        num_cols = int.from_bytes(f.read(4), byteorder='big')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape((num_images, num_rows, num_cols))
        images = images / 255.0
    return images

def label_data(filename):
    with open(filename, 'rb') as f:
        _ = int.from_bytes(f.read(4), byteorder='big')
        num_labels = int.from_bytes(f.read(4), byteorder='big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def shuffle_data(images, labels):
    idx = np.arange(len(images))
    np.random.shuffle(idx)
    return images[idx], labels[idx]

def load_data(data_dir, shuffle=True):
    train_images = image_data(data_dir + '/train-images.idx3-ubyte')
    train_labels = label_data(data_dir + '/train-labels.idx1-ubyte')
    test_images = image_data(data_dir + '/t10k-images.idx3-ubyte')
    test_labels = label_data(data_dir + '/t10k-labels.idx1-ubyte')
    
    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)

    if shuffle:
        train_images, train_labels = shuffle_data(train_images, train_labels)
    
    return (train_images, train_labels), (test_images, test_labels)

# ======== 데이터 로드 확인 코드 ========
if __name__ == "__main__":
    data_dir = "."  # 데이터셋이 위치한 디렉토리
    (train_images, train_labels), (test_images, test_labels) = load_data(data_dir)

    print(f"Train Images Shape: {train_images.shape}")  # 예: (60000, 28, 28)
    print(f"Train Labels Shape: {train_labels.shape}")  # 예: (60000, 10)
    print(f"Test Images Shape: {test_images.shape}")    # 예: (10000, 28, 28)
    print(f"Test Labels Shape: {test_labels.shape}")    # 예: (10000, 10)

    # 첫 번째 이미지와 레이블 확인
    print("First Train Image Pixel Values:")
    print(train_images[0])  # 첫 번째 이미지의 픽셀 값
    print(f"First Train Label (One-Hot): {train_labels[0]}")  # 첫 번째 이미지의 원-핫 레이블
    print(f"First Train Label (Integer): {np.argmax(train_labels[0])}")  # 정수형 레이블
