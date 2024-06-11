import numpy as np
import cv2
from scipy.fftpack import dct, idct

def apply_dct(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

def apply_idct(dct_image):
    return idct(idct(dct_image.T, norm='ortho').T, norm='ortho')

def compress_image(image, quantization_matrix):
    height, width = image.shape
    print(height, width)
    compressed_image = np.zeros_like(image, dtype=np.float32)
    
    # Apply DCT and quantization block-wise
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image[i:i+8, j:j+8]
            #print("compress block: ", block)
            dct_block = apply_dct(block)
            quantized_block = np.round(dct_block / quantization_matrix)
            #print("quant compress block: ", quantized_block)
            compressed_image[i:i+8, j:j+8] = quantized_block
    
    return compressed_image

def decompress_image(compressed_image, quantization_matrix):
    height, width = compressed_image.shape
    print(height, width)
    decompressed_image = np.zeros_like(compressed_image, dtype=np.float32)
    
    # Dequantization and apply IDCT block-wise
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            quantized_block = compressed_image[i:i+8, j:j+8]
            dequantized_block = quantized_block * quantization_matrix
            block = apply_idct(dequantized_block)
            decompressed_image[i:i+8, j:j+8] = block
    
    return np.clip(decompressed_image, 0, 255).astype(np.uint8)

def compress_image_color(image, quantization_matrix):
    height, width, channels = image.shape
    compressed_image = np.zeros_like(image, dtype=np.float32)
    
    # Apply DCT and quantization block-wise
    for channel in range(channels):
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                block = image[i:i+8, j:j+8, channel]
                dct_block = apply_dct(block)
                quantized_block = np.round(dct_block / quantization_matrix)
                compressed_image[i:i+8, j:j+8, channel] = quantized_block
    
    return compressed_image

def decompress_image_color(compressed_image, quantization_matrix):
    height, width, channels = compressed_image.shape
    decompressed_image = np.zeros_like(compressed_image, dtype=np.float32)
    
    # Dequantization and apply IDCT block-wise
    for channel in range(channels):
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                quantized_block = compressed_image[i:i+8, j:j+8, channel]
                dequantized_block = quantized_block * quantization_matrix
                block = apply_idct(dequantized_block)
                decompressed_image[i:i+8, j:j+8, channel] = block
    
    return np.clip(decompressed_image, 0, 255).astype(np.uint8)

# Example usage
quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

quantization_matrix2 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
])

quantization_matrix3 = np.array([
    [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
    [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
    [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
    [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
    [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
    [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
    [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
    [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
])

image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)
compressed_image = compress_image(image, quantization_matrix)
decompressed_image = decompress_image(compressed_image, quantization_matrix)

cv2.imwrite('compressed.jpg', compressed_image)
cv2.imwrite('decompressed.jpg', decompressed_image)

image_color = cv2.imread('example.jpg')
compressed_image_color = compress_image_color(image_color, quantization_matrix)
decompressed_image_color = decompress_image_color(compressed_image_color, quantization_matrix)

cv2.imwrite('compressed_color.jpg', compressed_image_color)
cv2.imwrite('decompressed_color.jpg', decompressed_image_color)

image_color = cv2.imread('example.jpg')
compressed_image_color = compress_image_color(image_color, quantization_matrix2)
decompressed_image_color = decompress_image_color(compressed_image_color, quantization_matrix2)

cv2.imwrite('compressed_color2.jpg', compressed_image_color)
cv2.imwrite('decompressed_color2.jpg', decompressed_image_color)

image_color = cv2.imread('example.jpg')
compressed_image_color = compress_image_color(image_color, quantization_matrix3)
decompressed_image_color = decompress_image_color(compressed_image_color, quantization_matrix3)

cv2.imwrite('compressed_color3.jpg', compressed_image_color)
cv2.imwrite('decompressed_color3.jpg', decompressed_image_color)
