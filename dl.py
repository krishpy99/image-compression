import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.optimizers import Adam

# Define the Encoder
def build_encoder(input_shape):
    encoder_input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    encoder_output = MaxPooling2D((2, 2), padding='same')(x)
    
    encoder = Model(encoder_input, encoder_output, name="encoder")
    return encoder

# Define the Decoder for Fine-Tuning
def build_decoder(encoded_shape):
    decoder_input = Input(shape=encoded_shape)
    x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(decoder_input)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    decoder_output = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    decoder = Model(decoder_input, decoder_output, name="decoder")
    return decoder

# Define the Autoencoder Model
input_shape = (128, 128, 3)  # Example input shape, adjust as necessary
encoder = build_encoder(input_shape)
decoder = build_decoder(encoder.output_shape[1:])
autoencoder = Model(encoder.input, decoder(encoder.output))

# Compile the Model
autoencoder.compile(optimizer=Adam(learning_rate=1e-5), loss='mse')

# Summary of the Model
autoencoder.summary()

# Load the image and preprocess
def load_and_preprocess_image(img_path, target_size=(32, 32)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

img_path = 'example.jpg'  # Replace with your image path
input_image = load_and_preprocess_image(img_path)

# Get the compressed representation from the encoder
compressed_representation = encoder.predict(input_image)
print("Compressed representation shape:", compressed_representation.shape)

# Reconstruct the image from the compressed representation
reconstructed_image = decoder.predict(compressed_representation)

# Remove the batch dimension and rescale the pixel values
reconstructed_image = np.squeeze(reconstructed_image)
reconstructed_image = (reconstructed_image * 255).astype(np.uint8)

# Display the original and reconstructed images
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(np.squeeze(input_image))

plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(reconstructed_image)

plt.show()
