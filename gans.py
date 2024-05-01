import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from PIL import Image

# Hàm tạo bộ sinh
# Hàm tạo bộ sinh
def build_generator(latent_dim, input_shape=(380, 380, 3)):
    base_model = EfficientNetB4(
        weights=None,
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )
    output = layers.Dense(380*380*3, activation='tanh')(base_model.output)
    return models.Model(inputs=base_model.input, outputs=output)


# Hàm tạo bộ phân biệt
def build_discriminator(input_shape):
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Hàm tạo và compiles GANs
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential([
        generator,
        discriminator
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# Hàm huấn luyện GANs
def train_gan(generator, discriminator, gan, latent_dim, X_train, num_epochs=100, batch_size=16):
    # Tạo callback để lưu lại các mô hình và các thông số log
    checkpoint_generator = ModelCheckpoint("generator.h5", monitor='loss', verbose=1, save_best_only=True)
    checkpoint_discriminator = ModelCheckpoint("discriminator.h5", monitor='loss', verbose=1, save_best_only=True)
    csv_logger_generator = CSVLogger("generator_log.csv", append=True)
    csv_logger_discriminator = CSVLogger("discriminator_log.csv", append=True)
    
    for epoch in range(num_epochs):
        # Tạo ra dữ liệu ngẫu nhiên cho bộ sinh
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        # Tạo ra các hình ảnh giả mạo từ bộ sinh
        fake_images = generator.predict(noise)
        # Chọn ngẫu nhiên một số hình ảnh từ dữ liệu thật
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        
        # Tạo dữ liệu huấn luyện cho bộ phân biệt
        X = np.concatenate([real_images, fake_images])
        # Nhãn cho dữ liệu thật và giả mạo
        y = np.ones(2 * batch_size)
        y[batch_size:] = 0
        
        # Huấn luyện bộ phân biệt
        d_loss = discriminator.train_on_batch(X, y)
        
        # Huấn luyện bộ sinh
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        y = np.ones(batch_size)
        g_loss = gan.train_on_batch(noise, y)
        
        # In ra tiến trình huấn luyện
        print(f"Epoch: {epoch + 1}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")
    
    # Lưu các mô hình sau khi huấn luyện
    generator.save("generator_final.keras")
    discriminator.save("discriminator_final.keras")

# Thiết lập các tham số
latent_dim = 100  # Kích thước của vector nhiễu đầu vào cho bộ sinh
num_epochs = 10000
batch_size = 16

# Tải dữ liệu
data_dir = "./Detect_chicken_sex_V3"
inputs = []
IMG_SIZE = (380, 380)
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image = Image.open(image_path)
        image = image.resize(IMG_SIZE)
        image_array = np.array(image)
        image_array = image_array.astype("float32") / 255.0
        inputs.append(image_array)
X_train = np.array(inputs)

# Tạo và biên dịch các mô hình
generator = build_generator(latent_dim)
discriminator = build_discriminator(input_shape=(380, 380, 3))
gan = build_gan(generator, discriminator)

# Huấn luyện GANs
train_gan(generator, discriminator, gan, latent_dim, X_train, num_epochs, batch_size)
