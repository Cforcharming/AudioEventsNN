from tensorflow.keras import layers
from datasets import mivia_3
import tensorflow as tf
import time


if __name__ == "__main__":
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    
    def generator_gen():
        model = tf.keras.Sequential()
        model.add(layers.Dense(128, use_bias=False, input_shape=(128, 1)))
        
        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)
        
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 128, 1)
        return model
    
    
    def discriminator_gen():
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        return model
    
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    generator = generator_gen()
    discriminator = discriminator_gen()
    
    
    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    
    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    
    
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)


    @tf.function
    def train_step(clear_mel, noisy_mel):
    
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noisy_mel, training=True)
        
            real_output = discriminator(clear_mel, training=True)
            fake_output = discriminator(generated_images, training=True)
        
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
    
        train_ds, test_ds = mivia_3.load_data(5)
        epochs = 30
        
        for epoch in range(epochs):
            start = time.time()
    
            for image_batch in train_ds:
                train_step(image_batch)
    
            if (epoch + 1) % 7 == 0:
                checkpoint.save(file_prefix='saved_params/gan/checkpoints/{epoch:04d}_ckpt')
    
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
