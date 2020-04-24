from models import inception_v3
from datasets import mivia_3
import tensorflow as tf
import numpy as np
import time
import os


def gan_run(logger):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        
        def downsample(filters, size, apply_batchnorm=True):
            initializer = tf.random_normal_initializer(0., 0.02)
            
            result = tf.keras.Sequential()
            result.add(
                tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                       kernel_initializer=initializer, use_bias=False))
            
            if apply_batchnorm:
                result.add(tf.keras.layers.BatchNormalization())
            
            result.add(tf.keras.layers.LeakyReLU())
            
            return result
        
        def upsample(filters, size, apply_dropout=False):
            initializer = tf.random_normal_initializer(0., 0.02)
            
            result = tf.keras.Sequential()
            result.add(
                tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                use_bias=False))
            
            result.add(tf.keras.layers.BatchNormalization())
            
            if apply_dropout:
                result.add(tf.keras.layers.Dropout(0.5))
            
            result.add(tf.keras.layers.ReLU())
            
            return result
        
        def generator_gen():
            inputs = tf.keras.layers.Input(shape=[128, 128, 1])
            
            down_stack = [
                # downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
                downsample(128, 4),  # (bs, 64, 64, 128)
                downsample(256, 4),  # (bs, 32, 32, 256)
                downsample(512, 4),  # (bs, 16, 16, 512)
                downsample(512, 4),  # (bs, 8, 8, 512)
                downsample(512, 4),  # (bs, 4, 4, 512)
                downsample(512, 4),  # (bs, 2, 2, 512)
                downsample(512, 4),  # (bs, 1, 1, 512)
            ]
            
            up_stack = [
                upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
                upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
                upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
                upsample(512, 4),  # (bs, 16, 16, 1024)
                upsample(256, 4),  # (bs, 32, 32, 512)
                upsample(128, 4),  # (bs, 64, 64, 256)
                # upsample(64, 4),  # (bs, 128, 128, 128)
            ]
            
            initializer = tf.random_normal_initializer(0., 0.02)
            last = tf.keras.layers.Conv2DTranspose(1, 4,
                                                   strides=2,
                                                   padding='same',
                                                   kernel_initializer=initializer,
                                                   activation='tanh')  # (bs, 256, 256, 3)
            
            x = inputs
            
            # Downsampling through the model
            skips = []
            for down in down_stack:
                x = down(x)
                skips.append(x)
            
            skips = reversed(skips[:-1])
            
            # Upsampling and establishing the skip connections
            for up, skip in zip(up_stack, skips):
                x = up(x)
                x = tf.keras.layers.Concatenate()([x, skip])
            
            x = last(x)
            
            return tf.keras.Model(inputs=inputs, outputs=x)
        
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        
        global_batch_size = 32 * strategy.num_replicas_in_sync
        
        def generator_loss(disc_generated_output, gen_output, tgt):
            gan_loss = tf.reduce_sum(
                loss_object(tf.ones_like(disc_generated_output), disc_generated_output) * (1./global_batch_size))
            
            # mean absolute error
            l1_loss = tf.reduce_mean(tf.abs(tgt - gen_output))
            
            total_gen_loss = gan_loss + (100 * l1_loss)
            
            return total_gen_loss, gan_loss, l1_loss
        
        generator = generator_gen()
        
        def discriminator_gen():
            initializer = tf.random_normal_initializer(0., 0.02)
            
            inp = tf.keras.layers.Input(shape=[128, 128, 1], name='input_image')
            tar = tf.keras.layers.Input(shape=[128, 128, 1], name='target_image')
            
            x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)
            
            # down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
            down2 = downsample(128, 4, False)(x)  # (bs, 64, 64, 128)
            down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)
            
            zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
            conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                          kernel_initializer=initializer,
                                          use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)
            
            batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
            
            leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
            
            zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
            
            last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                          kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
            
            return tf.keras.Model(inputs=[inp, tar], outputs=last)
        
        def discriminator_loss(disc_real_output, disc_generated_output):
            real_loss = tf.reduce_sum(loss_object(tf.ones_like(disc_real_output), disc_real_output)
                                      * (1./global_batch_size))
            
            generated_loss = tf.reduce_sum(loss_object(tf.zeros_like(disc_generated_output),
                                                       disc_generated_output) * (1./global_batch_size))
            
            total_disc_loss = real_loss + generated_loss
            
            return total_disc_loss
        
        discriminator = discriminator_gen()
        
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        checkpoint_dir = 'saved_params/gan/checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator,
                                         discriminator=discriminator)
        
        def train_step(noisy, clear):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_output = generator(noisy)
                disc_real_output = discriminator([noisy, clear])
                disc_generated_output = discriminator([noisy, gen_output])
                
                gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, clear)
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
            
            generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            
            generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
            return gen_total_loss, disc_loss, gen_output
        
        train_clear, test_clear = mivia_3.load_data(30)
        train_noisy5, test_noisy5 = mivia_3.load_data(5)
        # train_noisy10, test_noisy10 = mivia_3.load_data(10)
        
        eval_model = inception_v3.InceptionV3Model()
        eval_model.v3.compile(optimizer=eval_model.optimizer_obj,
                              loss=eval_model.loss_obj,
                              metrics=eval_model.metrics_obj)
        eval_model.v3.build(input_shape=[32, 128, 128, 1])
        eval_model.v3.load_weights('saved_params/v3/m2/final_ckpt').expect_partial()
        
        epochs = 99
        latest = tf.train.latest_checkpoint(checkpoint_prefix)
        if latest:
            checkpoint.restore(latest)
            logger.info('restored latest checkpoint')
        
        @tf.function
        def map_fun(data_tr, data_te):
            (dtr, ltr) = data_tr
            (dte, lte) = data_te
            dtr = generator(dtr)
            dte = generator(dte)
            data_tr = (dtr, ltr)
            data_te = (dte, lte)
            return data_tr, data_te
        
        @tf.function
        def one_step(n, c):
            gl, dl, pred = strategy.experimental_run_v2(train_step, args=(n, c))
            return gl, dl, pred
        try:
            for epoch in range(epochs):
                start = time.time()
                
                logger.info("Epoch: %d" % epoch)
                
                predictions = []
                labels = []
                
                # Train
                gen_loss, dis_loss = 0, 0
                steps = 0
                
                for (input_image, target) in zip(strategy.experimental_distribute_dataset(train_noisy5),
                                                 strategy.experimental_distribute_dataset(train_clear)):
                    (n5, l5) = input_image
                    (c30, l30) = target
                    per_replica_gen_loss, per_replica_dis_loss, per_replica_prediction = one_step(n5, c30)
                    
                    gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_gen_loss, axis=None)
                    dis_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_dis_loss, axis=None)
                    prediction = per_replica_prediction.values
                    ground_truth = c30.values
                    reduced_l = l30.values
                    for p in prediction:
                        for pp in p:
                            predictions.append(pp)
                    for lbs in reduced_l:
                        for ll in lbs:
                            labels.append(ll)
                    for gt in ground_truth:
                        for g in gt:
                            truth = g
                            break
                    
                    steps += 1
                    if steps % 100 == 0:
                        logger.info('100 steps trained.')
                        break
                
                # saving (checkpoint) the model every 20 epochs
                # if (epoch + 1) % 20 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
                
                np.savez('saved_params/gan/%02d.npz' % epoch,
                         pred=predictions[-1].numpy(),
                         truth=truth.numpy())
                
                logger.info('Time taken for epoch {} is {} sec\n gen loss: {}, dis loss: {}'.format(epoch + 1,
                                                                                                    time.time() - start,
                                                                                                    gen_loss,
                                                                                                    dis_loss))
                
                new_ds = tf.data.Dataset.from_tensor_slices((predictions, labels)).batch(32)
                for db in range(5, 31, 5):
                    logger.info('Evaluating performance on %ddB OF SNR' % db)
                    loss, acc = eval_model.v3.evaluate(x=new_ds, verbose=1)
                    logger.info("4 groups accuracy on dataset %s for SNR=%d: %5.2f" % ('mivia', db, acc))
        
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(e)
        finally:
            train, test = mivia_3.load_data().map(map_func=map_fun)
            try:
                eval_model.v3.fit(train, epochs=30, verbose=1, validation_data=test, shuffle=True)
            except KeyboardInterrupt:
                pass
            except Exception as e:
                logger.error(e)
            for db in range(5, 31, 5):
                logger.info('Evaluating performance on %ddB OF SNR' % db)
                loss, acc = eval_model.v3.evaluate(x=test, verbose=1)
                logger.info("4 groups accuracy on dataset %s for SNR=%d: %5.2f" % ('mivia', db, acc))
