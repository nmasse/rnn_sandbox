import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from atari_utils import GameWrapper
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import matplotlib.pyplot as plt
import os
import argparse
import pickle

gpu_idx = 2
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')

parser = argparse.ArgumentParser('')
parser.add_argument('--env_name', type=str, default='SpaceInvadersNoFrameskip-v4')
parser.add_argument('--training_size', type=int, default=50000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--load_saved_states', type=bool, default=False)
parser.add_argument('--binarize', type=bool, default=True)
parser.add_argument('--binary_threshold', type=int, default=35)
parser.add_argument('--increase_motion_weight', type=bool, default=True)
parser.add_argument('--save_fn_suffix', type=str, default='space_invaders')


args = parser.parse_args('')
print('Arguments:')
for k, v in vars(args).items():
    print(k,':', v)
print()


class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ENCODER
encoder_inputs = keras.Input(shape=(80, 80, 4))
x = layers.Conv2D(64, 6, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(128, 4, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(128, 4, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
print('Last conv layer shape', x.shape)
x = layers.Flatten()(x)
x = layers.Dense(2*args.latent_dim, activation="relu")(x)
z_mean = layers.Dense(args.latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(args.latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
print(encoder.summary())

# DECODER
latent_inputs = keras.Input(shape=(args.latent_dim,))
x = layers.Dense(5 * 5 * 256, activation="relu")(latent_inputs)
x = layers.Reshape((5, 5, 256))(x)
x = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(128, 4, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(128, 4, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(64, 6, activation="relu", strides=2, padding="same")(x)
if args.binarize:
    decoder_outputs = layers.Conv2DTranspose(4, 3, activation="sigmoid", padding="same")(x)
else:
    decoder_outputs = layers.Conv2DTranspose(4, 3, activation="linear", padding="same")(x)
print('Decoder shape', decoder_outputs.shape)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
print(decoder.summary())


class AnnealingCallback(keras.callbacks.Callback):
    def __init__(self):
        pass
    def on_epoch_begin(self, epoch, logs={}):

        if not hasattr(self.model, "kl_beta"):
            raise ValueError('Optimizer must have a "kl_beta" attribute.')
        w = tf.keras.backend.get_value(self.model.kl_beta)
        if args.binarize:
            new_weight = np.clip(w + 0.02, 0., 4.)
        else:
            new_weight = np.clip(w + 0.002, 0., 0.4) # roughly a factor of 10 difference betweem MSE and binary cross entropy losses
        tf.keras.backend.set_value(self.model.kl_beta, new_weight)
        print("Current KL Weight is " + str(tf.keras.backend.get_value(self.model.kl_beta)))


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.kl_beta = tf.Variable(0., trainable = False)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):

        data = tf.cast(data, tf.float32)
        if not args.binarize:
            data /= 255.
            data -= tf.reduce_min(data, axis = (1,2,3), keepdims=True)
            data /= tf.reduce_max(data, axis = (1,2,3), keepdims=True)

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            if args.increase_motion_weight:
                max_val = tf.reduce_max(data, axis=-1)
                min_min = tf.reduce_min(data, axis=-1)
                weights = 1 + 9.*(max_val - min_min)
            else:
                weights = 1.

            if args.binarize:
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        weights * keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                    )
                )
            else:
                mse = keras.losses.MSE(data, reconstruction)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(weights * mse, axis=(1, 2))
                )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = self.kl_beta * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# Collect training data
if args.load_saved_states:
    print('Loading saved states...')
    data = pickle.load(open('multi_env_saved_states_5envs.pkl','rb'))
    print(f'Training size {data["states"].shape[0]}')
else:

    env_names = [
        'SpaceInvadersNoFrameskip-v4',
    ]

    data = {'states':[], 'env_names': env_names}

    for i, env_name in enumerate(env_names):
        print(f'Current ENV: {env_name}')
        env = make_atari(env_name)
        env = wrap_deepmind(env, episode_life=False, frame_stack=True, scale=False)
        action_dim = env.action_space.n
        state  = env.reset()

        while len(data['states']) < (i+1)*args.training_size:
            for _ in range(15):
                action = np.random.choice(action_dim)
                state, reward, done, _ = env.step(action)
                if done:
                    state = env.reset()
                    skip = True
                    break
                else:
                    skip = False
            if skip:
                continue
            state = np.uint8(state)
            state = state[2:-2, :, :]
            state = state[:, 2:-2, :]
            data['states'].append(state)
            if len(data['states']) % 1000 == 0:
                print(f"State size {len(data['states'])}")

    data['states'] = np.stack(data['states'], axis = 0)
    """
    try:
        pickle.dump(data, open('space_invaders_saved_states.pkl','wb'), protocol=4)
    except:
        pass
    """

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-04))



if args.binarize:

    data['states'] = np.uint8(data['states'] >= args.binary_threshold) # 35 is chosen to include all salient features
    data['states'] = np.concatenate((data['states'], data['states'][:, :, ::-1, :]), axis = 0) # include horizontally flipped states
    print('Augmented')

#datagen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)

# TRAIN MODEL
N = 100
save_fn = f"saved_models/saved_vae_encoder_{args.save_fn_suffix}"
for i in range(4):
    print(f'LOOP {i}')
    vae.fit(data['states'], epochs=N, batch_size=args.batch_size, callbacks = [AnnealingCallback()])

    # save models

    vae.encoder.save(save_fn)
    #vae.decoder.save('saved_models/saved_decoder_' + args.save_fn_suffix)

# plot several examples
s = np.float32(data['states'][::8000, ...])
if not args.binarize:
    s /= 255.
z, _, z_sample = vae.encoder.predict(s)

z0 = vae.decoder.predict(z)
z0 = np.reshape(z0, (-1, 80,80,4))
z0 = np.max(z0, axis = -1)
z1 = vae.decoder.predict(z_sample)
z1 = np.reshape(z1, (-1, 80,80,4))
z1 = np.max(z1, axis = -1)
x_original = np.max(s, axis = -1)
for n in range(s.shape[0]):
    f, ax = plt.subplots(1,3,figsize=(14, 6))
    ax[0].imshow(x_original[n, ...], aspect='auto', interpolation='none')
    ax[1].imshow(z0[n, ...], aspect='auto', interpolation='none')
    ax[2].imshow(z1[n, ...], aspect='auto', interpolation='none')
    plt.show()
