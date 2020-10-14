import os
import tensorflow as tf
import numpy as np
from pdb import set_trace as bp

class VAEModel():
    def __init__(self, model, batch_size, img_size, lr_rate=0.003 , model_name='', log_dir=''):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.model = model()
        self.model.__init__(model_name, image_size = img_size)
        self.weight = tf.placeholder("float32", name='kl_weight')
        self.model_name = model_name
        self.batch_size = batch_size
        self.img_size = img_size
        self.lr_rate = lr_rate
        self.loss = None
        self.train_op = None
        self.summary_op = None
        self.log_dir = log_dir
        #self.writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
        self.build_network()
        self.saver = tf.train.Saver()


    def build_network(self):
        self.image_matrix = tf.placeholder('float32',
                                           [self.batch_size, self.img_size,
                                            self.img_size, 1],
                                           name='input')
        self.z_mean, self.z_std, __ = self.model.encoder(self.image_matrix,
                                                               is_train=True, reuse=False)
        self.z_std = tf.exp(self.z_std)
        #z_stddev = tf.matmul(z_A, tf.transpose(z_A))
        samples = tf.random_normal(tf.shape(self.z_mean), 0., 1., dtype=tf.float32)
        self.guessed_z = self.z_mean + self.z_std*samples

        #guessed_z = z_mean + (z_A_T * samples)
        self.decoder_output, self.y_out_prec = self.model.decoder(self.guessed_z, name="img",
                                                 is_train=True, reuse=False)

        z_mean_valid, z_std_valid, __ = self.model.encoder(self.image_matrix,
                                                                      is_train=False, reuse=True)
        #z_stddev_valid = tf.matmul(z_A_valid, tf.transpose(z_A_valid))
        z_std_valid = tf.exp(z_std_valid)

        samples_valid = tf.random_normal(tf.shape(z_mean_valid), 0., 1., dtype=tf.float32)
        guessed_z_valid = z_mean_valid + z_std_valid*samples_valid
        self.decoder_output_test, self.y_out_prec_test = self.model.decoder(guessed_z_valid,
                                                      name="img", is_train=False, reuse=True)

        #self.autoencoder_loss = 10.*self.vgg19.make_loss_op(self.image_matrix,self.decoder_output)

        l2_loss_1 = tf.reduce_sum(tf.multiply(tf.pow((self.decoder_output - self.image_matrix), 2), self.y_out_prec), [1,2,3])
        l2_loss_2 = tf.reduce_sum(tf.log(self.y_out_prec), [1,2,3])  # tf.reduce_sum(tf.log(y_out_cov),axis=1)
        self.autoencoder_loss = l2_loss_1 - l2_loss_2


        #def l2loss(x,y):
        #summed = tf.reduce_sum((y - x)**2,
        #                   [1, 2, 3])
        #sqrt_summed = tf.sqrt(summed)
        #l2_loss = summed
        #return l2_loss

        #self.autoencoder_loss = losses.l2loss(self.decoder_output, self.image_matrix)

        # self.autoencoder_loss = losses.gaussian_negative_log_likelihood(
        #     self.decoder_output, self.image_matrix, tf.exp(0.5)
        # )
        #self.autoencoder_loss = tf.reduce_sum(self.autoencoder_loss, [1,2,3])

        #self.true_residuals = tf.abs(self.image_matrix-self.decoder_output)
        #self.autoencoder_res_loss = losses.l2loss(self.res, self.true_residuals)
        #
        # residuals = np.abs(self.decoder_output - self.image_matrix)
        # summed = tf.reduce_sum(tf.square(self.res - residuals), axis=[1, 2, 3])
        # sqrt_summed = tf.sqrt(summed + 1e-10)
        # self.autoencoder_res_loss = sqrt_summed

        # 1d KL divergence

        self.latent_loss = 0.5 * tf.reduce_sum(tf.pow(self.z_mean,2) + tf.pow(self.z_std, 2) - 2 * tf.log(self.z_std) - 1, [1,2,3])

        #-0.5 * tf.reduce_sum(1 + tf.log(self.z_std) - tf.pow(self.z_mean, 2) - self.z_std) # losses.kl_loss_1d(self.z_mean, self.z_std)

        #latent_loss = tf.reduce_sum(
        #    tf.square(z_mean) + tf.square(z_stddev) - 2. * tf.log(z_stddev) - 1, [1, 2, 3])
        #return 0.5 * latent_loss

        # nd KL
        #self.latent_loss = losses.kl_cov_gaussian(self.z_mean, self.z_A_T)

        self.loss = tf.reduce_mean(self.autoencoder_loss + self.weight * self.latent_loss) #+ self.autoencoder_res_loss)

        ## validate
        l2_loss_1 = tf.reduce_sum(tf.multiply(tf.pow((self.decoder_output_test - self.image_matrix), 2), self.y_out_prec_test), axis=[1,2,3])
        l2_loss_2 = tf.reduce_sum(tf.log(self.y_out_prec_test), axis=[1,2,3])  # tf.reduce_sum(tf.log(y_out_cov),axis=1)
        self.autoencoder_loss_test = l2_loss_1 - l2_loss_2

        #self.autoencoder_loss_test = losses.l2loss(self.decoder_output_test, self.image_matrix)

        # self.autoencoder_loss_test = losses.gaussian_negative_log_likelihood(self.decoder_output_test,
        #                                                                          self.image_matrix,
        #                                                                          tf.exp(0.5)
        #                                                                          )
        #self.autoencoder_loss_test = tf.reduce_sum(self.autoencoder_loss_test, [1,2,3])
        #self.autoencoder_loss_test = 10.*self.vgg19.make_loss_op(self.image_matrix, self.decoder_output_test)
        #self.true_residuals_test = tf.abs(self.image_matrix - self.decoder_output_test)
        #self.autoencoder_res_loss_test = losses.l2loss(self.res_test, self.true_residuals_test)
        self.latent_loss_test = 0.5 * tf.reduce_sum(tf.pow(z_mean_valid,2) + tf.pow(z_std_valid, 2) - 2 * tf.log(z_std_valid) - 1, [1,2,3])

        #self.latent_loss_test = losses.kl_loss_1d(z_mean_valid, z_std_valid)

        self.loss_test = tf.reduce_mean(self.autoencoder_loss_test + self.weight*self.latent_loss_test)
                                        #+ self.autoencoder_res_loss_test)

        ###### GRADIENTS

        op_p_x_z = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((self.decoder_output - self.image_matrix), 2), self.y_out_prec), axis=[1,2,3]) \
                    + 0.5 * tf.reduce_sum(tf.log(self.y_out_prec), axis=[1,2,3]) - 0.5 * self.img_size * self.img_size * tf.log(2 * np.pi))

        op_q_z_x = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((self.guessed_z - self.z_mean), 2), tf.reciprocal(self.z_std)), axis=[1,2,3]) \
                    - 0.5 * tf.reduce_sum(tf.log(self.z_std), axis=[1,2,3]) - 0.5 * 1 * tf.log(2 * np.pi))

        op_p_z = (- 0.5 * tf.reduce_sum(
            tf.multiply(tf.pow((self.guessed_z - tf.zeros_like(self.z_mean)), 2), tf.reciprocal(tf.ones_like(self.z_std))), \
            axis=[1,2,3]) - 0.5 * tf.reduce_sum(tf.log(tf.ones_like(self.z_std)), axis=[1,2,3]) \
                  - 0.5 * 1 * tf.log(2 * np.pi))

        self.funop = op_p_x_z + op_p_z - op_q_z_x

        grd = tf.gradients(op_p_x_z + op_p_z - op_q_z_x, self.image_matrix)

        self.grd0 = grd[0]

    def initialize(self):
        with tf.device("/GPU:0"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.lr_rate).minimize(self.loss)
                self.train_op = tf.group([self.train_op, update_ops])
        self.sess.run(tf.initializers.global_variables())

    def summarize(self):
        tf.summary.scalar("train_lat_loss", tf.reduce_mean(self.latent_loss))
        # tf.summary.scalar("test_gen_loss", tf.reduce_mean(autoencoder_loss_test))
        tf.summary.scalar("test_lat_loss", tf.reduce_mean(self.latent_loss_test))
        self.summary_op = tf.summary.merge_all()

    def train(self, input_images, weight):
        #input_images, input_masks = next(batches)[:2]
        self.input_images = input_images.astype("float32")
        #input_masks = input_masks.astype("float32")
        feed_dict = {self.image_matrix: input_images,
                     self.weight: weight}
        self.sess.run(self.train_op, feed_dict)

    def validate(self, input_images, weight):
        self.input_images_test = input_images
        feed_dict = {self.image_matrix: self.input_images_test,
                     self.weight: weight}

        self.out_mu_test = self.sess.run(self.decoder_output_test, feed_dict)
        #self.residual_output_test = self.sess.run(self.res_test, feed_dict)

    def save(self,model_name, ep):
        self.saver.save(self.sess, os.path.join(self.log_dir)+'/' + model_name + '_step_'+ str(ep) + ".ckpt",)

    def load(self, model_folder, step=0):
        self.saver.restore(self.sess, model_folder)

    def sample(self):
        z = np.random.normal(0,1,(64,2,2,8*64))
        zeros = np.zeros((64,128,128,1))
        feed_dict = {self.guessed_z: z}
        self.samples = self.sess.run(self.decoder_output, feed_dict)
        return self.samples