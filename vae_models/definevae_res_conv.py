import os
import tensorflow as tf
import numpy as np
from utils import losses
from utils.batches import plot_batch
from utils import deeploss
from pdb import set_trace as bp

class VAEModel():
    def __init__(self, model, batch_size, img_size, lr_rate, model_name, log_dir):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.model = model()
        self.model.__init__(model_name, image_size = config["spatial_size"])
        self.weight = tf.placeholder("float32", name='kl_weight')
        self.model_name = model_name
        self.batch_size = batch_size
        self.img_size = img_size
        self.lr_rate = lr_rate
        self.loss = None
        self.train_op = None
        self.summary_op = None
        self.log_dir = log_dir
        self.writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
        self.build_network()
        self.saver = tf.train.Saver()


    def build_network(self):
        self.image_matrix = tf.placeholder('float32',
                                           [self.batch_size, self.img_size,
                                            self.img_size, 1],
                                           name='input')
        self.z_mean, self.z_std, self.res = self.model.encoder(self.image_matrix,
                                                               is_train=True, reuse=False)
        self.z_std = tf.exp(self.z_std)
        #z_stddev = tf.matmul(z_A, tf.transpose(z_A))
        samples = tf.random_normal(tf.shape(self.z_mean), 0., 1., dtype=tf.float32)
        self.guessed_z = self.z_mean + self.z_std*samples

        #guessed_z = z_mean + (z_A_T * samples)
        self.decoder_output = self.model.decoder(self.guessed_z, name="img",
                                                 is_train=True, reuse=False)

        z_mean_valid, z_std_valid, self.res_test = self.model.encoder(self.image_matrix,
                                                                      is_train=False, reuse=True)
        #z_stddev_valid = tf.matmul(z_A_valid, tf.transpose(z_A_valid))
        z_std_valid = tf.exp(z_std_valid)

        samples_valid = tf.random_normal(tf.shape(z_mean_valid), 0., 1., dtype=tf.float32)
        guessed_z_valid = z_mean_valid + z_std_valid*samples_valid
        self.decoder_output_test = self.model.decoder(guessed_z_valid,
                                                      name="img", is_train=False, reuse=True)

        #self.autoencoder_loss = 10.*self.vgg19.make_loss_op(self.image_matrix,self.decoder_output)
        self.autoencoder_loss = losses.l2loss(self.decoder_output, self.image_matrix)
        # self.autoencoder_loss = losses.gaussian_negative_log_likelihood(
        #     self.decoder_output, self.image_matrix, tf.exp(0.5)
        # )
        #self.autoencoder_loss = tf.reduce_sum(self.autoencoder_loss, [1,2,3])

        self.true_residuals = tf.abs(self.image_matrix-self.decoder_output)
        self.autoencoder_res_loss = losses.l2loss(self.res, self.true_residuals)
        #
        # residuals = np.abs(self.decoder_output - self.image_matrix)
        # summed = tf.reduce_sum(tf.square(self.res - residuals), axis=[1, 2, 3])
        # sqrt_summed = tf.sqrt(summed + 1e-10)
        # self.autoencoder_res_loss = sqrt_summed

        # 1d KL divergence
        self.latent_loss = losses.kl_loss_1d(self.z_mean, self.z_std)

        # nd KL
        #self.latent_loss = losses.kl_cov_gaussian(self.z_mean, self.z_A_T)

        self.loss = tf.reduce_mean(self.autoencoder_loss + self.weight*self.latent_loss + self.autoencoder_res_loss)

        ## validate
        self.autoencoder_loss_test = losses.l2loss(self.decoder_output_test, self.image_matrix)

        # self.autoencoder_loss_test = losses.gaussian_negative_log_likelihood(self.decoder_output_test,
        #                                                                          self.image_matrix,
        #                                                                          tf.exp(0.5)
        #                                                                          )
        #self.autoencoder_loss_test = tf.reduce_sum(self.autoencoder_loss_test, [1,2,3])
        #self.autoencoder_loss_test = 10.*self.vgg19.make_loss_op(self.image_matrix, self.decoder_output_test)
        self.true_residuals_test = tf.abs(self.image_matrix - self.decoder_output_test)
        self.autoencoder_res_loss_test = losses.l2loss(self.res_test, self.true_residuals_test)

        self.latent_loss_test = losses.kl_loss_1d(z_mean_valid, z_std_valid)

        self.loss_test = tf.reduce_mean(self.autoencoder_loss_test + self.weight*self.latent_loss_test
                                        + self.autoencoder_res_loss_test)

    def initialize(self):
        with tf.device("/gpu:0"):
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
        self.residual_output_test = self.sess.run(self.res_test, feed_dict)
    #
    # def visualize(self, model_name, ep):
    #     samples = self.sample()
    #     feed_dict = {self.image_matrix: self.input_images}
    #     self.out_mu = self.sess.run(self.decoder_output, feed_dict)
    #     # out_mu[input_masks==0.]=-3.5
    #     # out_std = sess.run(decoder_std, {image_matrix:input_images})
    #     # out_std = np.exp(out_std)
    #     self.residual_output = self.sess.run(self.res, feed_dict)
    #     if not os.path.exists('Results/'+ model_name + '_samples/'):
    #         os.makedirs('Results/'+ model_name + '_samples/')
    #     model_name = 'Results/' + model_name
    #     plot_batch(self.input_images, model_name + '_samples/gr_' + str(ep) + '.png')
    #     plot_batch(self.out_mu, model_name + '_samples/gn_mu_' + str(ep) + '.png')
    #     # plot_batch(decoded_embedding[:, :, :, np.newaxis].astype("uint8"),
    #     #            os.path.join(wd, model_name + "_samples/embed_" + str(ep) + ".png"))
    #     plot_batch(self.residual_output, model_name + '_samples/res_' + str(ep) + '.png')
    #     plot_batch(np.abs(self.input_images - self.out_mu), model_name + '_samples/gtres_' + str(ep) + '.png')
    #     plot_batch(samples, model_name + '_samples/generated_' + str(ep) + '.png')
    #
    #     plot_batch(self.input_images_test, model_name + '_samples/test_gr_' + str(ep) + '.png')
    #     plot_batch(self.out_mu_test, model_name + '_samples/test_gn_mu_' + str(ep) + '.png')
    #     # plot_batch(decoded_embedding[:, :, :, np.newaxis].astype("uint8"),
    #     #            os.path.join(wd, model_name + "_samples/embed_" + str(ep) + ".png"))
    #     plot_batch(self.residual_output_test, model_name + '_samples/test_res_' + str(ep) + '.png')
    #     plot_batch(np.abs(self.input_images_test - self.out_mu_test), model_name + '_samples/test_gtres_' + str(ep) + '.png')

    def save(self,model_name, ep):
        if not os.path.exists(os.path.join(self.log_dir, model_name)):
            os.makedirs(os.path.join(self.log_dir, model_name))
        self.saver.save(self.sess, os.path.join(self.log_dir, model_name)+'/' + model_name + ".ckpt", global_step=ep)

    def load(self, model_name, step):
        model_folder = os.path.join(self.log_dir, model_name)
        self.saver.restore(self.sess, model_folder + '/' + model_name + ".ckpt-" + str(step))

    def sample(self):
        z = np.random.normal(0,1,(64,2,2,8*64))
        zeros = np.zeros((64,128,128,1))
        feed_dict = {self.guessed_z: z}
        self.samples = self.sess.run(self.decoder_output, feed_dict)
        return self.samples