import time
from ops import *
from utils import *
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import tensorflow as tf
import numpy as np
import PIL.Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io as sio
from tensorflow.python import pywrap_tensorflow
import numpy


class StyleGAN(object):

    def __init__(self, sess, args):
        self.phase = args.phase
        self.progressive = args.progressive
        self.model_name = "StyleGAN"
        self.sess = sess
        self.dataset_name = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.update_dir = args.update_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        
        self.iteration = args.iteration * 1000
        self.max_iteration = args.max_iteration * 1000

        self.batch_size = args.batch_size
        self.img_size = args.img_size

        """ Hyper-parameter"""
        self.start_res = args.start_res
        self.resolutions = resolution_list(self.img_size) # [4, 8, 16, 32, 64, 128, 256, 512, 1024 ...]
        self.featuremaps = featuremap_list(self.img_size) # [512, 512, 512, 512, 256, 128, 64, 32, 16 ...]

        if not self.progressive :
            self.resolutions = [self.resolutions[-1]]
            self.featuremaps = [self.featuremaps[-1]]
            self.start_res = self.resolutions[-1]

        self.gpu_num = args.gpu_num

        self.z_dim = 64
        self.w_dim = 64
        self.n_mapping = 8

        self.w_ema_decay = 0.995 # Decay for tracking the moving average of W during training
        self.style_mixing_prob = 0.9 # Probability of mixing styles during training
        self.truncation_psi = 0.7 # Style strength multiplier for the truncation trick
        self.truncation_cutoff = 8 # Number of layers for which to apply the truncation trick

        self.batch_size_base = 4
        self.learning_rate_base = 0.0001#0.001

        self.train_with_trans = {4: False, 8: False, 16: True, 32: True, 64: True, 128: True, 256: True, 512: True, 1024: True}
        self.per_res = get_per_res(self.gpu_num)

        self.end_iteration = get_end_iteration(self.iteration, self.max_iteration, self.train_with_trans, self.resolutions, self.start_res)

        self.g_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        self.d_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

        self.sn = args.sn

        self.print_freq = {4: 300, 8: 500, 16: 300, 32: 500, 64: 700, 128: 1500, 256: 2500, 512: 7000, 1024: 7000}
        self.save_freq = {4: 300, 8: 500, 16: 300, 32: 500, 64: 700, 128: 1500, 256: 2500, 512: 7000, 1024: 7000}

        self.print_freq.update((x, y // self.gpu_num) for x, y in self.print_freq.items())
        self.save_freq.update((x, y // self.gpu_num) for x, y in self.save_freq.items())

     

        self.dataset = load_data(dataset_name=self.dataset_name)
        self.dataset_num = len(self.dataset)

        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        print()

        print("##### Information #####")
        print("# dataset : ", self.dataset_name)
        print("# dataset number : ", self.dataset_num)
        print("# gpu : ", self.gpu_num)
       

        print("# start resolution : ", self.start_res)
        print("# target resolution : ", self.img_size)
 

        print()

    ##################################################################################
    # Generator
    ##################################################################################

    def g_mapping(self, z, n_broadcast, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('g_mapping', reuse=reuse):

            x = pixel_norm(z)
            epsilon = 1e-8
            # run through mapping network
            for ii in range(self.n_mapping):
                with tf.variable_scope('FC_{:d}'.format(ii)):
                    x = fully_connected(x, units=self.w_dim, gain=np.sqrt(2), lrmul=0.01, sn=self.sn)
                    x = apply_bias(x, lrmul=0.01)
                    x = lrelu(x, alpha=0.2)

            with tf.variable_scope('Broadcast'):
                x = tf.tile(x[:, np.newaxis], [1, n_broadcast, 1])
       
        return x

    def g_synthesis(self, w_broadcasted, alpha, resolutions, featuremaps, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('g_synthesis', reuse=reuse):
            coarse_styles, middle_styles, fine_styles = get_style_class(resolutions, featuremaps)
            layer_index = 2

            """ initial layer """
            res = resolutions[0]
            n_f = featuremaps[0]
           
            x = synthesis_const_block(res, w_broadcasted, n_f, self.sn)

            """ remaining layers """
            if self.progressive :
                images_out = torgb(x, res=res, sn=self.sn)
                coarse_styles.pop(res, None)

                # Coarse style [4 ~ 8]
                # pose, hair, face shape
                for res, n_f in coarse_styles.items():
                    x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, sn=self.sn)
                    img = torgb(x, res, sn=self.sn)
                    images_out = upscale2d(images_out)
                    images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)

                    layer_index += 2

                # Middle style [16 ~ 32]
                # facial features, eye
                for res, n_f in middle_styles.items():
                    x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, sn=self.sn)
                    img = torgb(x, res, sn=self.sn)
                    images_out = upscale2d(images_out)
                    images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)

                    layer_index += 2

                # Fine style [64 ~ 1024]
                # color scheme
                for res, n_f in fine_styles.items():

                    x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, sn=self.sn)
                    img = torgb(x, res, sn=self.sn)
                    images_out = upscale2d(images_out)
                    images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)

                    layer_index += 2

            else :
                for res, n_f in zip(resolutions[1:], featuremaps[1:]) :
                    x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, sn=self.sn)

                    layer_index += 2
                images_out = torgb(x, resolutions[-1], sn=self.sn)

            return images_out

    def generator(self, z, alpha, target_img_size, is_training=True, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("generator", reuse=reuse):
            resolutions = resolution_list(target_img_size)
            featuremaps = featuremap_list(target_img_size)

            w_avg = tf.get_variable('w_avg', shape=[self.w_dim],
                                    dtype=tf.float32, initializer=tf.initializers.zeros(),
                                    trainable=False)

            """ mapping layers """
            n_broadcast = len(resolutions) * 2
            

            if is_training:
                """ apply regularization techniques on training """
                # update moving average of w
                w_broadcasted = self.g_mapping(z, n_broadcast)
                w_broadcasted = self.update_moving_average_of_w(w_broadcasted, w_avg)

                # perform style mixing regularization
                w_broadcasted = self.style_mixing_regularization(z, w_broadcasted, n_broadcast, resolutions)
               

            else :
                """ apply truncation trick on evaluation """
                w_broadcasted = z
               # w_broadcasted = self.truncation_trick(n_broadcast, w_broadcasted, w_avg, self.truncation_psi)

            """ synthesis layers """
         
            x = self.g_synthesis(w_broadcasted, alpha, resolutions, featuremaps)

            return x
 
    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, alpha, target_img_size, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("discriminator", reuse=reuse):
            resolutions = resolution_list(target_img_size)
            featuremaps = featuremap_list(target_img_size)
            featurelist = []
            r_resolutions = resolutions[::-1]
            r_featuremaps = featuremaps[::-1]

            """ set inputs """
            x = fromrgb(x_init, r_resolutions[0], r_featuremaps[0], self.sn)

            """ stack discriminator blocks """
            for index, (res, n_f) in enumerate(zip(r_resolutions[:-1], r_featuremaps[:-1])):
                res_next = r_resolutions[index + 1]
                n_f_next = r_featuremaps[index + 1]
 
                x = discriminator_block(x, res, n_f, n_f_next, self.sn)
                if res < 128:
                    featurelist.append(x)
                if self.progressive :
                    x_init = downscale2d(x_init)
                    y = fromrgb(x_init, res_next, n_f_next, self.sn)
                    x = smooth_transition(y, x, res, r_resolutions[0], alpha)

            """ last block """
            res = r_resolutions[-1]
            n_f = r_featuremaps[-1]
 
            logit = discriminator_last_block(x, res, n_f, n_f, self.sn)

            return logit,featurelist

    ##################################################################################
    # Technical skills
    ##################################################################################

    def update_moving_average_of_w(self, w_broadcasted, w_avg):
        with tf.variable_scope('WAvg'):
            batch_avg = tf.reduce_mean(w_broadcasted[:, 0], axis=0)
            update_op = tf.assign(w_avg, lerp(batch_avg, w_avg, self.w_ema_decay))

            with tf.control_dependencies([update_op]):
                w_broadcasted = tf.identity(w_broadcasted)

        return w_broadcasted

    def style_mixing_regularization(self, z, w_broadcasted, n_broadcast, resolutions):
        with tf.name_scope('style_mix'):
            z2 = tf.random_normal(tf.shape(z), dtype=tf.float32)
            w_broadcasted2 = self.g_mapping(z2, n_broadcast)
            layer_indices = np.arange(n_broadcast)[np.newaxis, :, np.newaxis]
            last_layer_index = (len(resolutions)) * 2

            mixing_cutoff = tf.cond(tf.random_uniform([], 0.0, 1.0) < self.style_mixing_prob,
                lambda: tf.random_uniform([], 1, last_layer_index, dtype=tf.int32),
                lambda: tf.constant(last_layer_index, dtype=tf.int32))

            w_broadcasted = tf.where(tf.broadcast_to(layer_indices < mixing_cutoff, tf.shape(w_broadcasted)),
                                     w_broadcasted,
                                     w_broadcasted2)
        return w_broadcasted

    def truncation_trick(self, n_broadcast, w_broadcasted, w_avg, truncation_psi):
        with tf.variable_scope('truncation'):
            layer_indices = np.arange(n_broadcast)[np.newaxis, :, np.newaxis]
            ones = np.ones(layer_indices.shape, dtype=np.float32)
            coefs = tf.where(layer_indices < self.truncation_cutoff, truncation_psi * ones, ones)
            w_broadcasted = lerp(w_avg, w_broadcasted, coefs)

        return w_broadcasted
        
    def create_variable_for_generator(self):
        
        return tf.get_variable('learnable_dlatents',shape=(1,18,self.w_dim),dtype='float32',
            initializer=tf.ones_initializer())
    

    def build_model(self):
        """ Graph """
        if self.phase == 'train' :
            self.d_loss_per_res = {}
            self.g_loss_per_res = {}
            self.generator_optim = {}
            self.discriminator_optim = {}
            self.alpha_summary_per_res = {}
            self.d_summary_per_res = {}
            self.g_summary_per_res = {}
            self.train_fake_images = {}
            self.train_real1_images = {}

            

            for res in self.resolutions[self.resolutions.index(self.start_res):]:
                g_loss_per_gpu = []
                d_loss_per_gpu = []
                train_fake_images_per_gpu = []
                train_real1_images = []


                batch_size = self.batch_size
                global_step = tf.get_variable('global_step_{}'.format(res), shape=[], dtype=tf.float32,
                                              initializer=tf.initializers.zeros(),
                                              trainable=False)
                alpha_const, zero_constant = get_alpha_const(self.iteration // 2, batch_size * self.gpu_num, global_step)

                # smooth transition variable
                do_train_trans = self.train_with_trans[res]

                alpha = tf.get_variable('alpha_{}'.format(res), shape=[], dtype=tf.float32,
                                        initializer=tf.initializers.ones() if do_train_trans else tf.initializers.zeros(),
                                        trainable=False)

                if do_train_trans:
                    alpha_assign_op = tf.assign(alpha, alpha_const)
                else:
                    alpha_assign_op = tf.assign(alpha, zero_constant)

                with tf.control_dependencies([alpha_assign_op]):
                    for gpu_id in range(self.gpu_num):
                        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                                # images
                                gpu_device = '/gpu:{}'.format(gpu_id)
                                
                                inputs = tf.data.Dataset.from_tensor_slices(self.dataset)
                                inputs = inputs. \
                                    apply(shuffle_and_repeat(self.dataset_num)). \
                                    apply(prefetch_to_device(gpu_device, None))
                                    # When using dataset.prefetch, use buffer_size=None to let it detect optimal buffer size

                                inputs_iterator = inputs.make_one_shot_iterator()
                                real_img = inputs_iterator.get_next()                               
                                real_img = preprocess_fit_train_image(real_img, res)                            
                                real_img = Normalize_real(real_img)

                                z = tf.random_normal(shape=[batch_size, self.z_dim])

                                fake_img = self.generator(z, alpha, res)
                                fake_img = Normalize_fake(fake_img)

                                shape1 = tf.shape(real_img)
                                real_img = tf.reshape(real_img,[1,shape1[0],shape1[1],shape1[2]])
                                
                                real_logit,_ = self.discriminator(real_img, alpha, res)
                                fake_logit,_ = self.discriminator(fake_img, alpha, res)

                                # compute loss
                                d_loss, g_loss = compute_loss(real_img, real_logit, fake_logit)

                                d_loss_per_gpu.append(d_loss)
                                g_loss_per_gpu.append(g_loss)
                                train_fake_images_per_gpu.append(fake_img)
                                train_real1_images.append(real_img)
    

                print("Create graph for {} resolution".format(res))

                d_vars, g_vars = filter_trainable_variables(res)

                d_loss = tf.reduce_mean(d_loss_per_gpu)
                g_loss = tf.reduce_mean(g_loss_per_gpu)

                d_lr = self.d_learning_rates.get(res, self.learning_rate_base)
                g_lr = self.g_learning_rates.get(res, self.learning_rate_base)

                if self.gpu_num == 1 :
                    colocate_grad = False
                else :
                    colocate_grad = True

                d_optim = tf.train.AdamOptimizer(d_lr, beta1=0, beta2=0.999, epsilon=1e-8).minimize(d_loss,
                                                                                                     var_list=d_vars,
                                                                                                     colocate_gradients_with_ops=colocate_grad)

                g_optim = tf.train.AdamOptimizer(g_lr, beta1=0, beta2=0.999, epsilon=1e-8).minimize(g_loss,
                                                                                                     var_list=g_vars,
                                                                                                     global_step=global_step,
                                                                                                     colocate_gradients_with_ops=colocate_grad)

                self.discriminator_optim[res] = d_optim
                self.generator_optim[res] = g_optim

                self.d_loss_per_res[res] = d_loss
                self.g_loss_per_res[res] = g_loss

                self.train_fake_images[res] = tf.concat(train_fake_images_per_gpu, axis=0)
                self.train_real1_images[res] = tf.concat(train_real1_images,axis=0)


                """ Summary """
                self.alpha_summary_per_res[res] = tf.summary.scalar("alpha_{}".format(res), alpha)

                self.d_summary_per_res[res] = tf.summary.scalar("d_loss_{}".format(res), self.d_loss_per_res[res])
                self.g_summary_per_res[res] = tf.summary.scalar("g_loss_{}".format(res), self.g_loss_per_res[res])

        elif self.phase == 'edit_test' :
            self.dlatent_edit = tf.placeholder(tf.float32,shape=[self.batch_size ,18,self.z_dim],name="input_domain_A") 
            alpha = tf.constant(0.0, dtype=tf.float32, shape=[])       
            self.fake_images_edit=self.generator(self.dlatent_edit, alpha=alpha, target_img_size=self.img_size,is_training=False)

        elif self.phase == 'optimize_all' :
            self.input_A = tf.placeholder(tf.float32,shape=[self.batch_size ,1024, 1, 1],name="input_A") 
            z = tf.random_normal(shape=[self.batch_size, self.z_dim])
            self.lr = tf.placeholder(tf.float32,name="lr_opti") 
            self.dlatent_mix_opti= self.g_mapping(z, 18)
            self.dlatent_variable_opti = self.create_variable_for_generator()
            
            alpha = tf.constant(0.0, dtype=tf.float32, shape=[])
                     
          
            self.fake_images_opti = self.generator(self.dlatent_variable_opti, alpha=alpha, target_img_size=self.img_size,is_training=False)
            self.fake_images_opti = Normalize_fake(self.fake_images_opti)
            recon_loss = L1_loss(self.input_A, self.fake_images_opti)
          
            fake_norm = tf.sqrt(tf.reduce_sum(tf.square(self.fake_images_opti)))
            input_norm = tf.sqrt(tf.reduce_sum(tf.square(self.input_A)))
            dot_fake_input = tf.reduce_sum(tf.multiply(self.fake_images_opti,self.input_A))
            recon_cos =(1-dot_fake_input/(fake_norm*input_norm))
           
            t_vars = tf.trainable_variables()
            g_vars = [var for var in t_vars if 'generator' in var.name]
            
            self.edit_loss_opti = recon_loss + 0.1*recon_cos           
            self.Edit_optim_opti = tf.train.AdamOptimizer(self.learning_rate_base, beta1=0.99, beta2=0.999).minimize(self.edit_loss_opti, var_list=self.dlatent_variable_opti)           
            self.update_optim_opti = tf.train.AdamOptimizer(self.learning_rate_base, beta1=0.99, beta2=0.999).minimize(self.edit_loss_opti, var_list=g_vars)
            #0.99


    ##################################################################################
    # Train
    ##################################################################################

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=1)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:

            start_res_idx = get_checkpoint_res(checkpoint_counter, self.per_res, self.iteration,
                                               self.start_res, self.img_size, self.gpu_num,
                                               self.end_iteration, self.train_with_trans)

            if not self.progressive :
                start_res_idx = 0

            start_batch_idx = checkpoint_counter

            for res_idx in range(self.resolutions.index(self.start_res), start_res_idx) :
                res = self.resolutions[res_idx]
                batch_size_per_res = self.per_res.get(res, self.batch_size_base) * self.gpu_num

                if self.train_with_trans[res]:
                    if res == self.img_size :
                        iteration = self.end_iteration
                    else :
                        iteration = self.iteration
                else :
                    iteration = self.iteration // 2

                if start_batch_idx - (iteration // batch_size_per_res) < 0:
                    break
                else:
                    start_batch_idx = start_batch_idx - (iteration // batch_size_per_res)
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")

        else:
            start_res_idx = self.resolutions.index(self.start_res)
            start_batch_idx = 0
            counter = 1
            print(" [!] Load failed...")

        start_time = time.time()

        for current_res_num in range(start_res_idx, len(self.resolutions)):

            current_res = self.resolutions[current_res_num]
            batch_size_per_res = self.per_res.get(current_res, self.batch_size_base) * self.gpu_num

            if self.progressive :
                if self.train_with_trans[current_res] :

                    if current_res == self.img_size :
                        current_iter = self.end_iteration // (batch_size_per_res*20)
                    else :
                        current_iter = self.iteration // (batch_size_per_res*2)
                else :
                    current_iter = (self.iteration // 2) // batch_size_per_res

            else :
                current_iter = self.end_iteration

            for idx in range(start_batch_idx, current_iter):

                # update D network
                _, summary_d_per_res, d_loss = self.sess.run([self.discriminator_optim[current_res],
                                                              self.d_summary_per_res[current_res],
                                                              self.d_loss_per_res[current_res]])

                self.writer.add_summary(summary_d_per_res, idx)

                # update G network
                _, summary_g_per_res, summary_alpha, g_loss = self.sess.run([self.generator_optim[current_res],
                                                                             self.g_summary_per_res[current_res],
                                                                             self.alpha_summary_per_res[current_res],
                                                                             self.g_loss_per_res[current_res]])

                self.writer.add_summary(summary_g_per_res, idx)
                self.writer.add_summary(summary_alpha, idx)

                # display training status
                counter += 1

                print("Current res: [%4d] [%6d/%6d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (current_res, idx, current_iter, time.time() - start_time, d_loss, g_loss))

                if np.mod(idx + 1, self.print_freq[current_res]) == 0:
                    samples = self.sess.run(self.train_fake_images[current_res])
                    real1 = self.sess.run(self.train_real1_images[current_res])

                    plt.subplot(411)
                    plt.plot(samples.reshape([current_res]))
                    plt.subplot(412)
                    plt.plot(real1.reshape([current_res]))
                    
                    plt.savefig('./{}/fake_img_{:04d}_{:06d}.png'.format(self.sample_dir, current_res, idx + 1))
                    plt.clf()
                if np.mod(idx + 1, self.save_freq[current_res]) == 0:
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_idx is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_idx = 0

            # save model
            self.save(self.checkpoint_dir, counter)


        # save model for final step
        self.save(self.checkpoint_dir, counter)
    

    @property
    def model_dir(self):

        if self.sn :
            sn = '_sn'
        else :
            sn = ''

        if self.progressive :
            progressive = '_progressive'
        else :
            progressive = ''

        return "{}_{}_{}to{}{}{}".format(self.model_name, self.dataset_name, self.start_res, self.img_size, progressive, sn)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        print(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


    
    def optimize_all(self):
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

        t_vars = tf.trainable_variables()
 
        G_vars = [var for var in t_vars if 'generator' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]
        
        saver_gan = tf.train.Saver(G_vars)
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        print(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            path_gan = os.path.join(checkpoint_dir, ckpt_name)
            print(path_gan)
            saver_gan.restore(self.sess, path_gan )
            #counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            
        else:
            print(" [*] Failed to find a checkpoint")
            exit()

 
        print(" [...] Begin training...")
        #加载数据
        trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        
        data , label = get_hrrp(trainB_dataset[0])
        data=data[300:301,:,:,:]#select random one
          
        dlatent_mix = []
        style_code = []
        
        for i in range(1600):
            dlatent_mix_per = self.sess.run([self.dlatent_mix_opti])
            dlatent_mix.append(dlatent_mix_per)
        dlatent_mix = np.reshape(dlatent_mix,[-1,18,self.w_dim])
        
        result_dir = os.path.join(self.result_dir, self.dataset_name)
        check_folder(result_dir)
        sio.savemat(self.result_dir+'/'+self.dataset_name+'/mstar_mix.mat',{'data':dlatent_mix}) 
       
        
        counter = 0
        training = True
        for bs in range(2):
            k1 = 0          
            batch_data = data
            
            while k1  < 2001:                
                
                train_feed_dict = {
                    self.input_A : batch_data,
                    
                }

                _,_, edit_loss = self.sess.run([self.Edit_optim_opti, self.update_optim_opti,self.edit_loss_opti], feed_dict = train_feed_dict)
                k1 += 1
                counter = counter+1
                print("Image: [%2d] counter:[%2d] edit_loss: [%.8f]" \
                 % (bs, k1,edit_loss))
           
             
        self.save(self.update_dir, counter)
             
        style_code=self.dlatent_variable_opti.eval()
        style_code = np.reshape(style_code, [1,18,self.w_dim])
        sio.savemat(self.result_dir+'/'+self.dataset_name+'/mstar_edit.mat',{'data':style_code}) 


    def edit_test(self):
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()
        trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        
        data , label = get_hrrp(trainB_dataset[0])
        label_out=label[0:1]
       
        dlatent_path = './results/'+self.dataset_name+'/'+'mstar_edit.mat'
        dlatent = get_dlatent(dlatent_path)
        dlatent = dlatent
        
        dlatent_mix_path = './results/'+self.dataset_name+'/'+'mstar_mix.mat'
        dlatent_mix = get_dlatent(dlatent_mix_path)
        dlatent_mix = (dlatent_mix)
        t_vars = tf.trainable_variables() 
        G_vars = [var for var in t_vars if 'generator' in var.name]
        saver_gan = tf.train.Saver(G_vars)
       
        
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(self.update_dir, self.model_dir)
        print(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            path_gan = os.path.join(checkpoint_dir, ckpt_name)
            print(path_gan)
            saver_gan.restore(self.sess, path_gan )
            #counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))s
            print(" [*] Success to read {}".format(ckpt_name))
            
        else:
            print(" [*] Failed to find a checkpoint")
            exit()
     

        length_all = 1600
        

        
        Generated_data = []
        for ii in range(0,length_all): 
           
            k_ii = ii % length_all 
                
            dlatent[0:1,9:12,:] = dlatent_mix[k_ii:k_ii+1,9:12,:]
            #10-14 9-12 8-11 select different layers
                
            cc = dlatent
            train_feed_dict = {self.dlatent_edit : cc}
            samples = self.sess.run([self.fake_images_edit], feed_dict = train_feed_dict)
            Generated_data.append(samples)
        
        Generated_data= np.array(Generated_data)
        Generated_data = np.reshape(Generated_data,[-1,1024,1,1])
        Generated_data = Normalize_G(Generated_data)
    
        Generated_data = np.reshape(Generated_data,[-1,1024])
        label=np.zeros(length_all)+label_out
        sio.savemat(self.result_dir+'/'+self.dataset_name+'/fake_'+'mix'+'.mat',{'data':Generated_data,'label':label})
        
        
        
def Normalize_real(data):
    
    min_ = tf.reduce_min(data,axis=0)
    max_ = tf.reduce_max(data,axis=0)
    data=(data-min_)/(max_-min_)
    return data

def Normalize_fake(data):
    min_ = tf.reduce_min(data,axis=1)
    max_ = tf.reduce_max(data,axis=1)

    data=(data-min_)/(max_-min_)

    return data

def get_hrrp(hrrp_path):    
    mat = sio.loadmat(hrrp_path)
    data = np.reshape(mat['data'],[-1,1024,1,1])
    label = mat['label'].transpose().astype(int)    
    data = Normalize_G(data)
    return data, label

def get_dlatent(dlatent_path):    
    mat = sio.loadmat(dlatent_path)
    data = np.reshape(mat['data'],[-1,18,64])
    return data

def Normalize_G(data):
    min_ = data.min(axis=1)
    max_ = data.max(axis=1)
    for i in range(data.shape[0]):
        '''Min-max scaling'''
        data[i,:] = (data[i,:] - min_[i])/(max_[i] - min_[i])
    return data

    