# -*- coding: utf-8 -*-

# Simple VAE to see if VAEs can learn sparsity inducing distributions
# Kerem Tezcan, CVL
# initial: 28.05.2017
# last mod: 30.05.2017 

from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.io
import scipy.optimize as sop
import SimpleITK as sitk
import time
import os
import subprocess
import sys
from datetime import datetime
import tensorflow as tf

from Patcher import Patcher
from vae_models.definevae_original import definevae

def vaerecon(us_ksp_r2, sensmaps, dcprojiter, n=10, lat_dim=60, patchsize=28, contRec='', parfact=10,
             num_iter=302, rescaled=False, half=False, regiter=15, reglmb=0.1, regtype='reg2_dc', usemeth=1, stepsize=1e-4,
             optScale=False, mode=[], chunks40=False, Melmodels='', N4BFcorr=False, z_multip=1.0, directapprox=0, vae_model='',
             logdir='', directapp=0, gt=None):
     print('xxxxxxxxxxxxxxxxxxx contRec is ' + contRec)
     print('xxxxxxxxxxxxxxxxxxx parfact is ' + str(parfact))
     import pickle

     # set parameters
     # ==============================================================================
     np.random.seed(seed=1)

     imsizer = us_ksp_r2.shape[0]
     imrizec = us_ksp_r2.shape[1]

     nsampl = 50  # 0

     # make a network and a patcher to use later
     # ==============================================================================

     x_rec, x_inp, funop, grd0, grd_dir, sess, grd_p_x_z0, grd_p_z0, grd_q_z_x0, grd20, y_out, y_out_prec, z_std_multip, op_q_z_x, mu, std, grd_q_zpl_x_az0, op_q_zpl_x, z_pl, z = definevae(lat_dim=lat_dim, patchsize=patchsize, mode=mode, vae_model=vae_model, batchsize=parfact*nsampl)

     if directapp:
          print('_____DIRECT APPROX_____')
          grd0 = grd_dir

     Ptchr = Patcher(imsize=[imsizer, imrizec], patchsize=patchsize, step=int(patchsize / 2), nopartials=True,
                     contatedges=True)

     nopatches = len(Ptchr.genpatchsizes)
     print("KCT-INFO: there will be in total " + str(nopatches) + " patches.")

     # define the necessary functions
     # ==============================================================================

     def FT(x):
          # inp: [nx, ny]
          # out: [nx, ny, ns]
          return np.fft.fftshift(np.fft.fft2(sensmaps * np.tile(x[:, :, np.newaxis], [1, 1, sensmaps.shape[2]]),
                                             axes=(0, 1)), axes=(0, 1))

     # def tFT(x):
     #        # inp: [nx, ny, ns]
     #        # out: [nx, ny]
     #        tft_x = np.fft.ifft2(np.fft.ifftshift(x, axes=(0, 1)), axes=(0, 1)) * np.conjugate(sensmaps)
     #
     #        rss = np.sqrt(np.sum(np.square(tft_x), axis=2))
     #
     #        rss = rss / (np.sqrt(np.sum(np.square(sensmaps*np.conjugate(sensmaps)),axis=2)) + 0.00000001)
     #
     #        return rss # root-sum-squared

     def tFT(x):
           # inp: [nx, ny, ns]
           # out: [nx, ny]

           temp = np.fft.ifft2(np.fft.ifftshift(x, axes=(0, 1)), axes=(0, 1))
           return np.sum(temp * np.conjugate(sensmaps), axis=2) / (np.sum(sensmaps * np.conjugate(sensmaps), axis=2) + 0.00000001)

     def UFT(x, uspat):
          # inp: [nx, ny], [nx, ny]
          # out: [nx, ny, ns]

          return np.tile(uspat[:, :, np.newaxis], [1, 1, sensmaps.shape[2]]) * FT(x)

     def tUFT(x, uspat):
          # inp: [nx, ny], [nx, ny]
          # out: [nx, ny]

          tmp1 = np.tile(uspat[:, :, np.newaxis], [1, 1, sensmaps.shape[2]])

          return tFT(tmp1 * x)

     def dconst(us):
          # inp: [nx, ny]
          # out: [nx, ny]

          return np.linalg.norm(UFT(us, uspat) - data) ** 2

     def dconst_grad(us):
          # inp: [nx, ny]
          # out: [nx, ny]
          return 2 * tUFT(UFT(us, uspat) - data, uspat)

     def likelihood(us):
          # inp: [parfact,ps*ps]
          # out: parfact
          us = np.abs(us)
          funeval = funop.eval(
               feed_dict={x_rec: np.tile(us, (nsampl, 1)), z_std_multip: z_multip})  # ,x_inp: np.tile(us,(nsampl,1))
          # funeval: [500x1]
          funeval = np.array(np.split(funeval, nsampl, axis=0))  # [nsampl x parfact x 1]
          return np.mean(funeval, axis=0).astype(np.float64)

     def likelihood_grad(us):
          # inp: [parfact, ps*ps]
          # out: [parfact, ps*ps]
          usc = us.copy()
          usabs = np.abs(us)

          grd0eval = grd0.eval(feed_dict={x_rec: np.tile(usabs, (nsampl, 1)),
                                          z_std_multip: z_multip})  # ,x_inp: np.tile(usabs,(nsampl,1))
          # grd0eval: [500x784]
          grd0eval = np.array(np.split(grd0eval, nsampl, axis=0))  # [nsampl x parfact x 784]


          sigmaeval = y_out_prec.eval(feed_dict={x_rec: np.tile(usabs, (nsampl, 1)),
                                          z_std_multip: z_multip})  # ,x_inp: np.tile(usabs,(nsampl,1))
          sigmaeval = np.array(np.split(sigmaeval, nsampl, axis=0))  # [nsampl x parfact x 784]

          mueval = y_out.eval(feed_dict={x_rec: np.tile(usabs, (nsampl, 1)),
                                          z_std_multip: z_multip})  # ,x_inp: np.tile(usabs,(nsampl,1))
          mueval = np.array(np.split(mueval, nsampl, axis=0))  # [nsampl x parfact x 784]

          #vareval = np.std(mueval, axis=0)  # V(MU(X))
          #vareval = np.mean(1/sigmaeval, axis=0)  # M(SIGMA)
          vareval = np.std(grd0eval, axis=0)  # V(SIGMA (X-MU(X)))

          # grd0_var = np.std(grd0eval, axis=0)
          grd0m = np.mean(grd0eval, axis=0)  # [parfact,784]

          #grd0m = usc / np.abs(usc) * grd0m
          where_not_0 = np.where(usc > 0)
          div = usc
          div[where_not_0] = usc[where_not_0] / np.abs(usc)[where_not_0].astype('float')

          grd0m = div * grd0m
          var0m = vareval

          return grd0m, var0m  # .astype(np.float64)

     def likelihood_grad_meth3(us):
          # inp: [parfact, ps*ps]
          # out: [parfact, ps*ps]
          usc = us.copy()
          usabs = np.abs(us)

          mueval = mu.eval(feed_dict={x_rec: np.tile(usabs, (nsampl, 1))})  # ,x_inp: np.tile(usabs,(nsampl,1))

          #          print("===============================================================")
          #          print("===============================================================")
          #          print("===============================================================")
          #          print(mueval)
          #          print("===============================================================")
          #          print("===============================================================")
          #          print("===============================================================")
          #          print(mueval.shape)
          #          print("===============================================================")
          #          print("===============================================================")
          #          print("===============================================================")
          #          print(len(mueval))
          #          print("===============================================================")
          #          print("===============================================================")
          #          print("===============================================================")

          stdeval = std.eval(feed_dict={x_rec: np.tile(usabs, (nsampl, 1))})  # ,x_inp: np.tile(usabs,(nsampl,1))

          zvals = mueval + np.random.rand(mueval.shape[0], mueval.shape[1]) * stdeval

          y_outeval = y_out.eval(feed_dict={z: zvals})
          y_out_preceval = y_out_prec.eval(feed_dict={z: zvals})

          tmp = np.tile(usabs, (nsampl, 1)) - y_outeval
          tmp = (-1) * tmp * y_out_preceval

          # grd0eval: [500x784]
          grd0eval = np.array(np.split(tmp, nsampl, axis=0))  # [nsampl x parfact x 784]
          grd0m = np.mean(grd0eval, axis=0)  # [parfact,784]

          where_not_0 = np.where(usc > 0)
          div = usc
          div[where_not_0] = usc[where_not_0] / np.abs(usc)[where_not_0]
          grd0m = div * grd0m

          return grd0m  # .astype(np.float64)

     def likelihood_grad_patches(ptchs):
          # inp: [np, ps, ps]
          # out: [np, ps, ps]
          # takes set of patches as input and returns a set of their grad.s
          # both grads are in the positive direction

          shape_orig = ptchs.shape

          ptchs = np.reshape(ptchs, [ptchs.shape[0], -1])

          grds = np.zeros([int(np.ceil(ptchs.shape[0] / parfact) * parfact), np.prod(ptchs.shape[1:])],
                          dtype=np.complex64)

          grds_vars = grds.copy()

          extraind = int(np.ceil(ptchs.shape[0] / parfact) * parfact) - ptchs.shape[0]
          ptchs = np.pad(ptchs, ((0, extraind), (0, 0)), mode='edge')

          for ix in range(int(np.ceil(ptchs.shape[0] / parfact))):
               if usemeth == 1:
                    grds[parfact * ix:parfact * ix + parfact, :], grds_vars[parfact * ix:parfact * ix + parfact, :] = likelihood_grad(
                         ptchs[parfact * ix:parfact * ix + parfact, :])
               else:
                    assert (1 == 0)

          grds = grds[0:shape_orig[0], :]

          grds_vars = grds_vars[0:shape_orig[0], :]

          return np.reshape(grds, shape_orig), np.reshape(grds_vars, shape_orig)

     def likelihood_patches(ptchs):
          # inp: [np, ps, ps]
          # out: 1

          fvls = np.zeros([int(np.ceil(ptchs.shape[0] / parfact) * parfact)])

          extraind = int(np.ceil(ptchs.shape[0] / parfact) * parfact) - ptchs.shape[0]
          ptchs = np.pad(ptchs, [(0, extraind), (0, 0), (0, 0)], mode='edge')

          for ix in range(int(np.ceil(ptchs.shape[0] / parfact))):
               fvls[parfact * ix:parfact * ix + parfact] = likelihood(
                    np.reshape(ptchs[parfact * ix:parfact * ix + parfact, :, :], [parfact, -1]))

          fvls = fvls[0:ptchs.shape[0]]

          return np.mean(fvls)

     def full_gradient(image):
          # inp: [nx*nx, 1]
          # out: [nx, ny], [nx, ny]

          # returns both gradients in the respective positive direction.
          # i.e. must

          ptchs = Ptchr.im2patches(np.reshape(image, [imsizer, imrizec]))
          ptchs = np.array(ptchs)

          grd_lik, grd_lik_var = likelihood_grad_patches(ptchs)
          grd_lik = (-1) * Ptchr.patches2im(grd_lik)
          grd_lik_var = Ptchr.patches2im(grd_lik_var)

          grd_dconst = dconst_grad(np.reshape(image, [imsizer, imrizec]))

          return grd_lik + grd_dconst, grd_lik, grd_dconst, grd_lik_var

     def full_funceval(image):
          # inp: [nx*nx, 1]
          # out: [1], [1], [1]

          tmpimg = np.reshape(image, [imsizer, imrizec])

          dc = dconst(tmpimg)

          ptchs = Ptchr.im2patches(np.reshape(image, [imsizer, imrizec]))
          ptchs = np.array(ptchs)

          lik = (-1) * likelihood_patches(np.abs(ptchs))

          return lik + dc, lik, dc

     def tv_proj(phs, mu=0.125, lmb=2, IT=225):
          phs = fb_tv_proj(phs, mu=mu, lmb=lmb, IT=IT)

          return phs

     def fgrad(im):
          imr_x = np.roll(im, shift=-1, axis=0)
          imr_y = np.roll(im, shift=-1, axis=1)
          grd_x = imr_x - im
          grd_y = imr_y - im

          return np.array((grd_x, grd_y))

     def fdivg(im):
          imr_x = np.roll(np.squeeze(im[0, :, :]), shift=1, axis=0)
          imr_y = np.roll(np.squeeze(im[1, :, :]), shift=1, axis=1)
          grd_x = np.squeeze(im[0, :, :]) - imr_x
          grd_y = np.squeeze(im[1, :, :]) - imr_y

          return grd_x + grd_y

     def f_st(u, lmb):

          uabs = np.squeeze(np.sqrt(np.sum(u * np.conjugate(u), axis=0)))

          tmp = 1 - lmb / uabs
          tmp[np.abs(tmp) < 0] = 0

          uu = u * np.tile(tmp[np.newaxis, :, :], [u.shape[0], 1, 1])

          return uu

     def fb_tv_proj(im, u0=0, mu=0.125, lmb=1, IT=15):
          sz = im.shape
          us = np.zeros((2, sz[0], sz[1], IT))
          us[:, :, :, 0] = u0

          for it in range(IT - 1):
               # grad descent step:
               tmp1 = im - fdivg(us[:, :, :, it])
               tmp2 = mu * fgrad(tmp1)

               tmp3 = us[:, :, :, it] - tmp2

               # thresholding step:
               us[:, :, :, it + 1] = tmp3 - f_st(tmp3, lmb=lmb)

               # endfor

          return im - fdivg(us[:, :, :, it + 1])

     def g_tv_eval(x):
          x_re = np.fft.fftshift(np.reshape(x, (imsizer,imrizec, 1)), axes=(0,1))

          data = tf.placeholder(tf.float64, shape=x_re.shape)

          x_tv = tf.image.total_variation(data)
          var_grad = tf.gradients(x_tv, [data])[0]

          var_grad_val = var_grad.eval(feed_dict={data: x_re})

          return np.fft.ifftshift(var_grad_val, axes=(0,1))

     def tv_norm(x):
          """Computes the total variation norm and its gradient. From jcjohnson/cnn-vis."""
          x = np.fft.fftshift(np.reshape(x, (imsizer,imrizec, 1)), axes=(0,1))

          x_diff = x - np.roll(x, -1, axis=1)
          y_diff = x - np.roll(x, -1, axis=0)
          grad_norm2 = x_diff ** 2 + y_diff ** 2 + np.finfo(np.float32).eps
          norm = np.sum(np.sqrt(grad_norm2))
          dgrad_norm = 0.5 / np.sqrt(grad_norm2)
          dx_diff = 2 * x_diff * dgrad_norm
          dy_diff = 2 * y_diff * dgrad_norm
          grad = dx_diff + dy_diff
          grad[:, 1:, :] -= dx_diff[:, :-1, :]
          grad[1:, :, :] -= dy_diff[:-1, :, :]

          return norm, np.reshape(np.fft.ifftshift(grad, axes=(0,1)), [-1])

     # make the data
     # ===============================

     uspat = np.abs(us_ksp_r2) > 0
     uspat = uspat[:, :, 0]
     data = us_ksp_r2

     trpat = np.zeros_like(uspat)
     trpat[:, 120:136] = 1

     # lrphase = np.angle( tUFT(data*trpat[:,:,np.newaxis],uspat) )
     # lrphase = pickle.load(open('/home/ktezcan/unnecessary_stuff/lowresphase','rb'))
     # truephase = pickle.load(open('/home/ktezcan/unnecessary_stuff/truephase','rb'))
     # lrphase = pickle.load(open('/home/ktezcan/unnecessary_stuff/usphase','rb'))
     # lrphase = pickle.load(open('/home/ktezcan/unnecessary_stuff/lrusphase','rb'))
     # lrphase = pickle.load(open('/home/ktezcan/unnecessary_stuff/lrmaskphase','rb'))

     # make the functions for POCS
     # =====================================
     numiter = num_iter

     multip = 0  # 0.1

     alphas = stepsize * np.ones(numiter)  # np.logspace(-4,-4,numiter)

     #     alphas=np.ones_like(np.logspace(-4,-4,numiter))*5e-3

     def feval(im):
          return full_funceval(im)

     def geval(im):
          t1, t2, t3, t4 = full_gradient(im)
          return np.reshape(t1, [-1]), np.reshape(t2, [-1]), np.reshape(t3, [-1]), np.reshape(t4, [-1])

     # initialize data
     recs = np.zeros((imsizer * imrizec, numiter + 2), dtype=complex)

     #     recs[:,0] = np.abs(tUFT(data, uspat).flatten().copy()) #kct

     recs[:, 0] = tUFT(data, uspat).flatten().copy()

     #pickle.dump(recs[:, 0], open(logdir + '_rec_0', 'wb'))
     n4bf = 1

     #     recs[:,0] = np.abs(tUFT(data, uspat).flatten().copy() )*np.exp(1j*lrphase).flatten()

     phaseregvals = []

     # pickle.dump(recs[:,0],open('/scratc_','wb'))

     print('contRec is ' + contRec)
     if contRec != '':
          try:
               print('KCT-INFO: reading from a previous pickle file ' + contRec)
               import pickle
               rr = pickle.load(open(contRec, 'rb'))
               recs[:, 0] = rr[:, -1]
               print('KCT-INFO: initialized to the previous recon from pickle: ' + contRec)
          except:
               print('KCT-INFO: reading from a previous numpy file ' + contRec)
               rr = np.load(contRec)
               recs[:, 0] = rr[:, -1]
               print('KCT-INFO: initialized to the previous recon from numpy: ' + contRec)

     n4biasfields = []

     recsarr = []

     for it in range(0, numiter - 2, 2):
          alpha = alphas[it]

          # first do N times magnitude prior iterations
          # ===============================================
          # ===============================================

          recstmp = recs[:, it].copy()

          ftot, f_lik, f_dc = 0,0,0#feval(recstmp)

          gtot, g_lik, g_dc, g_lik_var = geval(recstmp)

          tvnorm, tvgrad = tv_norm(np.abs(recstmp))

          lambda_lik = 0
          lambda_reg = 1
          recstmp_1 = recstmp - alpha * (lambda_lik * g_lik + lambda_reg * tvgrad)

          recs[:, it + 1] = recstmp_1.copy()

          print("it no: " + str(it) + " f_tot= " + str(ftot) + " f_lik= " + str(f_lik) + ' TV norm= ' + str(
               tvnorm) + " f_dc (1e6)= " +
                str(f_dc / 1e6) + " |g_lik|= " + str(np.linalg.norm(g_lik)) + " |g_dc|= " + str(
               np.linalg.norm(g_dc)) + ' |g_tv|= ' + str(np.linalg.norm(tvgrad)))

          # if it == 0:
          #      pickle.dump(recstmp, open(logdir + '_rec_0', 'wb'))
          #      pickle.dump(g_lik, open(logdir + '_rec_likgrad', 'wb'))
          #      pickle.dump(g_dc, open(logdir + '_rec_dcgrad', 'wb'))
          #      pickle.dump(tvgrad, open(logdir + '_rec_tvgrad', 'wb'))
          #      pickle.dump(tvgrad * g_lik_var, open(logdir + '_rec_tvmulvar', 'wb'))
          #      pickle.dump(g_lik_var, open(logdir + '_rec_var', 'wb'))
          #      exit()
          # now do again a data consistency projection
          # ===============================================
          # ===============================================

          tmp1 = UFT(np.reshape(recs[:, it + 1], [imsizer, imrizec]), (1 - uspat))
          tmp2 = UFT(np.reshape(recs[:, it + 1], [imsizer, imrizec]), (uspat))
          tmp3 = data * uspat[:, :, np.newaxis]

          tmp = tmp1 + multip * tmp2 + (1 - multip) * tmp3
          recs[:, it + 2] = tFT(tmp).flatten()

          #ftot, f_lik, f_dc = feval(recs[:, it + 2])
          #print('f_dc (1e6): ' + str(f_dc / 1e6) + '  perc: ' + str(100 * f_dc / np.linalg.norm(data) ** 2))

          # MSE CHECK
          recon_sli = np.reshape(recs[:, it + 2], (imsizer, imrizec))
          gt = np.reshape(gt, (imsizer, imrizec))

          rss = np.sqrt(
               np.sum(np.square(np.abs(sensmaps * np.tile(recon_sli[:, :, np.newaxis], [1, 1, sensmaps.shape[2]])
                                       )), axis=-1))

          nmse = np.sqrt(((np.fft.fftshift(rss) - gt) ** 2).mean()) / np.sqrt(((gt) ** 2).mean())
          print('NMSE: ', nmse)

     return recs, 0, phaseregvals, n4biasfields

#
# def vaerecon(us_ksp, sensmaps, uspat, lat_dim=60, patchsize=28, contRec='', parfact=10, num_iter=302, regiter=15, reglmb=0, regtype='TV', directapprox=0, usemeth=1, stepsize=1e-4, mode=[], z_multip=1.0, vae_model='', logdir=''):
#      # set parameters
#      #==============================================================================
#      np.random.seed(seed=1)
#
#      imsizer = us_ksp.shape[0] #252#256#252
#      imrizec = us_ksp.shape[1] #308#256#308
#      nsampl = 50
#
#      # make a network and a patcher to use later
#      #==============================================================================
#
#      x_rec, x_inp, funop, grd0, grd_dir, sess, grd_p_x_z0, grd_p_z0, grd_q_z_x0, grd20, y_out, y_out_prec, z_std_multip, op_q_z_x, mu, std, grd_q_zpl_x_az0, op_q_zpl_x, z_pl, z = definevae(lat_dim=lat_dim, patchsize=patchsize, mode=mode, vae_model=vae_model, batchsize=parfact*nsampl)
#
#      if directapprox:
#           print('Direct approx...')
#           grd0 = grd_dir
#
#      Ptchr=Patcher(imsize=[imsizer,imrizec],patchsize=patchsize,step=int(patchsize/2), nopartials=True, contatedges=True)
#
#      nopatches=len(Ptchr.genpatchsizes)
#      print("KCT-INFO: there will be in total " + str(nopatches) + " patches.")
#
#      # define the necessary functions
#      #==============================================================================
#
#      def dconst(us):
#           #inp: [nx, ny]
#           #out: [nx, ny]
#
#           return np.linalg.norm(UFT(us, uspat) / np.sqrt(imsizer * imrizec) - us_ksp) ** 2
#
#      def dconst_grad(us):
#           #inp: [nx, ny]
#           #out: [nx, ny]
#
#           return 2 * tUFT(UFT(us, uspat) / np.sqrt(imsizer * imrizec) - us_ksp, uspat) * np.sqrt(imsizer * imrizec)
#
#      def likelihood(us):
#           #inp: [parfact,ps*ps]
#           #out: parfact
#           us = np.abs(us)
#
#           funeval = funop.eval(feed_dict={x_rec: np.tile(us,(nsampl,1)), z_std_multip: z_multip }) # ,x_inp: np.tile(us,(nsampl,1))
#           #funeval: [500x1]
#           funeval=np.array(np.split(funeval,nsampl,axis=0))# [nsampl x parfact x 1]
#           return np.mean(funeval,axis=0).astype(np.float64)
#
#      def likelihood_grad(us):
#           #inp: [parfact, ps*ps]
#           #out: [parfact, ps*ps]
#
#           usc=us.copy()
#           usabs=np.abs(us)
#
#           grd0eval = grd0.eval(feed_dict={x_rec: np.tile(usabs,(nsampl,1)), z_std_multip: z_multip }) # ,x_inp: np.tile(usabs,(nsampl,1))
#
#           #grd0eval: [500x784]
#           grd0eval=np.array(np.split(grd0eval,nsampl,axis=0))# [nsampl x parfact x 784]
#
#           grd0m=np.mean(grd0eval,axis=0) #[parfact,784]
#
#           grd0m = usc/np.abs(usc)*grd0m
#
#           return grd0m #.astype(np.float64)
#      #
#      # def likelihood_grad_meth3(us):
#      #      #inp: [parfact, ps*ps]
#      #      #out: [parfact, ps*ps]
#      #      usc=us.copy()
#      #      usabs=np.abs(us)
#      #
#      #      mueval = mu.eval(feed_dict={x_rec: np.tile(usabs,(nsampl,1)) }) # ,x_inp: np.tile(usabs,(nsampl,1))
#      #      stdeval = std.eval(feed_dict={x_rec: np.tile(usabs,(nsampl,1)) }) # ,x_inp: np.tile(usabs,(nsampl,1))
#      #
#      #      zvals = mueval + np.random.rand(mueval.shape[0],mueval.shape[1])*stdeval
#      #
#      #      y_outeval = y_out.eval( feed_dict={ z : zvals } )
#      #      y_out_preceval = y_out_prec.eval( feed_dict={ z : zvals } )
#      #
#      #      tmp = np.tile(usabs,(nsampl,1)) - y_outeval
#      #      tmp =  (-1) * tmp * y_out_preceval
#      #
#      #      # grd0eval: [500x784]
#      #      grd0eval = np.array(np.split(tmp,nsampl,axis=0))# [nsampl x parfact x 784]
#      #      grd0m = np.mean(grd0eval,axis=0) #[parfact,784]
#      #
#      #      grd0m = usc/np.abs(usc)*grd0m
#      #
#      #      return grd0m #.astype(np.float64)
#      #
#      def likelihood_grad_patches(ptchs):
#           #inp: [np, ps, ps]
#           #out: [np, ps, ps]
#           #takes set of patches as input and returns a set of their grad.s
#           #both grads are in the positive direction
#
#           shape_orig = ptchs.shape
#           ptchs = np.reshape(ptchs, [ptchs.shape[0], -1] )
#
#           grds = np.zeros([int(np.ceil(float(ptchs.shape[0])/float(parfact))*parfact), np.prod(ptchs.shape[1:])], dtype=np.complex64)
#           extraind = int(np.ceil(float(ptchs.shape[0])/float(parfact))*parfact) - ptchs.shape[0]
#           ptchs = np.pad(ptchs, ((0, extraind), (0, 0)), mode='edge')
#
#           for ix in range(int(np.ceil(float(ptchs.shape[0])/float(parfact)))):
#                if usemeth==1:
#                     grds[parfact*ix:parfact*ix+parfact,:] = likelihood_grad(ptchs[parfact*ix:parfact*ix+parfact,:])
#
#                #elif usemeth==3:
#                     #grds[parfact*ix:parfact*ix+parfact,:]=likelihood_grad_meth3(ptchs[parfact*ix:parfact*ix+parfact,:])
#                else:
#                     assert(1==0)
#
#           grds=grds[0:shape_orig[0],:]
#           return np.reshape(grds, shape_orig)
#
#      def likelihood_patches(ptchs):
#           #inp: [np, ps, ps]
#           #out: 1
#           fvls=np.zeros([int(np.ceil(float(ptchs.shape[0])/float(parfact))*parfact) ])
#
#           extraind=int(np.ceil(float(ptchs.shape[0])/float(parfact))*parfact) - ptchs.shape[0]
#           ptchs=np.pad(ptchs,[ (0,extraind),(0,0), (0,0)  ],mode='edge' )
#
#           for ix in range(int(np.ceil(ptchs.shape[0]/parfact))):
#                fvls[parfact*ix:parfact*ix+parfact] = likelihood(np.reshape(ptchs[parfact*ix:parfact*ix+parfact,:,:],[parfact,-1]) )
#
#           fvls=fvls[0:ptchs.shape[0]]
#
#           return np.mean(fvls)
#
#      def full_gradient(image):
#           #inp: [nx*nx, 1]
#           #out: [nx, ny], [nx, ny]
#           #returns both gradients in the respective positive direction.
#           #i.e. must
#           img_tmp = np.reshape(image, [imsizer, imrizec])
#
#           ptchs = Ptchr.im2patches(img_tmp)
#           ptchs = np.array(ptchs)
#
#           grd_lik = likelihood_grad_patches(ptchs)
#
#           grd_lik = (-1) * Ptchr.patches2im(grd_lik)
#
#           tmp_img = np.repeat(img_tmp[:, :, np.newaxis], sensmaps.shape[-1], axis=2)
#           tmp_img_sens = sensmaps * tmp_img
#
#           grd_dconst = dconst_grad(tmp_img_sens)
#
#           grd_dconst = np.sqrt(np.sum(np.square(np.abs(grd_dconst)), axis=-1)).copy()  # root-sum-squared
#
#           return grd_lik + grd_dconst, grd_lik, grd_dconst
#
#
#      def feval(image):
#           #inp: [nx*nx, 1]
#           #out: [1], [1], [1]
#           tmpimg = np.reshape(image, [imsizer,imrizec])
#           recs_copy = np.repeat(tmpimg[:, :, np.newaxis], sensmaps.shape[-1], axis=2)
#
#           dc = dconst(sensmaps * recs_copy)
#
#           ptchs = Ptchr.im2patches(tmpimg)
#           ptchs = np.array(ptchs)
#
#           lik = (-1)*likelihood_patches(np.abs(ptchs))
#           return lik + dc, lik, dc
#
#      def tv_proj(phs,mu=0.125,lmb=2,IT=225):
#           phs = fb_tv_proj(phs,mu=mu,lmb=lmb,IT=IT)
#           return phs
#
#      def fgrad(im):
#           imr_x = np.roll(im,shift=-1,axis=0)
#           imr_y = np.roll(im,shift=-1,axis=1)
#           grd_x = imr_x - im
#           grd_y = imr_y - im
#           return np.array((grd_x, grd_y))
#
#      def fdivg(im):
#           imr_x = np.roll(np.squeeze(im[0,:,:]),shift=1,axis=0)
#           imr_y = np.roll(np.squeeze(im[1,:,:]),shift=1,axis=1)
#           grd_x = np.squeeze(im[0,:,:]) - imr_x
#           grd_y = np.squeeze(im[1,:,:]) - imr_y
#           return grd_x + grd_y
#
#      def f_st(u,lmb):
#           uabs = np.squeeze(np.sqrt(np.sum(u*np.conjugate(u),axis=0)))
#           tmp=1-lmb/uabs
#           tmp[np.abs(tmp)<0]=0
#           uu = u*np.tile(tmp[np.newaxis,:,:],[u.shape[0],1,1])
#           return uu
#
#      def fb_tv_proj(im, u0=0, mu=0.125, lmb=1, IT=15):
#           sz = im.shape
#           us=np.zeros((2,sz[0],sz[1],IT))
#           us[:,:,:,0] = u0
#
#           for it in range(IT-1):
#                #grad descent step:
#                tmp1 = im - fdivg(us[:,:,:,it])
#                tmp2 = mu*fgrad(tmp1)
#
#                tmp3 = us[:,:,:,it] - tmp2
#                #thresholding step:
#                us[:,:,:,it+1] = tmp3 - f_st(tmp3, lmb=lmb)
#
#           return im - fdivg(us[:,:,:,it+1])
#
#      def reg2_dcproj(usph, magim, niter=100, alpha_reg=0.05, alpha_dc=0.05):
#           # from  Separate Magnitude and Phase Regularization via Compressed Sensing,  Feng Zhao
#           # sph=usph+np.pi
#
#           ims = np.zeros((imsizer,imrizec,niter))
#           grds_reg = np.zeros((imsizer,imrizec,niter))
#           grds_dc = np.zeros((imsizer,imrizec,niter))
#           ims[:,:,0]=usph.copy()
#
#           regval = reg2eval(ims[:,:,0].flatten())
#
#           for ix in range(niter-1):
#               grd_reg = reg2grd(ims[:,:,ix].flatten()).reshape([imsizer,imrizec])  # *alpha*np.real(1j*np.exp(-1j*ims[:,:,ix])*    fdivg(fgrad(np.exp(  1j* ims[:,:,ix]    )))     )
#               grds_reg[:,:,ix]  = grd_reg
#               grd_dc = reg2_dcgrd(ims[:,:,ix].flatten() , magim).reshape([imsizer,imrizec])
#               grds_dc[:,:,ix]  = grd_dc
#               ims[:,:,ix+1] = ims[:,:,ix] + alpha_reg*grd_reg - alpha_dc*grd_dc
#
#               # regval = reg2eval(ims[:,:,ix+1].flatten())
#
#               # ims_re_sens = sensmaps * np.repeat((magim * np.exp(1j * ims[:, :, ix + 1]))[:, :, np.newaxis], sensmaps.shape[-1], axis=2)
#
#               # f_dc = dconst(ims_re_sens)
#
#               # print("norm grad reg: " + str(np.linalg.norm(grd_reg)))
#               # print("norm grad dc: " + str(np.linalg.norm(grd_dc)) )
#               # print("regval: " + str(regval))
#               # print("fdc: (*1e9) {0:.6f}".format(f_dc/1e9))
#
# #          np.save('/home/ktezcan/unnecessary_stuff/phase', ims)
# #          np.save('/home/ktezcan/unnecessary_stuff/grds_reg', grds_reg)
# #          np.save('/home/ktezcan/unnecessary_stuff/grds_dc', grds_dc)
# #          print("SAVED!!!!!!")
#           return ims[:,:,-1]#-np.pi
#
#      def reg2eval(im):
#           #takes in 1d, returns scalar
#           im=im.reshape([imsizer,imrizec])
#           phs = np.exp(1j*im)
#           return np.linalg.norm(fgrad(phs).flatten())
#
#      def reg2grd(im):
#           #takes in 1d, returns 1d
#           im = im.reshape([imsizer,imrizec])
#
#           return -2*np.real(1j*np.exp(-1j*im) * fdivg(fgrad(np.exp(1j * im)))).flatten()
#
#
#      def reg2_dcgrd(phim, magim):
#           #takes in 1d, returns 1d
#           phim=phim.reshape([imsizer,imrizec])
#           magim=magim.reshape([imsizer,imrizec])
#
#           bfestim_re = sensmaps * np.repeat((np.exp(1j*phim)*magim)[:, :, np.newaxis], sensmaps.shape[-1], axis=2)
#           UFT_bfestim = UFT(bfestim_re, uspat) / np.sqrt(imsizer * imrizec)
#
#           tUFT_bfestim = tUFT(UFT_bfestim - us_ksp, uspat) * np.sqrt(imsizer * imrizec)
#           tUFT_bfestim = np.sqrt(np.sum(np.square(np.abs(tUFT_bfestim)), axis=-1)).copy()  # root-sum-squared
#
#           return -2*np.real(1j*np.exp(-1j*phim)*magim * tUFT_bfestim).flatten()
#
#
#      def reg2_proj_ls(usph, niter=100):
#           # from  Separate Magnitude and Phase Regularization via Compressed Sensing,  Feng Zhao
#           # with line search
#           usph=usph+np.pi
#           ims = np.zeros((imsizer,imrizec,niter))
#           ims[:,:,0]=usph.copy()
#           regval = reg2eval(ims[:,:,0].flatten())
#           #print(regval)
#
#           for ix in range(niter-1):
#               currgrd = reg2grd(ims[:,:,ix].flatten())
#               res = sop.minimize_scalar(lambda alpha: reg2eval(ims[:,:,ix].flatten() + alpha * currgrd   ), method='Golden'    )
#               alphaopt = res.x
#
#               print("optimal alpha: " + str(alphaopt) )
#
#               ims[:,:,ix+1] = ims[:,:,ix] + alphaopt*currgrd.reshape([imsizer,imrizec]) # *alpha*np.real(1j*np.exp(-1j*ims[:,:,ix])*    fdivg(fgrad(np.exp(  1j* ims[:,:,ix]    )))     )
#               regval = reg2eval(ims[:,:,ix+1].flatten())
#
#               print("regval: " + str(regval) )
#
#           return ims[:,:,-1]-np.pi
#
#      def geval(img):
#           t1, t2, t3 = full_gradient(img)
#           return np.reshape(t1, [-1]), np.reshape(t2, [-1]), np.reshape(t3, [-1])
#
#      # make the functions for POCS
#      #=====================================
#      multip = 0 #0.1
#
#      alphas = stepsize*np.ones(num_iter) # np.logspace(-4,-4,numiter)
#      # alphas=np.ones_like(np.logspace(-4,-4,numiter))*5e-3
#
#      # initialize data
#      recs = np.zeros((imsizer*imrizec, num_iter+1), dtype=complex)
#
#      m0 = tUFT(us_ksp, uspat) * np.sqrt(imsizer * imrizec)
#
#      recs[:, 0] = np.sqrt(np.sum(np.square(np.abs(m0)), axis=-1)).flatten().copy() # root-sum-squared
#
#      phaseregvals = []
#
#      #pickle.dump(recs[:, 0], open(logdir + '_rec', 'wb'))
#
#      if contRec != '':
#           try:
#                print('KCT-INFO: reading from a previous pickle file '+contRec)
#                rr = pickle.load(open(contRec, 'rb'))
#                recs[:, 0] = rr[:, -1]
#                print('KCT-INFO: initialized to the previous recon from pickle: ' + contRec)
#           except:
#                print('KCT-INFO: reading from a previous numpy file '+contRec)
#                rr=np.load(contRec)
#                recs[:, 0] = rr[:, -1]
#                print('KCT-INFO: initialized to the previous recon from numpy: ' + contRec)
#
#      for it in range(0, num_iter-1, 2):
#           alpha = alphas[it]
#
#           recstmp = recs[:, it].copy()
#
#           n = 0
#           #for ix in range(n):
#
#           ftot, f_lik, f_dc = feval(recstmp)
#
#           gtot, g_lik, g_dc = geval(recstmp)
#
#           print("it no: " + str(it) + " f_tot= " + str(ftot) + " f_lik= " + str(f_lik) + " f_dc (1e6)= " + str(f_dc/1e6) + " |g_lik|= " + str(np.linalg.norm(g_lik)) + " |g_dc|= " + str(np.linalg.norm(g_dc)) )
#
#           if it == 0:
#                pickle.dump(g_lik, open(logdir + '_grad', 'wb'))
#
#           #recs[:, 19] = g_lik
#
#           recstmp = recstmp - alpha * g_lik
#           recs[:, it+1] = recstmp.copy()
#
#           # Now do a  DC projection
#           #recs[:,it+1] = recs[:,it+n]    # skip the DC projection
#
#           # now do a phase projection
#           #===============================================
#           #===============================================
#           #
#           # tmpa = np.abs(np.reshape(recs[:, it+1], [imsizer, imrizec]))
#           # tmpp = np.angle(np.reshape(recs[:, it+1], [imsizer, imrizec]))
#           # tmpatv = tmpa.copy().flatten()
#           #
#           # if reglmb == 0:
#           #      print("skipping phase proj")
#           #      tmpptv=tmpp.copy().flatten()
#           # else:
#           #      if regtype == 'TV':
#           #           tmpptv = tv_proj(tmpp, mu=0.125, lmb=reglmb, IT=regiter).flatten() #0.1, 15
#           #      elif regtype == 'reg2_proj': # Normal
#           #           tmpptv = reg2_dcproj(tmpp, tmpa, alpha_reg=reglmb, alpha_dc=reglmb, niter=100).flatten()
#           #           #print("KCT-dbg: reg2+DC pahse reg value is " + str(regval))
#           #      elif regtype == 'abs':
#           #           tmpptv = np.zeros_like(tmpp).flatten()
#           #      elif regtype == 'reg2_ls':
#           #           tmpptv = reg2_proj_ls(tmpp, niter=regiter).flatten() #0.1, 15
#           #           regval = reg2eval(tmpp)
#           #           phaseregvals.append(regval)
#           #           print("KCT-dbg: pahse reg value is " + str(regval))
#           #      else:
#           #           print("hey mistake!!!!!!!!!!")
#           #
#           #
#           # recs[:,it+n+2] = tmpatv*np.exp(1j*tmpptv)
#
#           # now do again a data consistency projection
#           #===============================================
#           #===============================================
#
#
#           recs_itr = np.reshape(recs[:, it+1], [imsizer, imrizec])
#           recs_sens = sensmaps * np.repeat(recs_itr[:, :, np.newaxis], sensmaps.shape[-1], axis=2)
#
#           tmp1 = UFT(recs_sens, (1 - uspat)) / np.sqrt(imsizer * imrizec)
#           tmp2 = UFT(recs_sens, uspat) / np.sqrt(imsizer * imrizec)
#
#           #tmp1 = UFT(np.reshape(recs[:,it+n+2],[imsizer,imrizec]), (1-uspat)  )
#           #tmp2 = UFT(np.reshape(recs[:,it+n+2],[imsizer,imrizec]), (uspat)  )
#           tmp3 = us_ksp * uspat
#
#           tmp = tmp1 + multip*tmp2 + (1-multip)*tmp3
#
#           #recs[:,it+n+3] = tFT(tmp).flatten()
#           recs[:, it+2] = np.sqrt(np.sum(np.square(np.abs(tFT(tmp))), axis=-1)).flatten().copy() * np.sqrt(imsizer * imrizec)
#
#           ftot, f_lik, f_dc = feval(recs[:,it+2])
#
#           print('f_dc (1e6): ' + str(f_dc/1e6) + '  perc: ' + str(100*f_dc/np.linalg.norm(us_ksp)**2))
#
#      return recs, phaseregvals

#
#
# def ADMM_recon(imo, usfact2us, usfactnet):
#
#      scipy.io.savemat('/home/ktezcan/unnecessary_stuff/imorig.mat', {'im_ori': imo})
#
#      print("starting the ADMMNet recon")
#
#      pars = (" 5*77, "
#      "clear all;"
#      "global usfact2us; "
#      "global usfactnet; "
#      "usfact2us = " + str(usfact2us) + "; "
#      "usfactnet = " + str(usfactnet) + "; "
#      "cd '/home/ktezcan/Code/from_other_people/Deep-ADMM-Net-master/Deep-ADMM-Net-master', "
#      "main_ADMM_Net_test_3, "
#      ""
#      "exit ")
#
#      proc = subprocess.run([" matlab -nosplash -r '"+ pars + "' "], stdout=subprocess.PIPE, shell=True)
#
# #     print(proc.stdout)
#
#      r = scipy.io.loadmat('/home/ktezcan/unnecessary_stuff/rec_image.mat')
#      imout_admm = r['rec_image']
#
#
#      print("ADMMNet recon ended")
#
#      return imout_admm
#
# def TV_recon(imo, uspat):
#
#
#      print("starting the BART TV recon")
#
#      import subprocess
#
#      path = "/scratch_net/bmicdl02/ktezcan/apps/BART/source-bart/bart-0.4.00/python"
#      os.environ["TOOLBOX_PATH"] = "/scratch_net/bmicdl02/ktezcan/apps/BART/source-bart/bart-0.4.00"
#      sys.path.append(path);
#
#      from bart import bart
#      import cfl
#
#
#      #uspatf = pickle.load(open('/home/ktezcan/modelrecon/recon_results/j_pat_'+flname,'rb'))
#
#      print("writing cfl files")
#
#      #cfl.writecfl('/scratch_net/bmicdl02/Data/test/ksp',  data )
#      cfl.writecfl('/scratch_net/bmicdl02/Data/test/ksp',  np.fft.fftshift(np.fft.fft2( (imo)))*uspat /np.sum(np.abs(imo)) )
#      cfl.writecfl('/scratch_net/bmicdl02/Data/test/sens', np.ones(imo.shape))
#
#      #proc = subprocess.run(["/scratch_net/bmicdl02/ktezcan/apps/BART/source-bart/bart-0.4.00//bart pics -R I:0:0.045 -R T:3:0:0.025 -u1 -C20 -i500 -d4 \
#
#      print("cfl files written")
#
#      proc = subprocess.run(["/scratch_net/bmicdl02/ktezcan/apps/BART/source-bart/bart-0.4.00//bart pics -R T:3:0:0.0075 -u1 -C20 -i4500 -d4 \
#                              /scratch_net/bmicdl02/Data/test/ksp /scratch_net/bmicdl02/Data/test/sens \
#                              /scratch_net/bmicdl02/Data/test/imout"], stdout=subprocess.PIPE, shell=True)
#
#      imout_tv=np.fft.ifftshift(cfl.readcfl('/scratch_net/bmicdl02/Data/test/imout'))
#      imout_tv = imout_tv * np.sum(np.abs(imo))/np.sum(np.abs(imout_tv))
#
#      print('BART TV recon ended')
#
#      return imout_tv
#
# def TV_reconu(usksp, uspat):
#
#
#      print("starting the BART TV recon")
#
#      import subprocess
#
#      path = "/scratch_net/bmicdl02/ktezcan/apps/BART/source-bart/bart-0.4.00/python"
#      os.environ["TOOLBOX_PATH"] = "/scratch_net/bmicdl02/ktezcan/apps/BART/source-bart/bart-0.4.00"
#      sys.path.append(path);
#
#      from bart import bart
#      import cfl
#
#
#      #uspatf = pickle.load(open('/home/ktezcan/modelrecon/recon_results/j_pat_'+flname,'rb'))
#
#      print("writing cfl files")
#
#      #cfl.writecfl('/scratch_net/bmicdl02/Data/test/ksp',  data )
#      cfl.writecfl('/scratch_net/bmicdl02/Data/test/ksp', usksp )
#      cfl.writecfl('/scratch_net/bmicdl02/Data/test/sens', np.ones(usksp.shape))
#
#      #proc = subprocess.run(["/scratch_net/bmicdl02/ktezcan/apps/BART/source-bart/bart-0.4.00//bart pics -R I:0:0.045 -R T:3:0:0.025 -u1 -C20 -i500 -d4 \
#
#      print("cfl files written")
#
#      proc = subprocess.run(["/scratch_net/bmicdl02/ktezcan/apps/BART/source-bart/bart-0.4.00//bart pics -R T:3:0:0.0075 -u1 -C20 -i4500 -d4 \
#                              /scratch_net/bmicdl02/Data/test/ksp /scratch_net/bmicdl02/Data/test/sens \
#                              /scratch_net/bmicdl02/Data/test/imout"], stdout=subprocess.PIPE, shell=True)
#
#      imout_tv=np.fft.ifftshift(cfl.readcfl('/scratch_net/bmicdl02/Data/test/imout'))
#
#
#      print('BART TV recon ended')
#
#      return imout_tv
#
#
# #do the matlab DLMRI recon
# #=======================================================================
# #=======================================================================
# #=======================================================================
# #=======================================================================
# #=======================================================================
#
# def DLMRI_recon(imo, uspat):
#
#      def FT (x):
#           #inp: [nx, ny]
#           #out: [nx, ny]
#           return np.fft.fftshift(    np.fft.fft2(  x , axes=(0,1)  ),   axes=(0,1)    )
#
#      def UFT(x, uspat):
#           #inp: [nx, ny], [nx, ny]
#           #out: [nx, ny]
#
#           return uspat*FT(x)
#
#      scipy.io.savemat('/home/ktezcan/unnecessary_stuff/uspat.mat', {'Q1': uspat})
#      dd=UFT(np.fft.fftshift(imo), uspat)
#      scipy.io.savemat('/home/ktezcan/unnecessary_stuff/imku.mat', {'imku': dd})
#
#      pars = (" 5*77, "
#      "clear all;"
#      "DLMRIparams.num = 100; "
#      "DLMRIparams.nu = 100000; "
#      "DLMRIparams.r = 2; "
#      "DLMRIparams.n = 36; "
#      "DLMRIparams.K2 = 36; "
#      "DLMRIparams.N = 200*36; "
#      "DLMRIparams.T0 = round((0.2)*DLMRIparams.n); "
#      "DLMRIparams.KSVDopt = 2; "
#      "DLMRIparams.thr = (0.023)*[2 2 2 2 1.4*ones(1,DLMRIparams.num-4)]; "
#      "DLMRIparams.numiterateKSVD = 15; "
#      "load '/home/ktezcan/unnecessary_stuff/uspat.mat' , "
#      "load '/home/ktezcan/unnecessary_stuff/imku.mat' , "
#      #"disp('CAME HERE 3'), "
#      "addpath '/home/ktezcan/Code/from_other_people/DLMRI/DLMRI_v6/', "
#      "[imo, ps] = DLMRI(imku,Q1,DLMRIparams,0,[],0);"
#      "imo = ifftshift(imo), "
#      "5*7, "
#      #"disp('CAME HERE 4'), "
#      "save '/home/ktezcan/unnecessary_stuff/imotestback.mat','imo', "
#      "exit ")
#
#      print("starting the DLMRI recon")
#
#      proc = subprocess.run([" matlab -nosplash -r '"+ pars + "' "], stdout=subprocess.PIPE, shell=True)
#
#      r = scipy.io.loadmat('/home/ktezcan/unnecessary_stuff/imotestback.mat')
#      imout_dlmri = np.fft.fftshift(r['imo'])
#
#      imout_dlmri = imout_dlmri * np.linalg.norm(imo) / np.linalg.norm(imout_dlmri)
#
#      print('DLMRI recon ended')
#
#      return imout_dlmri
# #
# #
# #def ADMM_recon(imo, usfact):
# #
# #     scipy.io.savemat('/home/ktezcan/unnecessary_stuff/imorig.mat', {'im_ori': imo})
# #
# #     print("starting the ADMMNet recon")
# #
# #     pars = (" 5*77, "
# #     "clear all;"
# #     "global usrat; "
# #     "usrat = " + str(usfact) + "; "
# #     "cd '/home/ktezcan/Code/from_other_people/Deep-ADMM-Net-master/Deep-ADMM-Net-master', "
# #     "main_ADMM_Net_test_2, "
# #     ""
# #     "exit ")
# #
# #     proc = subprocess.run([" matlab -nosplash -r '"+ pars + "' "], stdout=subprocess.PIPE, shell=True)
# #
# #     r = scipy.io.loadmat('/home/ktezcan/unnecessary_stuff/rec_image.mat')
# #     imout_admm = r['rec_image']
# #
# #
# #     print("ADMMNet recon ended")
# #
# #     return imout_admm
#
# def BM3D_MRI_recon(imo, uspat):
#
#      scipy.io.savemat('/home/ktezcan/unnecessary_stuff/imorig.mat', {'im_ori': imo})
#      scipy.io.savemat('/home/ktezcan/unnecessary_stuff/mask.mat', {'Q1': np.fft.fftshift(uspat) })
#
#      print("starting the BM3D MRI recon")
#
#      pars = (" 5*77, "
#      "cd '/home/ktezcan/Code/from_other_people/BM3D_MRI_toolbox/', "
#      "BM3D_MRI_v1_kerem, "
#      ""
#      "exit ")
#
#      proc = subprocess.run([" matlab -nosplash -r '"+ pars + "' "], stdout=subprocess.PIPE, shell=True)
#
#      r = scipy.io.loadmat('/home/ktezcan/unnecessary_stuff/imout_bm3d.mat')
#      imout_bm3d = r['im_rec_BM3D']
#
#      imout_bm3d = imout_bm3d * np.sum(np.abs(imo))/np.sum(np.abs(imout_bm3d))
#
#      print("BM3D MRI recon ended")
#
#      return imout_bm3d
#
#
#
#





