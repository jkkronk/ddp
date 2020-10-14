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
import pickle

from Patcher import Patcher
from vae_models.definevae_original import definevae
from utils import UFT, tUFT, tFT
import time

def vaerecon(us_ksp, sensmaps, uspat, lat_dim=60, patchsize=28, contRec='', parfact=10, num_iter=302, regiter=15, reglmb=0, regtype='TV', usemeth=1, stepsize=1e-4, mode=[], z_multip=1.0, vae_model='', logdir=''):
     # set parameters
     #==============================================================================
     np.random.seed(seed=1)
     
     imsizer = us_ksp.shape[0] #252#256#252
     imrizec = us_ksp.shape[1] #308#256#308
     
     nsampl = 4

     # make a network and a patcher to use later
     #==============================================================================

     x_rec, x_inp, funop, grd0, sess, grd_p_x_z0, grd_p_z0, grd_q_z_x0, grd20, y_out, y_out_prec, z_std_multip, op_q_z_x, mu, std, grd_q_zpl_x_az0, op_q_zpl_x, z_pl, z = definevae(lat_dim=lat_dim, patchsize=patchsize, mode=mode, vae_model=vae_model, batchsize=100)

     Ptchr=Patcher(imsize=[imsizer,imrizec],patchsize=patchsize,step=int(patchsize/2), nopartials=True, contatedges=True)

     nopatches=len(Ptchr.genpatchsizes)
     print("KCT-INFO: there will be in total " + str(nopatches) + " patches.")

     # define the necessary functions
     #==============================================================================

     def dconst(us):
          #inp: [nx, ny]
          #out: [nx, ny]
          
          return np.linalg.norm(UFT(us, uspat) - us_ksp) **2
     
     def dconst_grad(us):
          #inp: [nx, ny]
          #out: [nx, ny]

          return 2*tUFT(UFT(us, uspat) - us_ksp, uspat)
     
     def likelihood(us):
          #inp: [parfact,ps*ps]
          #out: parfact
          
          us = np.abs(us)
          funeval = funop.eval(feed_dict={x_rec: np.tile(us,(nsampl,1)), z_std_multip: z_multip }) # ,x_inp: np.tile(us,(nsampl,1))
          #funeval: [500x1]
          funeval=np.array(np.split(funeval,nsampl,axis=0))# [nsampl x parfact x 1]
          return np.mean(funeval,axis=0).astype(np.float64)
     
     def likelihood_grad(us):
          #inp: [parfact, ps*ps]
          #out: [parfact, ps*ps]
          
          usc=us.copy()
          usabs=np.abs(us)
          
          
          grd0eval = grd0.eval(feed_dict={x_rec: np.tile(usabs,(nsampl,1)), z_std_multip: z_multip }) # ,x_inp: np.tile(usabs,(nsampl,1))
          
          #grd0eval: [500x784]
          grd0eval=np.array(np.split(grd0eval,nsampl,axis=0))# [nsampl x parfact x 784]
          grd0m=np.mean(grd0eval,axis=0) #[parfact,784]

          grd0m = usc/np.abs(usc)*grd0m
                            

          return grd0m #.astype(np.float64)
     
     def likelihood_grad_meth3(us):
          #inp: [parfact, ps*ps]
          #out: [parfact, ps*ps]
          usc=us.copy()
          usabs=np.abs(us)
          
          mueval = mu.eval(feed_dict={x_rec: np.tile(usabs,(nsampl,1)) }) # ,x_inp: np.tile(usabs,(nsampl,1))
          stdeval = std.eval(feed_dict={x_rec: np.tile(usabs,(nsampl,1)) }) # ,x_inp: np.tile(usabs,(nsampl,1))
          
          zvals = mueval + np.random.rand(mueval.shape[0],mueval.shape[1])*stdeval
          
          y_outeval = y_out.eval( feed_dict={ z : zvals } )
          y_out_preceval = y_out_prec.eval( feed_dict={ z : zvals } )
          
          tmp = np.tile(usabs,(nsampl,1)) - y_outeval
          tmp =  (-1) * tmp * y_out_preceval

          # grd0eval: [500x784]
          grd0eval = np.array(np.split(tmp,nsampl,axis=0))# [nsampl x parfact x 784]
          grd0m = np.mean(grd0eval,axis=0) #[parfact,784]

          grd0m = usc/np.abs(usc)*grd0m

          return grd0m #.astype(np.float64)
     
     def likelihood_grad_patches(ptchs):
          #inp: [np, ps, ps] 
          #out: [np, ps, ps] 
          #takes set of patches as input and returns a set of their grad.s 
          #both grads are in the positive direction
          
          shape_orig = ptchs.shape
          ptchs = np.reshape(ptchs, [ptchs.shape[0], -1] )
          
          grds = np.zeros([int(np.ceil(ptchs.shape[0]/parfact)*parfact), np.prod(ptchs.shape[1:])], dtype=np.complex64)
          extraind = int(np.ceil(ptchs.shape[0]/parfact)*parfact) - ptchs.shape[0]
          ptchs = np.pad(ptchs,( (0,extraind),(0,0)  ), mode='edge' )

          for ix in range(int(np.ceil(ptchs.shape[0]/parfact))):
               if usemeth==1:
                    grds[parfact*ix:parfact*ix+parfact,:]=likelihood_grad(ptchs[parfact*ix:parfact*ix+parfact,:]) 
               elif usemeth==3:
                    grds[parfact*ix:parfact*ix+parfact,:]=likelihood_grad_meth3(ptchs[parfact*ix:parfact*ix+parfact,:]) 
               else:
                    assert(1==0)
                  
          grds=grds[0:shape_orig[0],:]
          return np.reshape(grds, shape_orig)
     
     def likelihood_patches(ptchs):
          #inp: [np, ps, ps] 
          #out: 1
          fvls=np.zeros([int(np.ceil(ptchs.shape[0]/parfact)*parfact) ])
          
          extraind=int(np.ceil(ptchs.shape[0]/parfact)*parfact) - ptchs.shape[0]
          ptchs=np.pad(ptchs,[ (0,extraind),(0,0), (0,0)  ],mode='edge' )

          for ix in range(int(np.ceil(ptchs.shape[0]/parfact))):
               fvls[parfact*ix:parfact*ix+parfact] = likelihood(np.reshape(ptchs[parfact*ix:parfact*ix+parfact,:,:],[parfact,-1]) )
               
          fvls=fvls[0:ptchs.shape[0]]
               
          return np.mean(fvls)

     def full_gradient(image):
          #inp: [nx*nx, 1]
          #out: [nx, ny], [nx, ny]
          #returns both gradients in the respective positive direction.
          #i.e. must 
          
          ptchs = Ptchr.im2patches(np.reshape(image, [imsizer,imrizec]))
          ptchs = np.array(ptchs)

          grd_lik = likelihood_grad_patches(ptchs)

          grd_lik = (-1) * Ptchr.patches2im(grd_lik)

          img_tmp = np.reshape(image, [imsizer, imrizec])

          tmp_img = np.repeat(img_tmp[:, :, np.newaxis], sensmaps.shape[-1], axis=2)
          tmp_img_sens = abs(sensmaps) * tmp_img

          grd_dconst = dconst_grad(tmp_img_sens)

          grd_dconst = np.sqrt(np.sum(np.square(grd_dconst), axis=-1)).copy()  # root-sum-squared

          return grd_lik + grd_dconst, grd_lik, grd_dconst
     
     
     def feval(image):
          #inp: [nx*nx, 1]
          #out: [1], [1], [1]
          tmpimg = np.reshape(image, [imsizer,imrizec])
          recs_copy = np.repeat(tmpimg[:, :, np.newaxis], sensmaps.shape[-1], axis=2)

          dc = dconst(abs(sensmaps) * recs_copy)

          ptchs = Ptchr.im2patches(np.reshape(image, [imsizer,imrizec]))
          ptchs = np.array(ptchs)
          
          lik = (-1)*likelihood_patches(np.abs(ptchs))
          return lik + dc, lik, dc    
     
     def tv_proj(phs,mu=0.125,lmb=2,IT=225):
          phs = fb_tv_proj(phs,mu=mu,lmb=lmb,IT=IT)
          return phs
     
     def fgrad(im):
          imr_x = np.roll(im,shift=-1,axis=0)
          imr_y = np.roll(im,shift=-1,axis=1)
          grd_x = imr_x - im
          grd_y = imr_y - im
          return np.array((grd_x, grd_y))
     
     def fdivg(im):
          imr_x = np.roll(np.squeeze(im[0,:,:]),shift=1,axis=0)
          imr_y = np.roll(np.squeeze(im[1,:,:]),shift=1,axis=1)
          grd_x = np.squeeze(im[0,:,:]) - imr_x
          grd_y = np.squeeze(im[1,:,:]) - imr_y
          return grd_x + grd_y

     def f_st(u,lmb):
          uabs = np.squeeze(np.sqrt(np.sum(u*np.conjugate(u),axis=0)))
          tmp=1-lmb/uabs
          tmp[np.abs(tmp)<0]=0
          uu = u*np.tile(tmp[np.newaxis,:,:],[u.shape[0],1,1])
          return uu
     
     def fb_tv_proj(im, u0=0, mu=0.125, lmb=1, IT=15):
          sz = im.shape
          us=np.zeros((2,sz[0],sz[1],IT))
          us[:,:,:,0] = u0
          
          for it in range(IT-1):
               #grad descent step:
               tmp1 = im - fdivg(us[:,:,:,it])
               tmp2 = mu*fgrad(tmp1)
               
               tmp3 = us[:,:,:,it] - tmp2
               #thresholding step:
               us[:,:,:,it+1] = tmp3 - f_st(tmp3, lmb=lmb)

          return im - fdivg(us[:,:,:,it+1])

     def reg2_dcproj(usph, magim, niter=100, alpha_reg=0.05, alpha_dc=0.05):
          # from  Separate Magnitude and Phase Regularization via Compressed Sensing,  Feng Zhao
          # sph=usph+np.pi

          ims = np.zeros((imsizer,imrizec,niter))
          grds_reg = np.zeros((imsizer,imrizec,niter))
          grds_dc = np.zeros((imsizer,imrizec,niter))
          ims[:,:,0]=usph.copy()

          regval = reg2eval(ims[:,:,0].flatten())

          for ix in range(niter-1):
              grd_reg = reg2grd(ims[:,:,ix].flatten()).reshape([imsizer,imrizec])  # *alpha*np.real(1j*np.exp(-1j*ims[:,:,ix])*    fdivg(fgrad(np.exp(  1j* ims[:,:,ix]    )))     )
              grds_reg[:,:,ix]  = grd_reg
              grd_dc = reg2_dcgrd(ims[:,:,ix].flatten() , magim).reshape([imsizer,imrizec])
              grds_dc[:,:,ix]  = grd_dc
              ims[:,:,ix+1] = ims[:,:,ix] + alpha_reg*grd_reg - alpha_dc*grd_dc

              # regval = reg2eval(ims[:,:,ix+1].flatten())

              # ims_re_sens = sensmaps * np.repeat((magim * np.exp(1j * ims[:, :, ix + 1]))[:, :, np.newaxis], sensmaps.shape[-1], axis=2)

              # f_dc = dconst(ims_re_sens)
              
              # print("norm grad reg: " + str(np.linalg.norm(grd_reg)))
              # print("norm grad dc: " + str(np.linalg.norm(grd_dc)) )
              # print("regval: " + str(regval))
              # print("fdc: (*1e9) {0:.6f}".format(f_dc/1e9))
          
#          np.save('/home/ktezcan/unnecessary_stuff/phase', ims)
#          np.save('/home/ktezcan/unnecessary_stuff/grds_reg', grds_reg)
#          np.save('/home/ktezcan/unnecessary_stuff/grds_dc', grds_dc)
#          print("SAVED!!!!!!")
          return ims[:,:,-1]#-np.pi    
     
     def reg2eval(im):
          #takes in 1d, returns scalar
          im=im.reshape([imsizer,imrizec])
          phs = np.exp(1j*im)
          return np.linalg.norm(fgrad(phs).flatten())
     
     def reg2grd(im):
          #takes in 1d, returns 1d
          im = im.reshape([imsizer,imrizec])

          return -2*np.real(1j*np.exp(-1j*im) * fdivg(fgrad(np.exp(1j * im)))).flatten()
     
     
     def reg2_dcgrd(phim, magim):
          #takes in 1d, returns 1d
          phim=phim.reshape([imsizer,imrizec])
          magim=magim.reshape([imsizer,imrizec])

          bfestim_re = abs(sensmaps) * np.repeat((np.exp(1j*phim)*magim)[:, :, np.newaxis], sensmaps.shape[-1], axis=2)
          UFT_bfestim = UFT(bfestim_re, uspat)

          tUFT_bfestim = tUFT(UFT_bfestim - us_ksp, uspat)
          tUFT_bfestim = np.sqrt(np.sum(np.square(tUFT_bfestim), axis=-1)).copy()  # root-sum-squared

          return -2*np.real(1j*np.exp(-1j*phim)*magim * tUFT_bfestim).flatten()
     
     
     def reg2_proj_ls(usph, niter=100):
          # from  Separate Magnitude and Phase Regularization via Compressed Sensing,  Feng Zhao
          # with line search
          usph=usph+np.pi
          ims = np.zeros((imsizer,imrizec,niter))
          ims[:,:,0]=usph.copy()
          regval = reg2eval(ims[:,:,0].flatten())
          #print(regval)

          for ix in range(niter-1):
              currgrd = reg2grd(ims[:,:,ix].flatten())
              res = sop.minimize_scalar(lambda alpha: reg2eval(ims[:,:,ix].flatten() + alpha * currgrd   ), method='Golden'    )
              alphaopt = res.x

              print("optimal alpha: " + str(alphaopt) )
               
              ims[:,:,ix+1] = ims[:,:,ix] + alphaopt*currgrd.reshape([imsizer,imrizec]) # *alpha*np.real(1j*np.exp(-1j*ims[:,:,ix])*    fdivg(fgrad(np.exp(  1j* ims[:,:,ix]    )))     )
              regval = reg2eval(ims[:,:,ix+1].flatten())

              print("regval: " + str(regval) )
             
          return ims[:,:,-1]-np.pi 

     # make the functions for POCS
     #=====================================
     multip = 0 #0.1
     
     alphas = stepsize*np.ones(num_iter) # np.logspace(-4,-4,numiter)
     # alphas=np.ones_like(np.logspace(-4,-4,numiter))*5e-3
     
     def geval(img):
          t1, t2, t3 = full_gradient(img)
          return np.reshape(t1,[-1]), np.reshape(t2,[-1]), np.reshape(t3,[-1])
     
     # initialize data
     recs = np.zeros((imsizer*imrizec,num_iter+30), dtype=complex)

     m0 = tUFT(us_ksp, uspat)
     recs[:, 0] = np.sqrt(np.sum(np.square(m0), axis=-1)).flatten().copy() # root-sum-squared

     phaseregvals = []

     pickle.dump(recs[:,0], open(logdir + '_rec', 'wb'))

     if contRec != '':
          try:
               print('KCT-INFO: reading from a previous pickle file '+contRec)
               rr = pickle.load(open(contRec, 'rb'))
               recs[:, 0] = rr[:, -1]
               print('KCT-INFO: initialized to the previous recon from pickle: ' + contRec)
          except:
               print('KCT-INFO: reading from a previous numpy file '+contRec)
               rr=np.load(contRec)
               recs[:, 0] = rr[:, -1]
               print('KCT-INFO: initialized to the previous recon from numpy: ' + contRec)

     for it in range(0, num_iter-1, 2):
          alpha=alphas[it]

           # FOR NEERAV:
           # 1. I would do your optimization here as x* = max_phi ELBO(N_phi(|x|)) with x as the current estimate of the image, i.e. recs[:,it]
           # then reform the complex image as x_new = x* * nnp.exp(1i*p.angle(x)) and do the rest of the optimization.
           #
           # 2. the second output here is the value of ELBO: ftot, f_lik, f_dc = feval(recs[:,it+1]).
           # I called f_lik for some other reasons, but it is the ELBO actually.
           #
           # 3. The confusing part is that the VAE works with patches, but you want to do the N_phi on the whole image. I am not sure, but the
           # only possible way seems to be to implement the patching operation in tensorflow, if you want to be able to pass the gradients through it...
           # Alternatively, you can use the full image size VAE, then it becomes nearly trivial...

          # first do N times magnitude prior iterations
          #===============================================
          #===============================================

          #m0 = sensmaps * tUFT(us_ksp, uspat)  # mult sensmaps
          #recs[:, 0] = np.sqrt(np.sum(np.square(m0), axis=-1)).flatten().copy()  # root-sum-squared

          #recstmp = np.reshape(recs[:, it].copy(), [imsizer, imrizec])
          #recstmp = np.repeat(recstmp[:, :, np.newaxis], sensmaps.shape[-1], axis=2)
          #recstmp = sensmaps * recstmp

          recstmp = recs[:, it].copy()

          n = 1
          for ix in range(n):

               ftot, f_lik, f_dc = feval(recstmp)

               gtot, g_lik, g_dc = geval(recstmp)

               print("it no: " + str(it+ix) + " f_tot= " + str(ftot) + " f_lik= " + str(f_lik) + " f_dc (1e6)= " + str(f_dc/1e6) + " |g_lik|= " + str(np.linalg.norm(g_lik)) + " |g_dc|= " + str(np.linalg.norm(g_dc)) )

               recstmp = recstmp - alpha * g_lik
               recs[:,it+ix+1] = recstmp.copy()
           
          # Now do a  DC projection
          recs[:,it+n+1] = recs[:,it+n]    # skip the DC projection

          # now do a phase projection
          #===============================================
          #===============================================

          tmpa = np.abs(np.reshape(recs[:, it+n+1], [imsizer, imrizec]))
          tmpp = np.angle(np.reshape(recs[:, it+n+1], [imsizer, imrizec]))
          tmpatv = tmpa.copy().flatten()
           
          if reglmb == 0:
               print("skipping phase proj")
               tmpptv=tmpp.copy().flatten()
          else:
               if regtype == 'TV':
                    tmpptv = tv_proj(tmpp, mu=0.125, lmb=reglmb, IT=regiter).flatten() #0.1, 15
               elif regtype == 'reg2_proj': # Normal
                    tmpptv = reg2_dcproj(tmpp, tmpa, alpha_reg=reglmb, alpha_dc=reglmb, niter=100).flatten()
                    #print("KCT-dbg: reg2+DC pahse reg value is " + str(regval))
               elif regtype == 'abs':
                    tmpptv = np.zeros_like(tmpp).flatten()
               elif regtype == 'reg2_ls':
                    tmpptv = reg2_proj_ls(tmpp, niter=regiter).flatten() #0.1, 15
                    regval = reg2eval(tmpp)
                    phaseregvals.append(regval)
                    print("KCT-dbg: pahse reg value is " + str(regval))
               else:
                    print("hey mistake!!!!!!!!!!")

           
          recs[:,it+n+2] = tmpatv*np.exp(1j*tmpptv)

          # now do again a data consistency projection
          #===============================================
          #===============================================


          recs_itr = np.reshape(recs[:,it+n+2],[imsizer,imrizec])
          recs_sens = abs(sensmaps) * np.repeat(recs_itr[:, :, np.newaxis], sensmaps.shape[-1], axis=2)

          tmp1 = UFT(recs_sens, (1 - uspat))
          tmp2 = UFT(recs_sens, uspat)

          #tmp1 = UFT(np.reshape(recs[:,it+n+2],[imsizer,imrizec]), (1-uspat)  )
          #tmp2 = UFT(np.reshape(recs[:,it+n+2],[imsizer,imrizec]), (uspat)  )
          tmp3 = us_ksp*uspat
               
          tmp = tmp1 + multip*tmp2 + (1-multip)*tmp3

          #recs[:,it+n+3] = tFT(tmp).flatten()
          fFT_tmp = tFT(tmp)
          recs[:, it+n+3] = np.sqrt(np.sum(np.square(fFT_tmp), axis=-1)).flatten().copy()

          ftot, f_lik, f_dc = feval(recs[:,it+1])

          print('f_dc (1e6): ' + str(f_dc/1e6) + '  perc: ' + str(100*f_dc/np.linalg.norm(us_ksp)**2))

     return recs, phaseregvals



def ADMM_recon(imo, usfact2us, usfactnet):

     scipy.io.savemat('/home/ktezcan/unnecessary_stuff/imorig.mat', {'im_ori': imo})
     
     print("starting the ADMMNet recon")
     
     pars = (" 5*77, "
     "clear all;"
     "global usfact2us; "
     "global usfactnet; "
     "usfact2us = " + str(usfact2us) + "; "
     "usfactnet = " + str(usfactnet) + "; "
     "cd '/home/ktezcan/Code/from_other_people/Deep-ADMM-Net-master/Deep-ADMM-Net-master', " 
     "main_ADMM_Net_test_3, "
     ""
     "exit ")
     
     proc = subprocess.run([" matlab -nosplash -r '"+ pars + "' "], stdout=subprocess.PIPE, shell=True)
     
#     print(proc.stdout)
     
     r = scipy.io.loadmat('/home/ktezcan/unnecessary_stuff/rec_image.mat')
     imout_admm = r['rec_image']
     
     
     print("ADMMNet recon ended")
     
     return imout_admm

def TV_recon(imo, uspat):
     
     
     print("starting the BART TV recon")
     
     import subprocess
     
     path = "/scratch_net/bmicdl02/ktezcan/apps/BART/source-bart/bart-0.4.00/python"
     os.environ["TOOLBOX_PATH"] = "/scratch_net/bmicdl02/ktezcan/apps/BART/source-bart/bart-0.4.00"
     sys.path.append(path);
     
     from bart import bart
     import cfl

     
     #uspatf = pickle.load(open('/home/ktezcan/modelrecon/recon_results/j_pat_'+flname,'rb'))
     
     print("writing cfl files")
     
     #cfl.writecfl('/scratch_net/bmicdl02/Data/test/ksp',  data )  
     cfl.writecfl('/scratch_net/bmicdl02/Data/test/ksp',  np.fft.fftshift(np.fft.fft2( (imo)))*uspat /np.sum(np.abs(imo)) )  
     cfl.writecfl('/scratch_net/bmicdl02/Data/test/sens', np.ones(imo.shape))
     
     #proc = subprocess.run(["/scratch_net/bmicdl02/ktezcan/apps/BART/source-bart/bart-0.4.00//bart pics -R I:0:0.045 -R T:3:0:0.025 -u1 -C20 -i500 -d4 \
     
     print("cfl files written")
                             
     proc = subprocess.run(["/scratch_net/bmicdl02/ktezcan/apps/BART/source-bart/bart-0.4.00//bart pics -R T:3:0:0.0075 -u1 -C20 -i4500 -d4 \
                             /scratch_net/bmicdl02/Data/test/ksp /scratch_net/bmicdl02/Data/test/sens \
                             /scratch_net/bmicdl02/Data/test/imout"], stdout=subprocess.PIPE, shell=True)
     
     imout_tv=np.fft.ifftshift(cfl.readcfl('/scratch_net/bmicdl02/Data/test/imout'))
     imout_tv = imout_tv * np.sum(np.abs(imo))/np.sum(np.abs(imout_tv))
     
     print('BART TV recon ended')
     
     return imout_tv

def TV_reconu(usksp, uspat):
     
     
     print("starting the BART TV recon")
     
     import subprocess
     
     path = "/scratch_net/bmicdl02/ktezcan/apps/BART/source-bart/bart-0.4.00/python"
     os.environ["TOOLBOX_PATH"] = "/scratch_net/bmicdl02/ktezcan/apps/BART/source-bart/bart-0.4.00"
     sys.path.append(path);
     
     from bart import bart
     import cfl

     
     #uspatf = pickle.load(open('/home/ktezcan/modelrecon/recon_results/j_pat_'+flname,'rb'))
     
     print("writing cfl files")
     
     #cfl.writecfl('/scratch_net/bmicdl02/Data/test/ksp',  data )  
     cfl.writecfl('/scratch_net/bmicdl02/Data/test/ksp', usksp )  
     cfl.writecfl('/scratch_net/bmicdl02/Data/test/sens', np.ones(usksp.shape))
     
     #proc = subprocess.run(["/scratch_net/bmicdl02/ktezcan/apps/BART/source-bart/bart-0.4.00//bart pics -R I:0:0.045 -R T:3:0:0.025 -u1 -C20 -i500 -d4 \
     
     print("cfl files written")
                             
     proc = subprocess.run(["/scratch_net/bmicdl02/ktezcan/apps/BART/source-bart/bart-0.4.00//bart pics -R T:3:0:0.0075 -u1 -C20 -i4500 -d4 \
                             /scratch_net/bmicdl02/Data/test/ksp /scratch_net/bmicdl02/Data/test/sens \
                             /scratch_net/bmicdl02/Data/test/imout"], stdout=subprocess.PIPE, shell=True)
     
     imout_tv=np.fft.ifftshift(cfl.readcfl('/scratch_net/bmicdl02/Data/test/imout'))
     
     
     print('BART TV recon ended')
     
     return imout_tv


#do the matlab DLMRI recon
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================

def DLMRI_recon(imo, uspat):
     
     def FT (x):
          #inp: [nx, ny]
          #out: [nx, ny]
          return np.fft.fftshift(    np.fft.fft2(  x , axes=(0,1)  ),   axes=(0,1)    )
     
     def UFT(x, uspat):
          #inp: [nx, ny], [nx, ny]
          #out: [nx, ny]
          
          return uspat*FT(x)
     
     scipy.io.savemat('/home/ktezcan/unnecessary_stuff/uspat.mat', {'Q1': uspat})
     dd=UFT(np.fft.fftshift(imo), uspat)
     scipy.io.savemat('/home/ktezcan/unnecessary_stuff/imku.mat', {'imku': dd})
     
     pars = (" 5*77, "
     "clear all;"
     "DLMRIparams.num = 100; " 
     "DLMRIparams.nu = 100000; "
     "DLMRIparams.r = 2; "
     "DLMRIparams.n = 36; "
     "DLMRIparams.K2 = 36; "
     "DLMRIparams.N = 200*36; "
     "DLMRIparams.T0 = round((0.2)*DLMRIparams.n); "
     "DLMRIparams.KSVDopt = 2; "
     "DLMRIparams.thr = (0.023)*[2 2 2 2 1.4*ones(1,DLMRIparams.num-4)]; "
     "DLMRIparams.numiterateKSVD = 15; "
     "load '/home/ktezcan/unnecessary_stuff/uspat.mat' , "
     "load '/home/ktezcan/unnecessary_stuff/imku.mat' , "
     #"disp('CAME HERE 3'), "
     "addpath '/home/ktezcan/Code/from_other_people/DLMRI/DLMRI_v6/', "
     "[imo, ps] = DLMRI(imku,Q1,DLMRIparams,0,[],0);"
     "imo = ifftshift(imo), "
     "5*7, "
     #"disp('CAME HERE 4'), "
     "save '/home/ktezcan/unnecessary_stuff/imotestback.mat','imo', "
     "exit ")
     
     print("starting the DLMRI recon")
     
     proc = subprocess.run([" matlab -nosplash -r '"+ pars + "' "], stdout=subprocess.PIPE, shell=True)
     
     r = scipy.io.loadmat('/home/ktezcan/unnecessary_stuff/imotestback.mat')
     imout_dlmri = np.fft.fftshift(r['imo'])
     
     imout_dlmri = imout_dlmri * np.linalg.norm(imo) / np.linalg.norm(imout_dlmri)
          
     print('DLMRI recon ended')
     
     return imout_dlmri
#
#
#def ADMM_recon(imo, usfact):
#
#     scipy.io.savemat('/home/ktezcan/unnecessary_stuff/imorig.mat', {'im_ori': imo})
#     
#     print("starting the ADMMNet recon")
#     
#     pars = (" 5*77, "
#     "clear all;"
#     "global usrat; "
#     "usrat = " + str(usfact) + "; "
#     "cd '/home/ktezcan/Code/from_other_people/Deep-ADMM-Net-master/Deep-ADMM-Net-master', " 
#     "main_ADMM_Net_test_2, "
#     ""
#     "exit ")
#     
#     proc = subprocess.run([" matlab -nosplash -r '"+ pars + "' "], stdout=subprocess.PIPE, shell=True)
#     
#     r = scipy.io.loadmat('/home/ktezcan/unnecessary_stuff/rec_image.mat')
#     imout_admm = r['rec_image']
#     
#     
#     print("ADMMNet recon ended")
#     
#     return imout_admm

def BM3D_MRI_recon(imo, uspat):

     scipy.io.savemat('/home/ktezcan/unnecessary_stuff/imorig.mat', {'im_ori': imo})
     scipy.io.savemat('/home/ktezcan/unnecessary_stuff/mask.mat', {'Q1': np.fft.fftshift(uspat) })
     
     print("starting the BM3D MRI recon")
     
     pars = (" 5*77, "
     "cd '/home/ktezcan/Code/from_other_people/BM3D_MRI_toolbox/', " 
     "BM3D_MRI_v1_kerem, "
     ""
     "exit ")
     
     proc = subprocess.run([" matlab -nosplash -r '"+ pars + "' "], stdout=subprocess.PIPE, shell=True)
     
     r = scipy.io.loadmat('/home/ktezcan/unnecessary_stuff/imout_bm3d.mat')
     imout_bm3d = r['im_rec_BM3D']
     
     imout_bm3d = imout_bm3d * np.sum(np.abs(imo))/np.sum(np.abs(imout_bm3d))
     
     print("BM3D MRI recon ended")
     
     return imout_bm3d









