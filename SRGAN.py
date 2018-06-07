
# coding: utf-8

# In[ ]:


import os
import numpy as np
import cv2
import torch
from torch.autograd import Variable
import pickle as pk
import logging


# In[ ]:


import visdom


# In[ ]:


# import matplotlib.pyplot as plt
# %matplotlib inline


# In[ ]:


class Option(object):
    def __init__(self):
        pass


# In[ ]:


opt = Option()
opt.max_epochs = 1000
opt.gpu_ids = [0]
opt.task = 'train'
opt.display_port = 8097
opt.num_images_per_batch = 8
opt.save_dir = '/home/guang/SuperRes/pytorch-srmobilegan/snapshots/'
opt.pretrained_state = None #'/home/guang/SuperRes/pytorch-srgan/snapshots/perform_test/state_epoch100_iter0.pkl' #None
opt.snapshot_subdir = 'test01'  # 'nightingale'
opt.snapshot_prefix_G = 'umobilenet256'
opt.snapshot_prefix_D = 'basic'
opt.snapshot_interval_epochs = 20 # In epochs
opt.snapshot_interval_iters = 50
opt.display_interval = 10  # In iterations
opt.display_env = opt.snapshot_subdir
opt.num_average_minibatches = None 
opt.learning_rate = 1e-4
opt.lambda_G = 1e-3
opt.no_lsgan = True
opt.dataset = 'starcraft1688' #'ilsvrc2012'
opt.num_crops_per_image = 2
opt.log_file = os.path.join('/home/guang/SuperRes/pytorch-srmobilegan/snapshots/', 
                            opt.snapshot_subdir,
                            'log.txt')


# In[ ]:


opt.save_dir = os.path.join(opt.save_dir, opt.snapshot_subdir)
if opt.task == 'train':
    if not os.path.isdir(opt.save_dir):
        os.makedirs(opt.save_dir)


logger = logging.getLogger("pytorch-srmobilegan")
logger.setLevel(logging.DEBUG)
logger.propagate = False

log_file = logging.FileHandler(opt.log_file)
log_file.setLevel(logging.DEBUG)

fmt = '%(asctime)s %(levelname)-8s: %(message)s'
fmt = logging.Formatter(fmt)

log_file.setFormatter(fmt)
logger.addHandler(log_file)

logger.info('Pretrained state "{}"'.format(opt.pretrained_state))
logger.info('Snapshot will be saved in "{}"'.format(opt.save_dir))


# In[ ]:





# In[ ]:


from models.network_test import define_D, GANLoss, define_G


# In[ ]:


def save_network(network, basename, gpu_ids):
    save_filename = '{}.pth'.format(basename)
    save_path = os.path.join(opt.save_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    network.cuda(device=opt.gpu_ids[0])
    logger.info('A model was saved to "{}"'.format(save_path))


# In[ ]:


def save_optimizer(optimizer, basename):
    save_filename = '{}.pth'.format(basename)
    save_path = os.path.join(opt.save_dir, save_filename)
    torch.save(optimizer.state_dict(), save_path)
    logger.info('A optimizer state was saved to "{}"'.format(save_path))


# In[ ]:


# def save_history(dict_history, basename):
#     save_filename = os.path.join(opt.save_dir, 'loss_history_{}.pkl'.format(basename))
#     with open(save_filename, 'w') as f:
#         pk.dump(save_filename, f)


# In[ ]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 1e-2)


# In[ ]:


#TODO no_lsgan and use sigmoid
opt.use_sigmoid = opt.no_lsgan
model_D = define_D(6, 64, 'basic', use_sigmoid=opt.use_sigmoid)


# In[ ]:


model_G = define_G(3, 3, 64, which_model_netG='unet_mobilenet_256')


# In[ ]:


loss_history = {
    'L2_loss': [],
    'Gan_G_loss': [], 
    'D_loss_real': [],
    'D_loss_fake': [],
}

if opt.pretrained_state is None:
    state = {
        'loss_history': {
            'L2_loss': [],
            'Gan_G_loss': [], 
            'D_loss_real': [],
            'D_loss_fake': [],
        },
        'model_G': None,
        'model_D': None,
        'optimizer_G': None,
        'optimizer_D': None,        
        
        'epochs': 0
    }
    model_G.apply(weights_init)
    model_D.apply(weights_init)
    model_G.cuda(device=opt.gpu_ids[0])
    model_D.cuda(device=opt.gpu_ids[0])
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=opt.learning_rate)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=opt.learning_rate)

else:
    with open(opt.pretrained_state) as f:
        state = pk.load(f)
    model_G.load_state_dict(torch.load(os.path.join(os.path.dirname(opt.pretrained_state), state['model_G'])))
    model_D.load_state_dict(torch.load(os.path.join(os.path.dirname(opt.pretrained_state), state['model_D'])))
    model_G.cuda(device=opt.gpu_ids[0])
    model_D.cuda(device=opt.gpu_ids[0])
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=opt.learning_rate)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=opt.learning_rate)
    optimizer_G.load_state_dict(torch.load(os.path.join(os.path.dirname(opt.pretrained_state), state['optimizer_G'])))
    optimizer_D.load_state_dict(torch.load(os.path.join(os.path.dirname(opt.pretrained_state), state['optimizer_D'])))   
    
    
    
loss_history = state['loss_history']


# In[ ]:





# In[ ]:


Tensor = torch.cuda.FloatTensor


# In[ ]:


gan_loss = GANLoss(tensor = Tensor, use_lsgan = not opt.no_lsgan)


# In[ ]:


# Load the dataset
if opt.dataset == 'starcraft':
    from datasets.starcraft import StarCraftDataset
    starcraft = StarCraftDataset('/home/guang/datasets/starcraft', N=opt.num_crops_per_image)
    dataset = torch.utils.data.DataLoader(
                starcraft,
                batch_size=opt.num_images_per_batch,
                num_workers=int(1),
                shuffle=True)
    n_image_pairs = len(starcraft)
elif opt.dataset == 'ilsvrc2012':
    from datasets.ilsvrc2012 import ILSVRC2012
    ilsvrc = ILSVRC2012('/home/guang/datasets/ILSVRC2012', N=opt.num_crops_per_image, scales=[2,3,4])
    dataset = torch.utils.data.DataLoader(
                ilsvrc,
                batch_size=opt.num_images_per_batch,
                num_workers=int(1),
                shuffle=True)
elif opt.dataset == 'starcraft1688':
    from datasets.starcraft1688 import StarCraft1688Dataset
    starcraft = StarCraft1688Dataset('/home/guang/datasets/starcraft1688', N=opt.num_crops_per_image)
    dataset = torch.utils.data.DataLoader(
                starcraft,
                batch_size=opt.num_images_per_batch,
                num_workers=int(1),
                shuffle=True)
    n_image_pairs = len(starcraft)    
if opt.num_average_minibatches is None:
    opt.num_average_minibatches = len(dataset)


# In[ ]:


vis = visdom.Visdom(port=opt.display_port)
vis.close()


# In[ ]:


def save_snapshot(model_G, model_D, optimizer_G, optimizer_D, state, epochs, iterations=0, save_state=True):
    epoch_txt = "_epoch{}_iter{}".format(epoch, iterations)
    save_network(model_G, opt.snapshot_prefix_G+epoch_txt, opt.gpu_ids[0])
    save_network(model_D, opt.snapshot_prefix_D+epoch_txt, opt.gpu_ids[0])
    
    if save_state:
        save_optimizer(optimizer_G, opt.snapshot_prefix_G+'_optimizer'+epoch_txt)
        save_optimizer(optimizer_D, opt.snapshot_prefix_D+'_optimizer'+epoch_txt)  
        state['model_G'] = opt.snapshot_prefix_G+epoch_txt+'.pth'
        state['model_D'] = opt.snapshot_prefix_D+epoch_txt+'.pth'
        state['optimizer_G'] = opt.snapshot_prefix_G+'_optimizer'+epoch_txt+'.pth'
        state['optimizer_D'] = opt.snapshot_prefix_D+'_optimizer'+epoch_txt+'.pth'
        state['epochs'] = epochs
        state['iterations'] = iterations
        with open(os.path.join(opt.save_dir, 'state{}.pkl'.format(epoch_txt)), 'w') as f:
                pk.dump(state, f)


# In[ ]:


import progressbar as pb

input_tensor = Tensor(1, 3, 256, 256)
target_tensor = Tensor(1, 3, 256, 256)
# label_fake = Variable(Tensor(opt.batch_size).fill_(0))
# label_real = Variable(Tensor(opt.batch_size).fill_(1))




epoch_start_with = state['epochs']
for epoch in range(epoch_start_with, opt.max_epochs+1):
    if epoch != epoch_start_with and (epoch) % opt.snapshot_interval_epochs == 0:
        save_snapshot(model_G, model_D, optimizer_G, optimizer_D, state, epoch, iterations=0)
        
    if epoch == opt.max_epochs:
        break
    
    widgets = [
        'Epoch {}: '.format(epoch), pb.Percentage(),
        ' ', pb.Bar(marker=pb.AnimatedMarker()),
        ' ', pb.ETA()
    ]
    bar = pb.ProgressBar(widgets=widgets, maxval=len(dataset)).start()

    
    batch_L2_loss = []
    batch_D_loss_real = []
    batch_D_loss_fake = []
    for minibatch_i, data in enumerate(dataset):
        
        if opt.snapshot_interval_epochs <= 1 and minibatch_i > 0 and minibatch_i % opt.snapshot_interval_iters == 0:
            save_snapshot(model_G, model_D, optimizer_G, optimizer_D, state, epoch, minibatch_i, False)
        
        log_msg = 'epoch {}, minibatch {}/{}'.format(epoch, minibatch_i, len(dataset))
        #vis.text(log_msg, win=0)
        # Stacking crops together as different inputs/targets
        input_data = []
        target_data = []
        for crop_i in range(opt.num_crops_per_image):
            input_data.append(data[crop_i*2])
            target_data.append(data[crop_i*2+1])
        input_data = torch.cat(input_data, dim=0)    
        target_data = torch.cat(target_data, dim=0)       
        
        input_tensor.resize_(input_data.size()).copy_(input_data)
        target_tensor.resize_(target_data.size()).copy_(target_data)
        input = Variable(input_tensor)
        target = Variable(target_tensor)
        output = model_G.forward(input)
        D_score_fake = model_D.forward(torch.cat((input, output), 1)) #####
        
        L2_loss = torch.nn.MSELoss()(output, target)
        GAN_loss_G = gan_loss(D_score_fake, True)
        G_loss = L2_loss + GAN_loss_G * opt.lambda_G
    
        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()
        
        batch_L2_loss_cpu = L2_loss.data.cpu().numpy()[0]
        batch_L2_loss.append(batch_L2_loss_cpu)
        log_msg += '\tL2_loss: {:.6f}'.format(batch_L2_loss_cpu)

#
        output = model_G.forward(input)
        D_score_fake = model_D.forward(torch.cat((input, output), 1)) #####
        D_score_real = model_D.forward(torch.cat((input, target), 1))
        D_loss_real = gan_loss(D_score_real, True)
        D_loss_fake = gan_loss(D_score_fake, False)
        D_loss = D_loss_real + D_loss_fake

        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()
        
        D_loss_real_cpu = D_loss_real.data.cpu().numpy()[0]
        D_loss_fake_cpu = D_loss_fake.data.cpu().numpy()[0]        
        
        batch_D_loss_real.append(D_loss_real_cpu)
        batch_D_loss_fake.append(D_loss_fake_cpu)
        
        log_msg += '\tD_loss_real: {:.6f}'.format(D_loss_real_cpu)
        log_msg += '\tD_loss_fake: {:.6f}'.format(D_loss_fake_cpu)
        logger.info(log_msg)
        
        vis.text(log_msg, win=1, env=opt.display_env)
        bar.update(minibatch_i + 1)
        
        if opt.display_interval > 0 and minibatch_i % opt.display_interval==0:
            #input_image = input.data.cpu().numpy()
            
            #vis.images(input_image*0.5 + 0.5, win=0, opts=dict(title='input_{}'.format(display_i), caption='LR'))            
       
            display_egs = torch.cat((input, output, target), dim=3)
            display_egs = display_egs.data.cpu().numpy()
            mean = [ 0.5, 0.5, 0.5 ]
            std = [ 0.5, 0.5, 0.5 ]
            for c in range(3):
                pass
                display_egs[:, c, :, :] *= std[c]
                display_egs[:, c, :, :] += mean[c]

            for display_i in range(min(opt.num_images_per_batch, 6)):
                vis.image(display_egs[display_i,:,:,:], win=100+display_i, env=opt.display_env)
           
    bar.finish()
    
    loss_history['L2_loss'].append(np.array(batch_L2_loss).mean())
    loss_history['D_loss_real'].append(np.array(batch_D_loss_real).mean())
    loss_history['D_loss_fake'].append(np.array(batch_D_loss_fake).mean()) 
    
    
    epoch_loss_message = '------[AVG]------'
    epoch_loss_message += '\t\tL2_loss: {:.6f}\tD_loss_real: {:.6f}\tD_loss_fake: {:.6f}'.format(loss_history['L2_loss'][-1], 
                                                                                  loss_history['D_loss_real'][-1], 
                                                                                  loss_history['D_loss_fake'][-1])
    logger.info(epoch_loss_message)   
    print epoch_loss_message

    vis.line(Y=np.array([loss_history['L2_loss'], 
                         loss_history['D_loss_real'], 
                         loss_history['D_loss_fake']]).T, 
                         X=np.arange(len(loss_history['L2_loss']))+1, win=3, env=opt.display_env, 
             opts={
                'title': 'loss over time',
                'xlabel': 'epoch',
                'ylabel': 'loss' ,
                'ytype': 'log',
                'legend': ['L2','D_real','D_fake']},
    )

