import time
import random
import argparse
import math
import json
from functools import reduce
import operator

import numpy as np
import torch
import torch.optim as optim

from gail.models import *
from gail.dataloader import *

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _train(args):
    # from learning.utils.env import launch_env
    # from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    #     DtRewardWrapper, ActionWrapper, ResizeWrapper
    # from learning.utils.teacher import PurePursuitExpert

    if args.get_samples:
        generate_expert_trajectorys(args)
    
    data = ExpertTrajDataset(args)

    G = Generator(action_dim=2).to(device)
    D = Discriminator(action_dim=2).to(device)
    state_dict = torch.load('models/G_imitate_2.pt'.format(args.checkpoint), map_location=device)
    G.load_state_dict(state_dict)
    # if args.use_checkpoint:
    #     state_dict = torch.load('models/G_{}.pt'.format(args.checkpoint), map_location=device)
    #     G.load_state_dict(state_dict)
    #     state_dict = torch.load('models/D_{}.pt'.format(args.checkpoint), map_location=device)
    #     D.load_state_dict(state_dict)

    D_optimizer = optim.SGD(
        D.parameters(), 
        lr = args.lrD,
        weight_decay=1e-3
        )

    G_optimizer = optim.SGD(
        G.parameters(),
        lr = args.lrG,
        weight_decay=1e-3,
    )

    avg_loss = 0
    avg_g_loss = 0
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.BCELoss()

    validation = args.epochs-1
    writer = SummaryWriter(comment='gail')
    for epoch in range(args.epochs):
        if epoch % int(args.epochs/(args.episodes*0.7)) == 0 and args.batch_size <= args.steps: #if divisible by 7 sample new trajectory?
            rand_int = np.random.randint(0,(args.episodes*0.7))
            observations = torch.FloatTensor(data[rand_int]['observation']).to(device)
            actions =  torch.FloatTensor(data[rand_int]['action']).to(device)

        
        batch_indices = np.random.randint(0, observations.shape[0], (args.batch_size))

        obs_batch = observations[batch_indices]
        act_batch = actions[batch_indices]

        model_actions, values = G(obs_batch)

        ## Update D

        exp_label = torch.full((args.batch_size,1), 1, device=device).float()
        policy_label = torch.full((args.batch_size,1), 0, device=device).float()

        ##Making labels soft
        # exp_label = torch.randn((args.batch_size,1), device=device).float()*0.1 + 1.1
        # exp_label = exp_label.clamp(0.7,1.2)

        # policy_label = torch.randn((args.batch_size,1), device=device).float()*0.1 + 1
        # policy_label = policy_label.clamp(0,0.3)
        ##
        for _ in range(20):

            D_optimizer.zero_grad()

            prob_expert = D(obs_batch,act_batch)
            expert_loss = loss_fn(prob_expert, exp_label)
            # writer.add_scalar("expert D loss", expert_loss, epoch)
            writer.add_scalar("D/expert probability", torch.mean(prob_expert), epoch)

            prob_policy = D(obs_batch,model_actions)
            policy_loss = loss_fn(prob_policy, policy_label)
            # writer.add_scalar("policy D loss", policy_loss, epoch)
            writer.add_scalar("D/policy probability", torch.mean(prob_policy), epoch)

            # loss = (expert_loss + policy_loss)

            loss = -(torch.mean(prob_expert) - torch.mean(prob_policy))

            writer.add_scalar("D/loss", loss, epoch)
            # if epoch % 10:
            loss.backward(retain_graph=True)
            D_optimizer.step()

            for p in D.parameters():
                p.data.clamp_(-0.01,0.01)

        ## Update G

        from torch.autograd import Variable
        import math
        def normal_log_density(x, mean, log_std, std):
            var = std.pow(2)
            log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
            return log_density.sum(1, keepdim=True)

        action_means, action_log_stds, action_stds = G.get_stats(model_actions)
        fixed_log_prob = normal_log_density(Variable(model_actions), action_means, action_log_stds, action_stds).data.clone()
        
        advantages = D(obs_batch,act_batch)-values; 
        
        def get_loss():
            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()
    
    
        def get_kl():
            mean1, log_std1, std1 = G.get_stats(G(obs_batch))
    
            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            return kl.sum(0, keepdim=True)
        
        
        def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
            x = torch.zeros(b.size()).to(device)
            r = b.clone().to(device)
            p = b.clone().to(device)
            rdotr = torch.dot(r, r).to(device)
            for i in range(nsteps):
                _Avp = Avp(p)
                alpha = rdotr / torch.dot(p, _Avp)
                x += alpha * p
                r -= alpha * _Avp
                new_rdotr = torch.dot(r, r)
                betta = new_rdotr / rdotr
                p = r + betta * p
                rdotr = new_rdotr
                if rdotr < residual_tol:
                    break
            return x
        
        def set_flat_params_to(model, flat_params):
            prev_ind = 0
            for param in model.parameters():
                flat_size = int(np.prod(list(param.size())))
                param.data.copy_(
                    flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
                prev_ind += flat_size
                
        def linesearch(model,
                       f,
                       x,
                       fullstep,
                       expected_improve_rate,
                       max_backtracks=10,
                       accept_ratio=.1):
            fval = f().data
            print("fval before", fval.item())
            for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
                xnew = x + stepfrac * fullstep
                set_flat_params_to(model, xnew)
                newfval = f().data
                actual_improve = fval - newfval
                expected_improve = expected_improve_rate * stepfrac
                ratio = actual_improve / expected_improve
                print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())
        
                if ratio.item() > accept_ratio and actual_improve.item() > 0:
                    print("fval after", newfval.item())
                    return True, xnew
            return False, x
        def get_flat_params_from(model):
            params = []
            for param in model.parameters():
                params.append(param.data.view(-1))

            flat_params = torch.cat(params)
            return flat_params
        
        def trpo_step(model, get_loss, get_kl, max_kl, damping):
            loss = get_loss()
            grads = torch.autograd.grad(loss, model.parameters())
            loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
        
            def Fvp(v):
                kl = get_kl()
                kl = kl.mean()
        
                grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
                flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        
                kl_v = (flat_grad_kl * Variable(v)).sum()
                grads = torch.autograd.grad(kl_v, model.parameters())
                flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
        
                return flat_grad_grad_kl + v * damping
        
            stepdir = conjugate_gradients(Fvp, -loss_grad, 10)
        
            shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
        
            lm = torch.sqrt(shs / max_kl)
            fullstep = stepdir / lm[0]
        
            neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
            print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))
        
            prev_params = get_flat_params_from(model)
            success, new_params = linesearch(model, get_loss, prev_params, fullstep,
                                             neggdotstepdir / lm[0])
            set_flat_params_to(model, new_params)
        
            return loss
        
        loss_g = trpo_step(G, get_loss, get_kl, args.max_kl, args.damping)
        
        
        

        G_optimizer.zero_grad()

        # loss_g = -(torch.mean(D(obs_batch,model_actions)))
        # # loss_g = loss_g.mean()
        # loss_g.backward(retain_graph=True)
        G_optimizer.step()

        avg_g_loss = loss_g.item()
        avg_loss = loss.item() 

        writer.add_scalar("G/loss", avg_g_loss, epoch) #should go towards -inf?
        print('epoch %d, D loss=%.3f, G loss=%.3f' % (epoch, avg_loss, avg_g_loss))

        # Periodically save the trained model
        if epoch % 200 == 0:
            torch.save(D.state_dict(), '{}/D2.pt'.format(args.model_directory))
            torch.save(G.state_dict(), '{}/G2.pt'.format(args.model_directory))
        if epoch % 1000 == 0:
            torch.save(D.state_dict(), '{}/D_epoch{}.pt'.format(args.model_directory,epoch))
            torch.save(G.state_dict(), '{}/G_epoch{}.pt'.format(args.model_directory,epoch))
        torch.cuda.empty_cache()
    torch.save(D.state_dict(), '{}/D2.pt'.format(args.model_directory))
    torch.save(G.state_dict(), '{}/G2.pt'.format(args.model_directory))
    # writer.add_graph("generator", G)
    # writer.add_graph("discriminator",D)



def trust_region_loss(model, ref_model, distribution, ref_distribution, loss, threshold):
    # Compute gradients from original loss
    loss.backward()#retain_variables=True)
    g = [param.grad.clone() for param in model.parameters()]
    model.zero_grad()
    
    # KL divergence k ← ∇θ0∙DKL[π(∙|s_i; θ_a) || π(∙|s_i; θ)]
    kl = distribution * (distribution.log() - ref_distribution.log())
    # Compute gradients from (negative) KL loss (increases KL divergence)
    (-kl.mean(1)).backward(retain_variables=True)
    k = [param.grad.clone() for param in model.parameters()]
    model.zero_grad()
    
    # Compute dot products of gradients
    k_dot_g = sum(torch.sum(k_p * g_p) for k_p, g_p in zip(k, g))
    k_dot_k = sum(torch.sum(k_p ** 2) for k_p in k)
    # Compute trust region update
    trust_factor = k_dot_k.data[0] > 0 and (k_dot_g - threshold) / k_dot_k or Variable(torch.zeros(1))
    trust_update = [g_p - trust_factor.expand_as(k_p) * k_p for g_p, k_p in zip(g, k)]
    trust_loss = 0
    for param, trust_update_p in zip(model.parameters(), trust_update):
        trust_loss += (param * trust_update_p).sum()

    return trust_loss