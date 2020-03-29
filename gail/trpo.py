import numpy as np

import torch

from torch.autograd import Variable


# def trpo_step(model, cost_function, env, num_episodes, num_steps):

    
#     observations = []
#     actions = []

#     ## Collect set of trajectories
#     for i in range(num_episodes):
#         obs = env.reset()
#         for j in range(num_steps):
#             obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)
#             action = model(obs)
#             action = action.squeeze().data.cpu().numpy()
#             obs, reward, done, info = env.step(action)
#             observations.append(obs)
#             actions.append(action)
#     env.close()

#     observations = torch.FloatTensor(observations)
#     actions = torch.FloatTensor(actions)

#     ## compute rewards
#     cost_function(observations,actions)

    
#     ## Compute Advantage estimates()


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