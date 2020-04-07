from gail.models import Encoder, Decoder
import numpy as np
import torch
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_vae(args):
    from learning.utils.env import launch_env
    from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
        DtRewardWrapper, ActionWrapper, ResizeWrapper, CartPole_Pixel
    from learning.utils.teacher import PurePursuitExpert, CartpoleController


    # env = launch_env()
    # env = ResizeWrapper(env)
    # env = NormalizeWrapper(env) 
    # env = ImgWrapper(env)
    # env = ActionWrapper(env)
    # env = DtRewardWrapper(env)

    env = launch_env(args.env_name)
    if args.env_name == 'duckietown':
        env = ResizeWrapper(env)
        env = NormalizeWrapper(env) 
        env = ImgWrapper(env)
        env = ActionWrapper(env)
        env = DtRewardWrapper(env)
        action_dim = 2
        expert = PurePursuitExpert(env=env)
    else:
        env = CartPole_Pixel(env)
        # print(env.reset())
        action_dim = 1
        expert = CartpoleController(env=env)


    

    observations = []

    E = Encoder(env.observation_space.shape).to(device)
    D = Decoder(env.observation_space.shape).to(device)

    e_state_dict = torch.load('{}/vae-encoder'.format(args.env_name))
    E.load_state_dict(e_state_dict)
    state_dict = torch.load('{}/vae-decoder'.format(args.env_name))
    D.load_state_dict(state_dict)

    optimE = torch.optim.Adam(E.parameters(),lr=1e-6)
    optimD = torch.optim.Adam(D.parameters(),lr=1e-6)

    for episode in range(0, 50):
        while True:
            try:
                print("Starting episode", episode)
                obs = env.reset() 
                for steps in range(0, 10):
                    # use our 'expert' to predict the next action.
                    action = expert.predict(None)
                    observation, reward, done, info = env.step(action)
                    observations.append(observation)
                # env.reset()
                break
            except:
                pass
        
    env.close()

    observations = np.array(observations)

    last_loss = 100000
    d_loss = 100000
    ep = 0
    newshape = env.observation_space.shape
    newshape = (newshape[1], newshape[2], newshape[0])
    while d_loss > 1e-5:
        optimE.zero_grad()
        optimD.zero_grad()
        batch_indices = np.random.randint(0, observations.shape[0], (args.batch_size))
        obs_batch = torch.FloatTensor(observations[batch_indices]).float().to(device).data

        loss = (obs_batch.view(obs_batch.size(0),-1) - D(E(obs_batch))).norm(2)

        print("epoch {}: loss {}".format(ep, loss))
        loss.backward()
        optimD.step()
        optimE.step()
        ep += 1
        d_loss =abs(last_loss - loss.data)
        last_loss = loss.data

        if ep %50 == 0:
            torch.save(E.state_dict(), '{}/vae-encoder'.format(args.env_name))
            torch.save(D.state_dict(), '{}/vae-decoder'.format(args.env_name))
        if ep % 50 == 0:
            plt.figure(0)
            plt.imshow(obs_batch[0].cpu().view(newshape))
            plt.figure(1)
            plt.imshow(D(E(obs_batch))[0].cpu().view(newshape).detach())
            plt.show()
        
    torch.save(E.state_dict(), '{}/vae-encoder'.format(args.env_name))
    torch.save(D.state_dict(), '{}/vae-decoder'.format(args.env_name))
