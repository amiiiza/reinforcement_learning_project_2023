from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer
import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy, time
from pathlib import Path

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

class DDPGAgent(BaseAgent):
    def __init__(self, config=None):
        super(DDPGAgent, self).__init__(config)
        self.device = self.cfg.device  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.name = 'ddpg'
        state_dim = self.observation_space_dim
        self.action_dim = self.action_space_dim
        self.max_action = self.cfg.max_action
        self.lr=self.cfg.lr
        self.pi = Policy(state_dim, self.action_dim, self.max_action).to(self.device)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=float(self.lr))

        self.q = Critic(state_dim, self.action_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=float(self.lr))

      
        self.buffer = ReplayBuffer((state_dim,), self.action_dim, max_size=int(float(1e6)))
        
        self.batch_size = self.cfg.batch_size
        self.gamma = self.cfg.gamma
        self.tau = self.cfg.tau
        
        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0 
        self.random_transition = 5000 # collect 5k random data for better exploration
        self.max_episode_steps=self.cfg.max_episode_steps
    

    def update(self,):
        """ After collecting one trajectory, update the pi and q for #transition times: """
        info = {}
        
        update_iter = self.buffer_ptr - self.buffer_head # update the network once per transition

        if self.buffer_ptr > self.random_transition: # update once we have enough data
            for _ in range(update_iter):
                info = self._update()
        
        # update the buffer_head:
        self.buffer_head = self.buffer_ptr
       
        return info

    ### I add this, it might be better to delete it and use ###
    
    def _update(self,):
        # get batch data
        batch = self.buffer.sample(self.batch_size, device=self.device)
        
        a_next = self.pi_target(batch.next_state)
        q_next = self.q_target(batch.next_state, a_next)
        
        # compute target q
        q_tar = batch.reward + self.gamma * q_next * batch.not_done
        
        # compute critic loss
        q_policy = self.q(batch.state, batch.action)
        critic_loss = torch.mean(torch.square(q_tar.detach() - q_policy))

        # optimize the critic
        critic_loss.backward()
        self.q_optim.step()

        # compute actor loss
        a_policy_new = self.pi(batch.state)
        q_policy_new = self.q(batch.state, a_policy_new)
        actor_loss = - torch.mean(q_policy_new)

        # optimize the actor
        actor_loss.backward()
        self.pi_optim.step()

        # update the target q and target pi using u.soft_update_params() function
        cu.soft_update_params(self.q, self.q_target, self.tau)
        cu.soft_update_params(self.pi, self.pi_target, self.tau)
        self.q_optim.zero_grad()
        self.pi_optim.zero_grad()

        return {}

    
    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        if observation.ndim == 1: 
            observation = observation[None] 
        x = torch.from_numpy(observation).float().to(self.device)
          
        action = self.pi(x)
            
        expl_noise = torch.normal(0, 0.3, size=action.shape).to(self.device)
            
        if not evaluation:
            action = action + expl_noise
            
        action = action.squeeze(0)
       
        return action, {} # just return a positional value


        
    def train_iteration(self):
        #start = time.perf_counter()
        # Run actual training        
        reward_sum, timesteps, done = 0, 0, False
        # Reset the environment and observe the initial state
        obs, _ = self.env.reset()
        while not done:
            
            # Sample action from policy
            if self.buffer_ptr < self.random_transition:
                action = torch.rand(self.action_dim)
            else:
                action, ـ = self.get_action(obs)

            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _, _ = self.env.step(to_numpy(action))

            # Store action's outcome (so that the agent can improve its policy)        
            
            done_bool = float(done) if timesteps < self.max_episode_steps else 0 
            self.record(obs, action, next_obs, reward, done_bool)
                
            # Store total episode reward
            reward_sum += reward
            timesteps += 1
            
            if timesteps >= self.max_episode_steps:
                done = True
            # update observation
            obs = next_obs.copy()

        # update the policy after one episode
        #s = time.perf_counter()
        info = self.update()
        #e = time.perf_counter()
        
        # Return stats of training
        info.update({
                    'episode_length': timesteps,
                    'ep_reward': reward_sum,
                    })
        
        end = time.perf_counter()
        return info
        
    def train(self):
        if self.cfg.save_logging:
            L = cu.Logger() # create a simple logger to record stats
        start = time.perf_counter()
        total_step=0
        run_episode_reward=[]
        log_count=0
        
        for ep in range(self.cfg.train_episodes + 1):
            # collect data and update the policy
            train_info = self.train_iteration()
            train_info.update({'episodes': ep})
            total_step+=train_info['episode_length']
            train_info.update({'total_step': total_step})
            run_episode_reward.append(train_info['ep_reward'])
            
            if total_step>self.cfg.log_interval*log_count:
                average_return=sum(run_episode_reward)/len(run_episode_reward)
                if not self.cfg.silent:
                    print(f"Episode {ep} Step {total_step} finished. Average episode return: {average_return}")
                if self.cfg.save_logging:
                    train_info.update({'average_return':average_return})
                    L.log(**train_info)
                run_episode_reward=[]
                log_count+=1

        if self.cfg.save_model:
            self.save_model()
            
        logging_path = str(self.logging_dir)+'/logs'   
        if self.cfg.save_logging:
            L.save(logging_path, self.seed)
        self.env.close()

        end = time.perf_counter()
        train_time = (end-start)/60
        print('------ Training Finished ------')
        print(f'Total traning time is {train_time}mins')
        
    def load_model(self):
        # define the save path, do not modify
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        
        d = torch.load(filepath)
        self.q.load_state_dict(d['q'])
        self.q_target.load_state_dict(d['q_target'])
        self.pi.load_state_dict(d['pi'])
        self.pi_target.load_state_dict(d['pi_target'])
    
    def save_model(self):   
        # define the save path, do not modify
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        
        torch.save({
            'q': self.q.state_dict(),
            'q_target': self.q_target.state_dict(),
            'pi': self.pi.state_dict(),
            'pi_target': self.pi_target.state_dict()
        }, filepath)
        print("Saved model to", filepath, "...")
        
    def record(self, state, action, next_state, reward, done):
        """ Save transitions to the buffer. """
        self.buffer_ptr += 1
        self.buffer.add(state, action, next_state, reward, done)
        
        
