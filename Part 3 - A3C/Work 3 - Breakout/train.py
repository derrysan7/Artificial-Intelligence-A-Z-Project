# Training the AI

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
        
def train(rank, params, shared_model, optimizer): #rank = to shift the seed so that each agent is desynchronized. its deterministic with respect to the seed
    torch.manual_seed(params.seed + rank)
    env = create_atari_env(params.env_name)
    env.seed(params.seed + rank) #to align each agent to different environment, we're using seed
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    state = env.reset() #this will initialize the state as numpy array of dimension 1 (b&w) by 42x42
    state = torch.from_numpy(state)
    done = True
    episode_length = 0
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict()) #will get the shared model to do its small exploration
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        values = [] #the output of critic
        log_probs = []
        rewards = []
        entropies = []
        for step in range(params.num_steps): #for loop over the exploration step
            value, action_values, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx))) #this will get us several output: v values(critic), q values(actor), tuple of hidden state and cell state
            prob = F.softmax(action_values)
            log_prob = F.log_softmax(action_values)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)
            action = prob.multinomial(num_samples = 1).data #take a random draw of action according to the distribution of probability of the softmax function
            log_prob = log_prob.gather(1, Variable(action)) #update the log_prob associated with the action
            values.append(value)
            log_probs.append(log_prob)
            state, reward, done, _ = env.step(action.numpy()) #play the action in the env
            done = (done or episode_length >= params.max_episode_length) #max 10000 unit
            reward = max(min(reward, 1), -1) #will make sure the reward between -1 and +1
            if done:
                episode_length = 0
                state = env.reset()
            state = torch.from_numpy(state)
            rewards.append(reward)
            if done:
                break #if its done, we stop the exploration and move on to the next operation which will be update the shared model
        R = torch.zeros(1, 1) #cumulative reward
        if not done:
            value, _, _= model((Variable(state.unsqueeze(0)), (hx, cx))) #what we want right now is the cumulative reward to be equal to the value of the last state reached by the shared network
            R = value.data
        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R) #set cumul reward as torch variable
        gae = torch.zeros(1, 1) #A(a,s) = Q(a,s) - V(s); GAE = Generalized Advantage Estimation; The advantage of playing action A, by absorbing the state S, so its the function of the action A and the state S
        for i in reversed(range(len(rewards))): #i = step
            R = params.gamma * R + rewards[i] # R = r_0 + gamma * r_1 + gamma*2 + r_2 + .... + gamma*(n-1) * r(n-1) + gamma*nb_steps + V(last state)
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2) # Q*(a*,s) = V*(s) #0.5 = discount factor
            TD = rewards[i] + params.gamma * values[i+1].data - values[i].data
            gae = gae * params.gamma * params.tau + TD # gae = sum_i (gamma*tau)^i * TD(i)
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i] # policy_loss = - sum_i log(pi_i)*gae + 0.01*H_i #pi_i = softmax probability of the action
        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward() #we give twice as much important to the policy loss
        torch.nn.utils.clip_grad_norm(model.parameters(), 40) #to make sure the gradient won't take extremely large values and degenerate the algorithm #the norm of the gradient stays between 0 and 40
        ensure_shared_grads(model, shared_model)
        optimizer.step()
