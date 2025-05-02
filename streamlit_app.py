import streamlit as st
import torch
import torch.nn.functional as F # Fixed typo: Ffunctional -> F
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
# Fixed: Use Normal for univariate action space
from torch.distributions import Normal # MultivariateNormal
import copy
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from io import BytesIO
import time

# Set page configuration
st.set_page_config(
    page_title="CSTR Control with AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and introduction
st.title("Chemical Reactor Control with AI Methods")
st.markdown("""
This application allows you to experiment with AI-based control methods for a Continuous Stirred Tank Reactor (CSTR).
You can adjust various parameters on the left sidebar and see how different AI approaches perform in controlling the chemical process.

**Two AI methods are implemented:**
- **Stochastic Policy Search (SPS)**: A random search algorithm that explores the parameter space to find a good initial policy.
- **Policy Gradient (PG)**: A reinforcement learning approach that refines a policy (potentially initialized by SPS) using gradient descent based on simulated experience.

Adjust the parameters, run the simulation/training, and visualize the results below.
""")

# Define essential functions for the CSTR model
eps = np.finfo(float).eps

# @st.cache_data # Caching ODE integration might be complex due to changing inputs (u)
def cstr(x, t, u, Tf, q, Caf, V, rho, Cp, mdelH, EoverR, k0, UA):
    """ CSTR model differential equations """
    # == Inputs == #
    Tc = u  # Temperature of cooling jacket (K)

    # == States == #
    Ca = x[0]  # Concentration of A in CSTR (mol/m^3)
    T = x[1]  # Temperature in CSTR (K)

    # == Equations == #
    rA = k0 * np.exp(-EoverR / T) * Ca  # reaction rate
    dCadt = q / V * (Caf - Ca) - rA  # Calculate concentration derivative
    dTdt = q / V * (Tf - T) \
           + mdelH / (rho * Cp) * rA \
           + UA / V / rho / Cp * (Tc - T)  # Calculate temperature derivative

    # == Return xdot == #
    xdot = np.zeros(2)
    xdot[0] = dCadt
    xdot[1] = dTdt
    return xdot

# @st.cache_data # PID depends on history, caching might be incorrect
def PID(Ks, x, x_setpoint, e_history, Tc_lb, Tc_ub):
    """ Basic PID controller calculation """
    Ks = np.array(Ks)
    Ks = Ks.reshape(7, order='C')

    # K gains
    KpCa = Ks[0]; KiCa = Ks[1]; KdCa = Ks[2]
    KpT = Ks[3]; KiT = Ks[4]; KdT = Ks[5]
    Kb = Ks[6]
    # setpoint error
    e = x_setpoint - x
    # control action
    # Ensure e_history is 2D for sum and indexing
    e_hist_array = np.array(e_history)
    if e_hist_array.ndim == 1:
         e_hist_array = e_hist_array.reshape(1, -1) # Reshape if only one past error exists

    u = KpCa * e[0] + KiCa * np.sum(e_hist_array[:, 0]) + KdCa * (e[0] - e_hist_array[-1, 0])
    u += KpT * e[1] + KiT * np.sum(e_hist_array[:, 1]) + KdT * (e[1] - e_hist_array[-1, 1])
    u += Kb
    u = min(max(u, Tc_lb), Tc_ub) # Apply bounds

    return u

# Neural Network definitions
class Net(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        self.dtype = torch.float
        self.args = kwargs
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cpu") # Keep on CPU for simplicity
        self.input_size = self.args['input_size']
        self.output_sz = self.args['output_size']
        self.hs1 = self.input_size * 2
        self.hs2 = self.output_sz * 2
        self.hidden1 = torch.nn.Linear(self.input_size, self.hs1)
        self.hidden2 = torch.nn.Linear(self.hs1, self.hs2)
        self.output = torch.nn.Linear(self.hs2, self.output_sz)

    def forward(self, x):
        x = x.view(1, -1).float() # Adjust view for linear layer
        y = F.leaky_relu(self.hidden1(x), 0.1)
        y = F.leaky_relu(self.hidden2(y), 0.1)
        y = F.relu6(self.output(y))  # range (0,6)
        return y

class Net_TL(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Net_TL, self).__init__()
        self.dtype = torch.float
        self.args = kwargs
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cpu") # Keep on CPU
        self.input_size = self.args['input_size']
        self.output_sz = self.args['output_size'] # Initial output size (e.g., 1 for SPS)
        self.hs1 = self.input_size * 2
        self.hs2 = self.output_sz * 2 # Base hs2 on initial output size
        self.hidden1 = torch.nn.Linear(self.input_size, self.hs1)
        self.hidden2 = torch.nn.Linear(self.hs1, self.hs2)
        self.output = torch.nn.Linear(self.hs2, self.output_sz) # Initial output layer

    def forward(self, x):
        x = x.view(1, -1).float() # Adjust view for linear layer
        y = F.leaky_relu(self.hidden1(x), 0.1)
        y = F.leaky_relu(self.hidden2(y), 0.1)
        y = F.relu6(self.output(y)) # range (0,6) - applied to both mean and stddev precursor
        return y

    def increaseClassifier(self, m: torch.nn.Linear, hs2_size):
        """ Increases the output layer size by 1 """
        old_out_features = m.out_features
        new_out_features = old_out_features + 1

        # Create new layer
        m2 = nn.Linear(m.in_features, new_out_features)

        # Initialize new weights and biases (e.g., copy old, add small random for new)
        new_weight = torch.cat((m.weight.data, torch.randn(1, m.in_features) * 0.01), dim=0)
        new_bias = torch.cat((m.bias.data, torch.randn(1) * 0.01), dim=0)

        m2.weight = nn.parameter.Parameter(new_weight, requires_grad=True)
        m2.bias = nn.parameter.Parameter(new_bias, requires_grad=True)

        # Update network's output size tracker
        self.output_sz = new_out_features
        # Adjust hs2 size if needed, though typically based on initial output_sz
        # self.hs2 = new_out_features * 2 # Optional: Resize preceding layer too
        # self.hidden2 = nn.Linear(self.hs1, self.hs2) # Recreate if resizing

        return m2

    def incrHere(self):
        """ Increases the output layer for PG (mean + std dev) """
        if self.output.out_features == 1: # Only increase if it's currently 1
             self.output = self.increaseClassifier(self.output, self.hs2)
        elif self.output.out_features > 2: # If already increased wrongly, reset
             st.warning("Resetting output layer size during incrHere.")
             self.output_sz = 1 # Reset tracker
             self.hs2 = self.output_sz * 2
             self.hidden2 = torch.nn.Linear(self.hs1, self.hs2)
             self.output = torch.nn.Linear(self.hs2, self.output_sz) # Recreate initial
             self.output = self.increaseClassifier(self.output, self.hs2) # Now increase correctly


# Policy functions
def sample_uniform_params(params_prev, param_max, param_min):
    """ Sample parameters uniformly within bounds """
    params = {k: torch.rand(v.shape) * (param_max - param_min) + param_min
              for k, v in params_prev.items()}
    return params

def sample_local_params(params_prev, radius_max, radius_min):
    """ Sample parameters locally around previous best """
    params = {k: torch.rand(v.shape) * (radius_max - radius_min) + radius_min + v
              for k, v in params_prev.items()}
    # Ensure parameters stay within reasonable global bounds if needed
    # param_max_global = 5
    # param_min_global = -5
    # params = {k: torch.clamp(p, param_min_global, param_max_global) for k, p in params.items()}
    return params

def select_action(control_mean, control_sigma):
    """
    Sample control action from the Normal distribution.
    input: Mean (scalar Tensor), Standard Deviation (scalar Tensor)
    Output: Control action (scalar Tensor), log_probability (scalar Tensor), entropy (scalar Tensor)
    """
    # Ensure sigma is positive
    control_sigma = torch.clamp(control_sigma, min=eps) # Use softplus? F.softplus(control_sigma)
    dist = Normal(control_mean, control_sigma)
    control_choice = dist.sample()  # sample control from N(mu, std)
    log_prob = dist.log_prob(control_choice)  # compute log prob of this action
    entropy = dist.entropy()  # compute the entropy

    # Detach tensors that are only used for simulation steps, not gradient calculation later
    return control_choice.detach(), log_prob, entropy.detach()


def mean_std(m, s, mean_range, mean_lb, std_range, std_lb):
    '''
    Problem specific restrictions on predicted mean and standard deviation.
    m, s are outputs from Relu6 (range 0-6)
    '''
    # Ensure inputs are tensors
    m_tensor = torch.as_tensor(m, dtype=torch.float)
    s_tensor = torch.as_tensor(s, dtype=torch.float)

    # Calculate mean: Scale 0-6 output to mean_range and add lower bound
    mean = (m_tensor / 6.0) * torch.as_tensor(mean_range, dtype=torch.float) + torch.as_tensor(mean_lb, dtype=torch.float)

    # Calculate std dev: Scale 0-6 output to std_range and add lower bound
    # Add epsilon to prevent std dev from being exactly zero
    std = (s_tensor / 6.0) * torch.as_tensor(std_range, dtype=torch.float) + torch.as_tensor(std_lb, dtype=torch.float) + eps

    return mean, std


# Main functions for evaluation and plotting
def plot_simulation(Ca_dat, T_dat, Tc_dat, t, Ca_des, T_des, title="Final Policy Simulation"):
    """ Plots a single simulation run """
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=14)

    # Concentration Plot
    axs[0].plot(t, Ca_dat, 'r-', lw=2, label='Actual Ca')
    axs[0].step(t, Ca_des, '--', lw=1.5, color='black', label='Setpoint Ca')
    axs[0].set_ylabel('Ca (mol/m^3)')
    axs[0].legend(loc='best')
    axs[0].grid(True)

    # Temperature Plot
    axs[1].plot(t, T_dat, 'c-', lw=2, label='Actual T')
    axs[1].step(t, T_des, '--', lw=1.5, color='black', label='Setpoint T')
    axs[1].set_ylabel('T (K)')
    axs[1].legend(loc='best')
    axs[1].grid(True)

    # Control Action Plot
    # Ensure Tc_dat matches time steps correctly (usually one less action than states)
    axs[2].step(t[:-1], Tc_dat, 'b--', lw=2, label='Cooling Tc') # Plot Tc[:-1] against t[:-1]
    axs[2].set_ylabel('Cooling T (K)')
    axs[2].set_xlabel('Time (min)')
    axs[2].legend(loc='best')
    axs[2].set_xlim(min(t), max(t))
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
    return fig

def plot_training(t, Ca_train_list, T_train_list, Tc_train_list, Ca_des, T_des, title="Training Trajectories"):
    """ Plots multiple training trajectories """
    repetitions = len(Ca_train_list)
    if repetitions == 0:
        st.warning("No training data to plot.")
        return None

    c_ = [(repetitions - float(i)) / repetitions for i in range(repetitions)] # Transparency gradient

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=14)

    # Concentration Plot
    for run_i in range(repetitions):
        axs[0].plot(t, Ca_train_list[run_i], 'r-', lw=1, alpha=max(0.1, c_[run_i]*0.7)) # Ensure alpha > 0
    axs[0].step(t, Ca_des, '--', lw=1.5, color='black', label='Setpoint Ca')
    axs[0].set_ylabel('Ca (mol/m^3)')
    # axs[0].set_ylim([min(Ca_des)*0.95, max(Ca_des)*1.05]) # Dynamic Ylim
    axs[0].legend(loc='best')
    axs[0].grid(True)

    # Temperature Plot
    for run_i in range(repetitions):
        axs[1].plot(t, T_train_list[run_i], 'c-', lw=1, alpha=max(0.1, c_[run_i]*0.7))
    axs[1].step(t, T_des, '--', lw=1.5, color='black', label='Setpoint T')
    axs[1].set_ylabel('T (K)')
    # axs[1].set_ylim([min(T_des)*0.98, max(T_des)*1.02]) # Dynamic Ylim
    axs[1].legend(loc='best')
    axs[1].grid(True)

    # Control Action Plot
    for run_i in range(repetitions):
         # Ensure Tc length matches time steps correctly
         axs[2].step(t[:-1], Tc_train_list[run_i], 'b--', lw=1, alpha=max(0.1, c_[run_i]*0.7)) # Plot Tc[:-1] against t[:-1]
    axs[2].set_ylabel('Cooling T (K)')
    axs[2].set_xlabel('Time (min)')
    axs[2].legend(['Jacket Temperature'], loc='best')
    axs[2].set_xlim(min(t), max(t))
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout
    return fig


def plot_pg_rewards(rewards_m_record, rewards_std_record):
    """ Plots the mean and standard deviation of rewards during PG training """
    if not rewards_m_record:
        st.warning("No PG reward data to plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = np.arange(len(rewards_m_record))
    rewards_m = np.array(rewards_m_record)
    rewards_std = np.array(rewards_std_record)

    ax.plot(iterations, rewards_m, 'black', linewidth=1.5, label='Mean Reward')
    ax.fill_between(iterations, rewards_m - 2 * rewards_std, rewards_m + 2 * rewards_std,
                      color='C0', alpha=0.2, label='±2 Std Dev')
    ax.set_title('Policy Gradient Training Rewards')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.legend(loc='lower right')
    ax.grid(True)
    plt.tight_layout()
    return fig

# Define the objective function / simulation environment
def J_PolicyCSTR(policy, data_res, policy_alg='PID',
                 collect_training_data=False, # Default to False unless during training runs
                 traj=False, episode=False):
    """
    Simulates the CSTR with a given policy and calculates cost/reward.
    policy: The control policy (PID gains, SPS network, or PG network)
    data_res: Dictionary containing simulation parameters and storage for results
    policy_alg: 'PID', 'SPS_RL', or 'PG_RL'
    collect_training_data: Boolean, if True, appends results to lists in data_res
    traj: Boolean, if True, returns full trajectories (Ca, T, Tc)
    episode: Boolean, if True, returns (reward, sum_logprob) for PG training
    """
    # Unpack simulation parameters from data_res
    t = data_res['t']
    x0 = data_res['x0']
    noise_level = data_res['noise_level'] # Renamed from 'noise'
    Ca_des = data_res['Ca_des']
    T_des = data_res['T_des']
    Tc_ub = data_res['Tc_ub']
    Tc_lb = data_res['Tc_lb']
    x_norm_mean, x_norm_std = data_res['x_norm'] # Unpack mean and std
    u_norm_range, u_norm_lb = data_res['u_norm'] # Unpack range and lb
    # CSTR parameters
    cstr_params = data_res['cstr_params']

    # Initialize state and data storage for this run
    Ca = np.zeros_like(t)
    T = np.zeros_like(t)
    Tc = np.zeros_like(t) # Will store calculated Tc[0] to Tc[n-2]
    Ca[0], T[0] = x0[0], x0[1]
    x = copy.deepcopy(x0)
    e_history = [] # For PID derivative term

    # Initialize lists for PG
    log_probs = []
    rewards_step = []

    # Simulate CSTR step-by-step
    for i in range(len(t) - 1):
        # Current state and setpoint
        current_state = x
        x_sp = np.array([Ca_des[i], T_des[i]])
        error = x_sp - current_state

        # Append current error to history (needed before PID calculation)
        e_history.append(error)

        # Calculate control action based on policy
        ts = [t[i], t[i + 1]] # Time interval for ODE solver

        #### PID Controller ####
        if policy_alg == 'PID':
            if i == 0: # Handle first step for derivative term
                 Tc_i = PID(policy, current_state, x_sp, np.array([[0, 0]]), Tc_lb, Tc_ub)
            else:
                 Tc_i = PID(policy, current_state, x_sp, e_history, Tc_lb, Tc_ub)

        #### Stochastic Policy Search (SPS) Network ####
        elif policy_alg == 'SPS_RL':
            # Prepare network input: current state + current error
            xk = np.hstack((current_state, error))
            # Normalize input
            xknorm = (xk - x_norm_mean) / x_norm_std
            xknorm_torch = torch.tensor(xknorm, dtype=torch.float)
            # Get action from policy (deterministic mean for SPS)
            # Output is scaled 0-6
            u_k_scaled = policy(xknorm_torch).item() # Get scalar output
            # Denormalize/scale action
            u_k = (u_k_scaled / 6.0) * u_norm_range + u_norm_lb
            # Apply bounds
            Tc_i = min(max(u_k, Tc_lb), Tc_ub)

        #### Policy Gradient (PG) Network ####
        elif policy_alg == 'PG_RL':
            # Prepare network input
            xk = np.hstack((current_state, error))
            # Normalize input
            xknorm = (xk - x_norm_mean) / x_norm_std
            xknorm_torch = Tensor(xknorm)
            # Get mean and std dev precursor from policy network
            # Network output shape is (1, 2) after incrHere()
            nn_output = policy(xknorm_torch)[0] # Shape (2,)
            m_scaled, s_scaled = nn_output[0], nn_output[1]

            # Problem-specific scaling for mean and std dev
            mean_uk, std_uk = mean_std(m_scaled, s_scaled,
                                       mean_range=u_norm_range, mean_lb=u_norm_lb,
                                       std_range=data_res['pg_std_range'], std_lb=data_res['pg_std_lb']) # Use PG specific std range

            # Sample action from Normal distribution
            u_k_tensor, logprob_k, _ = select_action(mean_uk, std_uk) # logprob_k is needed for PG update
            u_k = u_k_tensor.item() # Get scalar value

            # Apply bounds (Clipping introduces bias, consider alternatives if problematic)
            Tc_i = min(max(u_k, Tc_lb), Tc_ub)

            # Store log probability for this step
            log_probs.append(logprob_k)

        else:
            raise ValueError("Unknown policy_alg specified.")

        # Store calculated control action
        Tc[i] = Tc_i

        # Simulate CSTR dynamics for one step
        y = odeint(cstr, current_state, ts, args=(Tc_i, *cstr_params)) # Pass Tc_i and other params

        # Get next state from ODE solver
        next_state = y[-1]

        # Add process noise/disturbance (optional)
        disturbance = np.random.uniform(low=-1, high=1, size=2) * np.array([0.1, 5]) # Example scaling
        Ca[i + 1] = next_state[0] + noise_level * disturbance[0]
        T[i + 1] = next_state[1] + noise_level * disturbance[1]

        # Ensure states are physically plausible (e.g., concentration >= 0)
        Ca[i + 1] = max(Ca[i + 1], 0)
        # T[i + 1] = max(T[i + 1], 0) # Temperature likely doesn't need floor

        # Update state for next iteration
        x[0] = Ca[i + 1]
        x[1] = T[i + 1]

        # Calculate step reward (negative cost) for PG
        if policy_alg == 'PG_RL':
            step_error_cost = (np.abs(error[0]) / 0.2) + (np.abs(error[1]) / 15.0) # Scaled error cost
            step_u_mag_cost = np.abs(Tc_i - Tc_lb) / (Tc_ub - Tc_lb) * 0.1 # Scaled magnitude cost (0.1 weight)
            step_u_cha_cost = 0
            if i > 0:
                step_u_cha_cost = np.abs(Tc_i - Tc[i-1]) / (Tc_ub - Tc_lb) * 0.1 # Scaled change cost (0.1 weight)
            step_reward = -(step_error_cost + step_u_mag_cost + step_u_cha_cost)
            rewards_step.append(step_reward)


    # == Calculate objective == #
    # Use calculated errors (e_history) and actions (Tc[:-1])
    e_hist_array = np.array(e_history) # Shape (n_steps-1, 2)
    # Tracking error cost (sum over time)
    error_cost = np.sum(np.abs(e_hist_array[:, 0]) / 0.2 + np.abs(e_hist_array[:, 1]) / 15)

    # Control action magnitude cost
    # Use Tc[0] to Tc[n-2], which has length n-1
    u_mag_cost = np.sum(np.abs(Tc[:-1] - Tc_lb) / (Tc_ub - Tc_lb)) * 0.1 # Scaled and weighted

    # Control action change cost
    # Compare Tc[1]..Tc[n-2] with Tc[0]..Tc[n-3] => length n-2
    u_cha_cost = np.sum(np.abs(Tc[1:-1] - Tc[:-2]) / (Tc_ub - Tc_lb)) * 0.1 # Scaled and weighted


    # Total cost (for SPS)
    total_cost = error_cost + u_mag_cost + u_cha_cost

    # Collect data for plotting training progress if requested
    if collect_training_data:
        data_res['Ca_train'].append(copy.deepcopy(Ca))
        data_res['T_train'].append(copy.deepcopy(T))
        # Store the n-1 calculated actions
        data_res['Tc_train'].append(copy.deepcopy(Tc[:-1]))
        # Storing costs can be useful too
        # data_res['err_train'].append(error_cost)
        # data_res['u_mag_train'].append(u_mag_cost)
        # data_res['u_cha_train'].append(u_cha_cost)

    # Return based on flags
    if episode: # For PG training episode
        # Return sum of step rewards and the list of log probabilities
        # Note: PG usually works with sum of rewards, not negative cost
        episode_reward = np.sum(rewards_step)
        return episode_reward, log_probs # Return list of log_probs tensors

    if traj: # For final simulation plot
        # Return full state and action trajectories
        return Ca, T, Tc[:-1] # Return n-1 actions
    else: # For SPS objective function
        # Return total cost (to be minimized)
        return total_cost


# Function to run SPS (Stochastic Policy Search)
def run_sps(data_res, e_tot, e_shr, shrink_ratio, radius, ratio_ls_rs, param_max, param_min):
    """ Runs the Stochastic Policy Search algorithm """
    # Clear previous training data stored in data_res
    data_res['Ca_train'] = []
    data_res['T_train'] = []
    data_res['Tc_train'] = []
    # data_res['err_train'] = [] # Optional cost tracking
    # data_res['u_mag_train'] = []
    # data_res['u_cha_train'] = []

    # Problem dimensions
    nu = 1 # Single output (Tc)
    nx = 2 # Two states (Ca, T)
    # Input size: nx states + nx errors = 4
    hyparams = {'input_size': nx + nx, 'output_size': nu}
    n_steps = data_res['n'] # Simulation steps

    # Policy initialization
    policy_net = Net(**hyparams)
    params_init = policy_net.state_dict()

    # Initialize rewards/costs (SPS minimizes cost)
    best_cost = float('inf')
    best_policy = copy.deepcopy(params_init)

    # Adapt evaluations
    evals_rs = max(1, round(e_tot * ratio_ls_rs)) # Ensure at least 1 random search eval
    evals_ls = e_tot - evals_rs

    # Progress bar setup
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    # === Random Search Phase ===
    status_text.text(f"Starting Random Search Phase ({evals_rs} evaluations)...")
    for policy_i in range(evals_rs):
        # Sample new parameters uniformly
        NNparams_RS = sample_uniform_params(params_init, param_max, param_min)
        # Load parameters into the network
        policy_net.load_state_dict(NNparams_RS)
        # Evaluate the policy (get cost)
        cost = J_PolicyCSTR(policy_net, data_res, collect_training_data=True,
                            policy_alg='SPS_RL')

        # Update best policy if current one is better (lower cost)
        if cost < best_cost:
            best_cost = cost
            best_policy = copy.deepcopy(NNparams_RS)
            status_text.text(f"Random Search [{policy_i + 1}/{evals_rs}]: New best cost = {best_cost:.4f}")

        # Update progress bar
        progress = (policy_i + 1) / e_tot
        progress_bar.progress(progress)

    st.write(f"Random Search Phase Complete. Best cost found: {best_cost:.4f}")

    # === Local Search Phase ===
    status_text.text(f"Starting Local Search Phase ({evals_ls} evaluations)...")
    current_radius = radius # Initial local search radius
    # Calculate initial bounds for local sampling based on radius
    r_max = current_radius * param_max
    r_min = current_radius * param_min
    iter_i = 0 # Local search iteration count
    fail_i = 0 # Counter for consecutive failures to improve

    while iter_i < evals_ls:
        # Shrink radius if stuck
        if fail_i >= e_shr:
            fail_i = 0
            current_radius = current_radius * shrink_ratio
            r_max = current_radius * param_max
            r_min = current_radius * param_min
            st.write(f"Shrinking local search radius to {current_radius:.4f}")

        # Sample new parameters locally around the current best
        NNparams_LS = sample_local_params(best_policy, r_max, r_min)

        # Load and evaluate the new policy
        policy_net.load_state_dict(NNparams_LS)
        cost = J_PolicyCSTR(policy_net, data_res, collect_training_data=True,
                            policy_alg='SPS_RL')

        # Update best policy if improved
        if cost < best_cost:
            best_cost = cost
            best_policy = copy.deepcopy(NNparams_LS)
            fail_i = 0 # Reset failure counter
            status_text.text(f"Local Search [{iter_i + 1}/{evals_ls}]: New best cost = {best_cost:.4f}")
        else:
            fail_i += 1 # Increment failure counter

        # Increment iteration counter
        iter_i += 1

        # Update progress bar
        progress = (evals_rs + iter_i) / e_tot
        progress_bar.progress(progress)

    end_time = time.time()
    st.success(f"SPS training completed in {end_time - start_time:.2f} seconds! Final best cost: {best_cost:.4f}")

    # Create the final best policy network
    policy_net_SPS_RL = Net(**hyparams)
    policy_net_SPS_RL.load_state_dict(best_policy)

    # Return the best policy state dict, the best cost, and the network object
    return best_policy, best_cost, policy_net_SPS_RL


# Function to run Policy Gradient (REINFORCE)
def run_pg(data_res, initial_policy_state_dict, lr, total_it, n_episodes):
    """ Runs the Policy Gradient (REINFORCE) algorithm """
    # Clear previous training data
    data_res['Ca_train'] = []
    data_res['T_train'] = []
    data_res['Tc_train'] = []

    # Problem dimensions and network setup
    nx = 2; nu = 1
    # Input size: nx states + nx errors = 4
    # Initial output size is 1 (from SPS)
    hyparams = {'input_size': nx + nx, 'output_size': 1}

    # Initialize policy network using Net_TL for transfer learning
    policy_net_pg = Net_TL(**hyparams)

    # Load state dict from SPS (or provided initial)
    policy_net_pg.load_state_dict(initial_policy_state_dict)

    # IMPORTANT: Increase output layer for PG (mean + std dev)
    st.write("Adapting network for Policy Gradient (adding std dev output)...")
    policy_net_pg.incrHere()
    st.write(f"Network output size after incrHere: {policy_net_pg.output.out_features}")
    if policy_net_pg.output.out_features != 2:
        st.error("Failed to adjust network output layer size correctly for PG!")
        return None, None, None # Return None if setup failed


    # Define optimizer and scheduler
    optimizer_pol = optim.Adam(policy_net_pg.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_pol, mode='max', factor=0.5, patience=max(5, int(total_it / (n_episodes*10))), # Reduce LR if reward plateaus
        verbose=True, min_lr=1e-6, cooldown=5)

    # Calculate number of epochs
    n_epochs = max(1, int(total_it / n_episodes))
    st.write(f"Running PG for {n_epochs} epochs with {n_episodes} episodes per epoch.")

    # --- REINFORCE Algorithm ---
    def reinforce_epoch(policy_network, optimizer, scheduler, epoch_num, n_episodes_per_epoch):
        """ Performs one epoch of the REINFORCE algorithm """
        policy_network.train() # Set network to training mode
        optimizer.zero_grad() # Zero gradients at the start of the epoch

        batch_log_probs = [] # Store log_probs from all episodes in the batch
        batch_rewards = [] # Store total rewards from all episodes

        # --- Run Episodes ---
        for epi_i in range(n_episodes_per_epoch):
            # Run one episode using the current policy
            episode_reward, episode_log_probs_list = J_PolicyCSTR(
                policy_network, data_res, policy_alg='PG_RL',
                collect_training_data=False, # Don't collect detailed trajectory data during PG training
                episode=True # Get reward and log_probs
            )

            # Store results for this episode
            batch_rewards.append(episode_reward)
            # We need to associate each log_prob with the final episode reward
            # Simplest REINFORCE: sum log probs and associate with total reward
            # More advanced: Use reward-to-go or baseline
            sum_log_probs = torch.stack(episode_log_probs_list).sum() # Sum log probs for the episode
            batch_log_probs.append(sum_log_probs)


        # --- Calculate Loss and Update Policy ---
        rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float)

        # Baseline: Subtract mean reward to reduce variance
        baseline = rewards_tensor.mean()
        rewards_normalized = (rewards_tensor - baseline) # / (rewards_tensor.std() + eps) # Optional: normalize by std dev

        # Calculate policy gradient loss
        # Loss = - E[ (R - b) * log pi(a|s) ]
        # We use gradient ascent on reward, so minimize negative of this
        loss = 0
        for log_prob, R_norm in zip(batch_log_probs, rewards_normalized):
             loss += -log_prob * R_norm # Accumulate loss weighted by normalized reward

        loss /= n_episodes_per_epoch # Average the loss over the batch

        # Backpropagate and update
        loss.backward()
        # Gradient clipping (optional but often helpful)
        # torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=1.0)
        optimizer.step()

        # --- Record and Return Epoch Stats ---
        epoch_mean_reward = np.mean(batch_rewards)
        epoch_std_reward = np.std(batch_rewards)

        # Scheduler step based on mean reward
        scheduler.step(epoch_mean_reward)

        return epoch_mean_reward, epoch_std_reward, policy_network


    # --- Training Loop ---
    rewards_m_record = []
    rewards_std_record = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    with st.spinner('Running Policy Gradient Training...'):
        for epoch_i in range(n_epochs):
            # Run one epoch
            mean_r, std_r, policy_net_pg = reinforce_epoch(
                policy_net_pg, optimizer_pol, scheduler, epoch_i, n_episodes
            )

            # Record results
            rewards_m_record.append(mean_r)
            rewards_std_record.append(std_r)

            # Update progress
            progress = (epoch_i + 1) / n_epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch: {epoch_i + 1}/{n_epochs}, Mean Reward: {mean_r:.3f} ± {std_r:.3f}")

            # Optional: Collect data for training plot periodically
            if epoch_i % max(1, n_epochs // 10) == 0: # Plot every 10% of epochs
                 J_PolicyCSTR(policy_net_pg, data_res, policy_alg='PG_RL', collect_training_data=True)


    end_time = time.time()
    final_reward = rewards_m_record[-1] if rewards_m_record else float('nan')
    st.success(f"Policy Gradient training completed in {end_time - start_time:.2f} seconds! Final mean reward: {final_reward:.4f}")

    policy_net_pg.eval() # Set network to evaluation mode after training
    return rewards_m_record, rewards_std_record, policy_net_pg

# ==============================================================================
# Streamlit Sidebar - User Inputs (Defaults from Script)
# ==============================================================================
with st.sidebar:
    st.header("Configuration")

    ai_method = st.selectbox("Select AI Method", ["Stochastic Policy Search (SPS)", "Policy Gradient (PG)"])

    # --- Simulation Parameters ---
    st.subheader("Simulation Parameters")
    # Script values: tp=25, n=101, noise=0.1
    t_end_default = 25
    n_steps_default = 101
    noise_level_default = 0.1
    t_end = st.slider("Simulation Time (min)", 5, 50, value=t_end_default)
    n_steps = st.slider("Number of Steps", 50, 500, value=n_steps_default)
    noise_level = st.slider("Process Noise Level", 0.0, 1.0, value=noise_level_default, step=0.01)

    # --- Setpoints ---
    st.subheader("Setpoints (Step Changes)")
    # Script values: Ca=[0.8, 0.9], T=[330, 320] with change at n/2 (t=12.5)
    # App uses 3 steps, map script changes: SP1 -> SP1, SP2 -> SP2, SP1 -> SP3
    ca_sp1_default = 0.8
    ca_sp2_default = 0.9
    ca_sp3_default = 0.8 # Return to initial
    t_sp1_default = 330.0
    t_sp2_default = 320.0
    t_sp3_default = 330.0 # Return to initial
    # Default timing approx t_end/3 and 2*t_end/3 based on t_end_default=25
    t_change1_default = 8
    t_change2_default = 17

    ca_sp1 = st.number_input("Initial Ca Setpoint", 0.5, 1.5, value=ca_sp1_default, step=0.01)
    t_change1 = st.slider("Time of 1st SP Change", 0, t_end, value=t_change1_default)
    ca_sp2 = st.number_input("Second Ca Setpoint", 0.5, 1.5, value=ca_sp2_default, step=0.01)
    t_change2 = st.slider("Time of 2nd SP Change", t_change1, t_end, value=t_change2_default)
    ca_sp3 = st.number_input("Final Ca Setpoint", 0.5, 1.5, value=ca_sp3_default, step=0.01)

    t_sp1 = st.number_input("Initial T Setpoint (K)", 300.0, 400.0, value=t_sp1_default, step=0.5)
    t_sp2 = st.number_input("Second T Setpoint (K)", 300.0, 400.0, value=t_sp2_default, step=0.5)
    t_sp3 = st.number_input("Final T Setpoint (K)", 300.0, 400.0, value=t_sp3_default, step=0.5)

    # --- Control Parameters ---
    st.subheader("Control Parameters")
    # Script values: Tc_lb=295, Tc_ub=305
    Tc_lb_default = 295.0
    Tc_ub_default = 305.0
    Tc_lb = st.slider("Cooling Tc Lower Bound (K)", 250.0, 300.0, value=Tc_lb_default, step=0.5)
    Tc_ub = st.slider("Cooling Tc Upper Bound (K)", 300.0, 350.0, value=Tc_ub_default, step=0.5)

    # --- Normalization Parameters ---
    st.subheader("Normalization (Approx. Ranges)")
    # Script values: mean=[.8, 315, 0, 0], std=[.1, 10, .1, 20] (assuming std from script comment)
    x_norm_ca_mean_default = 0.8
    x_norm_t_mean_default = 315.0
    x_norm_ca_err_mean_default = 0.0
    x_norm_t_err_mean_default = 0.0
    x_norm_ca_std_default = 0.1
    x_norm_t_std_default = 10.0
    x_norm_ca_err_std_default = 0.1
    x_norm_t_err_std_default = 20.0

    # State: [Ca, T, Ca_err, T_err]
    x_norm_ca_mean = st.number_input("State Norm: Mean Ca", 0.5, 1.5, value=x_norm_ca_mean_default, step=0.01)
    x_norm_t_mean = st.number_input("State Norm: Mean T", 300.0, 400.0, value=x_norm_t_mean_default, step=0.5)
    x_norm_ca_err_mean = st.number_input("State Norm: Mean Ca Err", -0.5, 0.5, value=x_norm_ca_err_mean_default, step=0.01)
    x_norm_t_err_mean = st.number_input("State Norm: Mean T Err", -50.0, 50.0, value=x_norm_t_err_mean_default, step=0.1)

    x_norm_ca_std = st.number_input("State Norm: Std Dev Ca", 0.01, 0.5, value=x_norm_ca_std_default, step=0.01)
    x_norm_t_std = st.number_input("State Norm: Std Dev T", 1.0, 50.0, value=x_norm_t_std_default, step=0.5)
    x_norm_ca_err_std = st.number_input("State Norm: Std Dev Ca Err", 0.01, 0.5, value=x_norm_ca_err_std_default, step=0.01)
    x_norm_t_err_std = st.number_input("State Norm: Std Dev T Err", 1.0, 50.0, value=x_norm_t_err_std_default, step=0.5)

    # Action: Tc (derived from Tc bounds)
    u_norm_range = Tc_ub - Tc_lb
    u_norm_lb = Tc_lb
    st.text(f"Action Norm: Range={u_norm_range:.1f}, LB={u_norm_lb:.1f}")


    # --- AI Method Specific Parameters ---
    if ai_method == "Stochastic Policy Search (SPS)":
        st.subheader("SPS Parameters")
        # Script values: e_tot=500, e_shr=~17, shrink=0.9, radius=0.1, ratio=0.1, params=[-5, 5]
        sps_e_tot_default = 500
        sps_e_shr_default = 17
        sps_shrink_ratio_default = 0.9
        sps_radius_default = 0.1
        sps_ratio_ls_rs_default = 0.1
        sps_param_max_default = 5.0
        sps_param_min_default = -5.0

        sps_e_tot = st.number_input("Total Evaluations", 10, 1000, value=sps_e_tot_default, step=10)
        sps_e_shr = st.number_input("Evals before Shrink", 2, 50, value=sps_e_shr_default, step=1)
        sps_shrink_ratio = st.slider("Shrink Ratio", 0.1, 0.95, value=sps_shrink_ratio_default, step=0.05)
        sps_radius = st.slider("Initial Local Search Radius", 0.01, 0.5, value=sps_radius_default, step=0.01)
        sps_ratio_ls_rs = st.slider("Ratio Local/Random Search", 0.1, 0.9, value=sps_ratio_ls_rs_default, step=0.05)
        sps_param_max = st.number_input("Param Max Bound (Initial)", 1.0, 10.0, value=sps_param_max_default, step=0.1)
        sps_param_min = st.number_input("Param Min Bound (Initial)", -10.0, -1.0, value=sps_param_min_default, step=0.1)

    elif ai_method == "Policy Gradient (PG)":
        st.subheader("Policy Gradient Parameters")
        # Script values: lr=0.0001, total_it=2000, n_episodes=50, std_range=0.001 (default in func)
        pg_lr_default = 1e-4
        pg_total_it_default = 2000
        pg_n_episodes_default = 50
        pg_std_range_default = 0.001 # From script's mean_std default
        pg_std_lb_default = 0.001 # Script uses eps, use small value

        pg_lr = st.number_input("Learning Rate", 1e-6, 1e-2, value=pg_lr_default, format="%.1e")
        pg_total_it = st.number_input("Total Training Steps (Iterations)", 100, 10000, value=pg_total_it_default, step=100)
        pg_n_episodes = st.number_input("Episodes per Epoch", 5, 100, value=pg_n_episodes_default, step=1)
        pg_std_range = st.number_input("Std Dev Scaling Range", 0.0001, 5.0, value=pg_std_range_default, step=0.001, format="%.4f")
        pg_std_lb = st.number_input("Std Dev Scaling LB", 0.0, 0.5, value=pg_std_lb_default, step=0.001, format="%.3f")
        st.info("PG uses the best policy found by SPS (if run previously) as a starting point (Transfer Learning).")


    run_button = st.button("Run Simulation & Training")

# ==============================================================================
# (Rest of the Streamlit App Code remains the same)
# ...
# ==============================================================================

if 'sps_policy_dict' not in st.session_state:
    st.session_state.sps_policy_dict = None
    st.session_state.sps_cost = None
    st.session_state.policy_net_SPS_RL = None # Stores the actual torch model object
    st.session_state.sps_train_data = None # To store training plots data
    st.session_state.pg_rewards_m = None
    st.session_state.pg_rewards_std = None
    st.session_state.policy_net_PG_RL = None # Stores the PG model object
    st.session_state.pg_train_data = None # To store training plots data from PG


# ==============================================================================
# Main Application Logic
# ==============================================================================

# Prepare data dictionary once based on sidebar inputs
t = np.linspace(0, t_end, n_steps)

# Create setpoint arrays based on step changes
Ca_des = np.piecewise(t, [t < t_change1, (t >= t_change1) & (t < t_change2), t >= t_change2], [ca_sp1, ca_sp2, ca_sp3])
T_des = np.piecewise(t, [t < t_change1, (t >= t_change1) & (t < t_change2), t >= t_change2], [t_sp1, t_sp2, t_sp3])

# Initial conditions (e.g., steady state near first setpoint or arbitrary)
x0 = np.array([ca_sp1, t_sp1]) # Start at the first setpoint

# Store CSTR physical parameters (assuming they are constant here)
cstr_params_values = (
    350,    # Tf
    100,    # q
    1,      # Caf
    100,    # V
    1000,   # rho
    0.239,  # Cp
    5e4,    # mdelH
    8750,   # EoverR
    7.2e10, # k0
    5e4     # UA
)


# pack your four normalization values into vectors
x_norm_mean = np.array([
    x_norm_ca_mean,
    x_norm_t_mean,
    x_norm_ca_err_mean,
    x_norm_t_err_mean
])
x_norm_std = np.array([
    x_norm_ca_std,
    x_norm_t_std,
    x_norm_ca_err_std,
    x_norm_t_err_std
])


# Structure for simulation data and parameters
data_res = {
    't': t,
    'n': n_steps,
    'x0': x0,
    'noise_level': noise_level,
    'Ca_des': Ca_des,
    'T_des': T_des,
    'Tc_ub': Tc_ub,
    'Tc_lb': Tc_lb,
    'x_norm': (x_norm_mean, x_norm_std), # Tuple of mean and std arrays
    'u_norm': (u_norm_range, u_norm_lb), # Tuple of range and lower bound
    'cstr_params': cstr_params_values,
    # Lists to be populated during training runs
    'Ca_train': [], 'T_train': [], 'Tc_train': [],
    # PG specific params
    'pg_std_range': 1.0, # Default, updated from sidebar if PG is run
    'pg_std_lb': 0.01    # Default, updated from sidebar if PG is run
}



if run_button:
    st.session_state.pg_rewards_m = None # Clear previous PG results if re-running
    st.session_state.pg_rewards_std = None
    st.session_state.policy_net_PG_RL = None
    st.session_state.pg_train_data = None


    if ai_method == "Stochastic Policy Search (SPS)":
        st.header("Running Stochastic Policy Search (SPS)")
        # Run SPS
        best_policy_dict, best_cost, policy_net_SPS = run_sps(
            data_res, sps_e_tot, sps_e_shr, sps_shrink_ratio, sps_radius,
            sps_ratio_ls_rs, sps_param_max, sps_param_min
        )
        # Store results
        st.session_state.sps_policy_dict = best_policy_dict
        st.session_state.sps_cost = best_cost
        st.session_state.policy_net_SPS_RL = policy_net_SPS
        st.session_state.sps_train_data = {
             'Ca_train': data_res['Ca_train'],
             'T_train': data_res['T_train'],
             'Tc_train': data_res['Tc_train']
        }
        # Clear PG results if we just ran SPS
        st.session_state.policy_net_PG_RL = None
        st.session_state.pg_rewards_m = None


    elif ai_method == "Policy Gradient (PG)":
        st.header("Running Policy Gradient (PG)")
        # Check if there's an SPS policy to start from
        if st.session_state.sps_policy_dict is None:
            st.warning("No SPS policy found. Initializing PG with random weights.")
            # Initialize a random policy as starting point
            nx = 2; nu = 1
            hyparams_init = {'input_size': nx + nx, 'output_size': nu}
            initial_policy_net = Net(**hyparams_init) # Use Net for initial random params
            initial_policy_state_dict = initial_policy_net.state_dict()
        else:
            st.info("Using best policy from previous SPS run as starting point.")
            initial_policy_state_dict = st.session_state.sps_policy_dict

        # Update PG std params in data_res
        data_res['pg_std_range'] = pg_std_range
        data_res['pg_std_lb'] = pg_std_lb

        # Run PG
        rewards_m, rewards_std, policy_net_PG = run_pg(
            data_res, initial_policy_state_dict, pg_lr, pg_total_it, pg_n_episodes
        )

        # Store results
        if policy_net_PG is not None: # Check if PG run was successful
            st.session_state.pg_rewards_m = rewards_m
            st.session_state.pg_rewards_std = rewards_std
            st.session_state.policy_net_PG_RL = policy_net_PG
            st.session_state.pg_train_data = {
                 'Ca_train': data_res['Ca_train'],
                 'T_train': data_res['T_train'],
                 'Tc_train': data_res['Tc_train']
            }
        else:
            st.error("Policy Gradient training failed.")


# ==============================================================================
# Display Results
# ==============================================================================
st.header("Results")

# Determine which policy to evaluate based on last run or stored state
final_policy_net = None
final_policy_label = ""
training_data_to_plot = None

if st.session_state.policy_net_PG_RL is not None:
    final_policy_net = st.session_state.policy_net_PG_RL
    final_policy_label = "Policy Gradient (PG)"
    training_data_to_plot = st.session_state.pg_train_data
    policy_type_for_eval = 'PG_RL'
elif st.session_state.policy_net_SPS_RL is not None:
    final_policy_net = st.session_state.policy_net_SPS_RL
    final_policy_label = "Stochastic Policy Search (SPS)"
    training_data_to_plot = st.session_state.sps_train_data
    policy_type_for_eval = 'SPS_RL'

# --- Display Training Plots ---
with st.expander("Show Training Trajectories", expanded=False):
    if training_data_to_plot:
         st.write(f"Plotting {len(training_data_to_plot['Ca_train'])} trajectories from {final_policy_label} training.")
         fig_train = plot_training(
             t,
             training_data_to_plot['Ca_train'],
             training_data_to_plot['T_train'],
             training_data_to_plot['Tc_train'],
             Ca_des, T_des,
             title=f"{final_policy_label} Training Trajectories"
         )
         if fig_train:
             st.pyplot(fig_train)
         else:
              st.write("No training data available to plot.")
    else:
        st.info("Run a training method (SPS or PG) to see training trajectories.")


# --- Display PG Reward Plot ---
if st.session_state.pg_rewards_m is not None:
     with st.expander("Show Policy Gradient Reward Curve", expanded=True):
          st.write("Mean reward per epoch during PG training.")
          fig_pg_rewards = plot_pg_rewards(st.session_state.pg_rewards_m, st.session_state.pg_rewards_std)
          if fig_pg_rewards:
              st.pyplot(fig_pg_rewards)
          else:
               st.write("No PG reward data available.")


# --- Display Final Policy Simulation ---
st.subheader(f"Final Simulation using Best {final_policy_label} Policy")
if final_policy_net is not None:
    with st.spinner(f"Running final simulation with {final_policy_label} policy..."):
        # Run the simulation once with the final policy to get trajectories
        Ca_final, T_final, Tc_final = J_PolicyCSTR(
            final_policy_net, data_res, policy_alg=policy_type_for_eval, traj=True
        )

    st.write("Simulation using the best policy found during the last training run.")
    fig_final = plot_simulation(Ca_final, T_final, Tc_final, t, Ca_des, T_des, title=f"Simulation with Final {final_policy_label} Policy")
    st.pyplot(fig_final)

    # Display final cost/reward metric
    if final_policy_label == "Policy Gradient (PG)" and st.session_state.pg_rewards_m:
        st.metric(label="Final Mean Reward (PG)", value=f"{st.session_state.pg_rewards_m[-1]:.4f}")
    elif final_policy_label == "Stochastic Policy Search (SPS)" and st.session_state.sps_cost is not None:
        st.metric(label="Final Best Cost (SPS)", value=f"{st.session_state.sps_cost:.4f}")

else:
    st.info("Run a training method (SPS or PG) first to generate a final policy for simulation.")


# Add explanations or further analysis sections if desired
# st.markdown("---")
# st.subheader("Analysis")
# st.write("...")