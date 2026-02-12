"""
Utility Functions for ASCN Distributed Multi-UAV Formation Control System.

Provides formation configuration, parameter initialization, collision detection,
and auxiliary computation functions.

Reference:
    "Distributed Formation Control Driven by Recursive Equilibrium Networks
     Under Communication Attacks"
    T. Luan, F. Huang, J. Li, Y. Yang
    Knowledge-Based Systems, 2026.
"""

import torch
import numpy as np


def get_formation_config(formation_type):
    """
    Get formation configuration.
    
    Args:
        formation_type: Formation type ('triangle', 'square', 'pentagon')
    
    Returns:
        n_agents: Number of UAV agents
        formation_positions: Relative formation positions [n_agents, 3]
    """
    if formation_type == 'triangle':
        n_agents = 3
        # Equilateral triangle formation, centered at origin
        formation_positions = np.array([
            [0.0, -1.5, 0.0],   # Bottom
            [0.0, 0.75, 1.3],   # Upper right
            [0.0, 0.75, -1.3]   # Upper left
        ])
    elif formation_type == 'square':
        n_agents = 4
        # Square formation
        formation_positions = np.array([
            [0.0, -2.0, -2.0],  # Bottom-left
            [0.0, -2.0, 2.0],   # Top-left
            [0.0, 2.0, 2.0],    # Top-right
            [0.0, 2.0, -2.0]    # Bottom-right
        ])
    elif formation_type == 'pentagon':
        n_agents = 5
        # Pentagon formation
        angles = np.linspace(0, 2*np.pi, n_agents, endpoint=False)
        radius = 2.0
        formation_positions = np.array([
            [0.0, radius * np.cos(angle), radius * np.sin(angle)]
            for angle in angles
        ])
    else:
        raise ValueError(f"Unsupported formation type: {formation_type}")
    
    return n_agents, formation_positions


def get_compact_formation_config(formation_type):
    """
    Get compact formation configuration (used during obstacle traversal).
    
    Args:
        formation_type: Formation type
        
    Returns:
        formation_positions: Compact formation relative positions
    """
    if formation_type == 'triangle':
        formation_positions = np.array([
            [0.0, -0.8, 0.0],
            [0.0, 0.4, 0.7],
            [0.0, 0.4, -0.7]
        ])
    elif formation_type == 'square':
        formation_positions = np.array([
            [0.0, -1.0, -1.0],
            [0.0, -1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, -1.0]
        ])
    elif formation_type == 'pentagon':
        angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
        radius = 1.0  # Compact radius
        formation_positions = np.array([
            [0.0, radius * np.cos(angle), radius * np.sin(angle)]
            for angle in angles
        ])
    else:
        raise ValueError(f"Unsupported formation type: {formation_type}")
    
    return formation_positions


def calculate_collisions(x, sys, min_dist):
    """
    Calculate the number of collisions in the system.
    
    Args:
        x: System state history over all time steps [num_steps, state_dim]
        sys: System model
        min_dist: Minimum safe distance
    
    Returns:
        n_coll: Number of collisions
    """
    n_coll = 0
    # Iterate over all time steps
    for t in range(x.shape[0]):
        # Get all agent positions at current time step
        positions = []
        for i in range(sys.n_agents):
            pos_idx = i * 6  # Position index for each agent
            positions.append(x[t, pos_idx:pos_idx+3])
        
        # Check pairwise distances between all agents
        for i in range(sys.n_agents):
            for j in range(i+1, sys.n_agents):
                # Compute distance between two agents
                delta = positions[i] - positions[j]
                distance_sq = torch.sum(delta ** 2)
                
                # Record as collision if distance is less than minimum safe distance
                if distance_sq < min_dist ** 2:
                    n_coll += 1
    
    return n_coll


def set_params(formation_type='square', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Set system parameters and hyperparameters.
    
    Args:
        formation_type: Formation type ('triangle', 'square', 'pentagon')
        device: Computation device (CPU/GPU)
    
    Returns:
        Tuple containing all parameters
    """
    # Get number of agents based on formation type
    n_agents, _ = get_formation_config(formation_type)
    
    # System parameters
    min_dist = 0.8      # Minimum safe distance between agents
    t_end = 150         # Total simulation time steps
    Ts = 0.05           # Time step size
    
    # Get initial conditions and target states
    x0, xbar = get_ini_cond(formation_type, device)
    
    # Learning hyperparameters
    learning_rate = 1e-2  # Learning rate
    epochs = 1            # Training epochs
    
    # State weight matrix - higher weight on position, lower weight on velocity
    position_weight = 5.0
    velocity_weight = 0.5
    Q_weights = []
    for _ in range(n_agents):
        # Position weights (x, y, z)
        Q_weights.extend([position_weight] * 3) 
        # Velocity weights (vx, vy, vz)
        Q_weights.extend([velocity_weight] * 3)
    
    Q = torch.diag(torch.tensor(Q_weights, device=device))
    
    # Adjusted loss function weights
    alpha_u = 0.05      # Reduced control input loss weight, allowing larger control inputs
    alpha_ca = 1e3      # Collision avoidance loss weight (unchanged)
    alpha_side = 1e2    # Reduced boundary loss weight, as boundary is not the main concern
    alpha_form = 1e1    # Reduced formation maintenance loss weight, less strict during takeoff
    alpha_x = 1e-1      # Increased state tracking loss weight, encouraging active target pursuit
    alpha_vel = 0.2     # Reduced velocity constraint loss weight, allowing faster speeds
    alpha_smooth = 0.1  # Reduced control smoothness loss weight, allowing aggressive control changes
    alpha_obstacle = 20 # Obstacle avoidance loss weight, comparable to collision avoidance
    
    # RBN network parameters - dynamically generated per agent count
    n_xi = np.array([30] * n_agents)  # RBN internal state dimension
    l = np.array([30] * n_agents)     # RBN number of nonlinear layers
    
    # Training parameters
    n_traj = 1      # Number of trajectories per training step
    std_ini = 0.1   # Standard deviation of initial condition perturbation
    
    return min_dist, t_end, n_agents, x0, xbar, learning_rate, epochs, Q, \
           alpha_u, alpha_ca, alpha_side, alpha_form, alpha_x, alpha_vel, alpha_smooth, \
           alpha_obstacle, n_xi, l, n_traj, std_ini, Ts, formation_type


def get_ini_cond(formation_type='square', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Set initial and target positions for UAV agents based on formation type.
    
    Args:
        formation_type: Formation type
        device: Computation device (CPU/GPU)
    
    Returns:
        x0: Initial state vector
        xbar: Target state vector
    """
    n_agents, formation_positions = get_formation_config(formation_type)
    
    # Create state vectors (each agent has 6 states: x, y, z, vx, vy, vz)
    state_dim = n_agents * 6
    x0 = torch.zeros(state_dim, device=device)
    xbar = torch.zeros(state_dim, device=device)
    
    # Set initial positions (left side) and target positions (right side)
    initial_offset = [-8, 0, 0]  # Initial position offset
    target_offset = [8, 0, 0]    # Target position offset
    
    # Fill state vectors
    for i in range(n_agents):
        # Initial positions
        x0[i*6] = formation_positions[i, 0] + initial_offset[0]     # x
        x0[i*6 + 1] = formation_positions[i, 1] + initial_offset[1] # y
        x0[i*6 + 2] = formation_positions[i, 2] + initial_offset[2] # z
        # Initial velocities are zero
        x0[i*6 + 3:i*6 + 6] = 0
        
        # Target positions
        xbar[i*6] = formation_positions[i, 0] + target_offset[0]     # x
        xbar[i*6 + 1] = formation_positions[i, 1] + target_offset[1] # y
        xbar[i*6 + 2] = formation_positions[i, 2] + target_offset[2] # z
        # Target velocities are zero
        xbar[i*6 + 3:i*6 + 6] = 0
    
    return x0, xbar


def calculate_formation_error(x, sys, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Calculate formation error - deviation of actual formation from ideal formation.
    
    Args:
        x: System state
        sys: System model
        
    Returns:
        formation_error: Formation error value
    """
    # Extract all agent positions
    positions = []
    for i in range(sys.n_agents):
        pos_idx = i * 6
        positions.append(x[pos_idx:pos_idx+3])
    
    # Compute formation centroid
    centroid = torch.zeros(3, device=device)
    for pos in positions:
        centroid += pos
    centroid /= sys.n_agents
    
    # Get current ideal formation using system's get_ideal_formation method
    ideal_formation = sys.get_ideal_formation(x)
    
    # Compute error between actual and ideal positions
    formation_error = 0.0
    for i in range(sys.n_agents):
        ideal_pos = centroid + ideal_formation[i]
        error = torch.sum((positions[i] - ideal_pos) ** 2)
        formation_error += error
    
    return formation_error


def calculate_mean_distance_to_target(x, xbar, sys):
    """
    Calculate mean distance from agents to target positions.
    
    Args:
        x: Current system state
        xbar: Target state
        sys: System model
        
    Returns:
        mean_dist: Mean distance
    """
    total_dist = 0.0
    
    for i in range(sys.n_agents):
        pos_idx = i * 6
        current_pos = x[pos_idx:pos_idx+3]
        target_pos = xbar[pos_idx:pos_idx+3]
        
        # Compute Euclidean distance
        dist = torch.sqrt(torch.sum((current_pos - target_pos) ** 2))
        total_dist += dist
    
    mean_dist = total_dist / sys.n_agents
    return mean_dist


def save_trajectory_data(filename, x_log, u_log, sys):
    """
    Save trajectory data to numpy file.
    
    Args:
        filename: Output filename
        x_log: State history
        u_log: Control input history
        sys: System model
    """
    # Convert to CPU tensors, then to numpy arrays
    x_log_np = x_log.cpu().detach().numpy()
    u_log_np = u_log.cpu().detach().numpy()
    xbar_np = sys.xbar.cpu().detach().numpy()
    
    # Save as numpy file
    np.savez(
        filename,
        states=x_log_np,
        controls=u_log_np,
        target=xbar_np,
        n_agents=sys.n_agents,
        t_end=x_log_np.shape[0]
    )
    
    print(f"Trajectory data saved to {filename}")


def generate_circular_trajectory(t, radius, center, freq, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Generate reference path for circular trajectory.
    
    Args:
        t: Current time step
        radius: Circle radius
        center: Center coordinates [x, y, z]
        freq: Number of time steps to complete one revolution
        device: Computation device
        
    Returns:
        position: Reference position [x, y, z]
        velocity: Reference velocity [vx, vy, vz]
    """
    # Compute current angle
    angle = 2 * np.pi * t / freq
    
    # Compute position
    x = center[0] + radius * torch.cos(torch.tensor(angle, device=device))
    y = center[1] + radius * torch.sin(torch.tensor(angle, device=device))
    z = center[2]
    
    # Compute velocity
    vx = -radius * 2 * np.pi / freq * torch.sin(torch.tensor(angle, device=device))
    vy = radius * 2 * np.pi / freq * torch.cos(torch.tensor(angle, device=device))
    vz = 0
    
    position = torch.tensor([x, y, z], device=device)
    velocity = torch.tensor([vx, vy, vz], device=device)
    
    return position, velocity


def create_position_mask(n_agents, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Create position mask for adding perturbation only to positions, not velocities.
    
    Args:
        n_agents: Number of UAV agents
        device: Computation device
        
    Returns:
        mask: Position mask tensor
    """
    state_dim = n_agents * 6
    mask = torch.zeros(state_dim, device=device)
    
    for i in range(n_agents):
        # Set mask to 1 for position components (x, y, z)
        mask[i*6:i*6+3] = 1.0
        # Velocity components (vx, vy, vz) remain 0
    
    return mask


def calculate_energy_consumption(u_log):
    """
    Calculate energy consumption from control inputs.
    
    Args:
        u_log: Control input history [num_steps, control_dim]
        
    Returns:
        energy: Total energy consumption
    """
    # Sum of squares as an estimate of energy consumption
    energy = torch.sum(u_log ** 2)
    return energy


def validate_formation_config(formation_type):
    """
    Validate whether the formation configuration is valid.
    
    Args:
        formation_type: Formation type
        
    Returns:
        bool: Whether the configuration is valid
    """
    valid_formations = ['triangle', 'square', 'pentagon']
    if formation_type not in valid_formations:
        print(f"Error: Unsupported formation type '{formation_type}'")
        print(f"Supported formation types: {valid_formations}")
        return False
    
    n_agents, positions = get_formation_config(formation_type)
    print(f"Formation type: {formation_type}")
    print(f"Number of agents: {n_agents}")
    print(f"Formation positions:\n{positions}")
    
    return True


def get_formation_info(formation_type):
    """
    Get detailed formation information.
    
    Args:
        formation_type: Formation type
        
    Returns:
        dict: Dictionary containing formation information
    """
    n_agents, formation_positions = get_formation_config(formation_type)
    compact_positions = get_compact_formation_config(formation_type)
    
    info = {
        'formation_type': formation_type,
        'n_agents': n_agents,
        'state_dim': n_agents * 6,
        'control_dim': n_agents * 3,
        'standard_positions': formation_positions,
        'compact_positions': compact_positions
    }
    
    return info


def verify_heterogeneous_equivalence(comm_network, theta=0.5):
    """
    Verify effective capacity equivalence of heterogeneous communication network.
    
    Args:
        comm_network: Communication network object
        theta: QoS exponent parameter (default: 0.5)
        
    Returns:
        dict: Equivalence verification results
    """
    # Homogeneous baseline network parameters
    homo_bandwidth = 50*1024
    homo_delay = 0.02
    homo_loss = 0.05
    
    # Compute homogeneous network effective capacity
    homo_ec = homo_bandwidth * (1 - homo_loss) / (1 + theta * homo_delay * homo_bandwidth)
    
    # Compute effective capacity for each heterogeneous channel
    channel_ecs = []
    channel_types = {}
    
    for channel in comm_network.channels.values():
        ec = channel.bandwidth * (1 - channel.packet_loss_rate) / \
             (1 + theta * channel.base_delay * channel.bandwidth)
        channel_ecs.append(ec)
        
        # Get channel type
        ch_type = getattr(channel, 'channel_type', 'unknown')
        channel_types[ch_type] = channel_types.get(ch_type, 0) + 1
    
    # Compute average effective capacity
    hetero_ec_avg = sum(channel_ecs) / len(channel_ecs) if channel_ecs else 0
    
    # Compute equivalence ratio
    equivalence_ratio = hetero_ec_avg / homo_ec if homo_ec > 0 else 0
    
    result = {
        'homo_ec': homo_ec,
        'hetero_ec_avg': hetero_ec_avg,
        'equivalence_ratio': equivalence_ratio,
        'is_equivalent': 0.95 <= equivalence_ratio <= 1.05,
        'channel_distribution': channel_types,
        'total_channels': len(comm_network.channels)
    }
    
    return result
