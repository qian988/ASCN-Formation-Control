#!/usr/bin/env python
"""
Distributed Multi-UAV Formation Control System with Adaptive Mode Switching
Supports 3 cylindrical obstacles, formation weight parameters removed.
"""
import numpy as np
import torch
import scipy.io as sio
import os 
import time
import random
from datetime import datetime

from models import QuadcopterSystem, DistributedController
# Formation-related loss functions removed
from loss_functions import (f_loss_states, f_loss_u, f_loss_ca, f_loss_side, 
                            f_loss_velocity, f_loss_cylinder_obstacle,
                            CYLINDER_OBSTACLES)
from utils import calculate_collisions, set_params, create_position_mask, validate_formation_config
from attack_models import DOSAttackModel, AttackType, AttackScenarioGenerator, MultiPhaseAttack

# ===== Configuration Parameters =====
FORMATION_TYPE = 'triangle'       # Options: 'triangle', 'square', 'pentagon'
ADVERSARIAL_TRAINING = True       # Enable adversarial training
BASIC_TRAINING_RATIO = 0.1       # Proportion of basic training epochs
HETEROGENEOUS_COMM = False        # Enable heterogeneous communication
MODE_SWITCHING_ENABLED = True     # Enable mode switching functionality

# Validate formation configuration
if not validate_formation_config(FORMATION_TYPE):
    print("Formation configuration validation failed, exiting")
    exit(1)

# Set computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Formation type: {FORMATION_TYPE.upper()}")
print(f"Distributed control mode: Enabled")
print(f"Adversarial training mode: {'Enabled' if ADVERSARIAL_TRAINING else 'Disabled'}")
print(f"Adaptive mode switching: {'Enabled' if MODE_SWITCHING_ENABLED else 'Disabled'}")
print(f"Obstacle configuration:")
print(f"   Cylinder 1: x=-2.5, y=1, z_range=[-10, 10], radius=0.5")
print(f"   Cylinder 2: x=-2.5, y=-1, z_range=[-10, 10], radius=0.5")
print(f"   Cylinder 3: x=2.5, y=0, z_range=[-10, 10], radius=0.5")
print(f"Note: Formation weight parameters removed, focusing on obstacle avoidance")

# Set random seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# ===== Difficulty-First Strategy Class =====
class DifficultyFirstScheduler:
    """Difficulty-first training scheduler"""
    def __init__(self, total_epochs, basic_ratio=0.1):
        self.total_epochs = total_epochs
        self.basic_epochs = int(total_epochs * basic_ratio)
        self.adversarial_epochs = total_epochs - self.basic_epochs
        
        self.performance_history = {
            'normal': [],
            'light': [],
            'medium': [], 
            'heavy': [],
            'extreme': []
        }
        
        self.recent_losses = {
            'normal': float('inf'),
            'light': float('inf'),
            'medium': float('inf'),
            'heavy': float('inf'),
            'extreme': float('inf')
        }
        
    def get_scenario(self, epoch, training_attack_models):
        """Select training scenario based on difficulty-first strategy"""
        if epoch < self.basic_epochs:
            return [], "Basic training"
        
        adv_epoch = epoch - self.basic_epochs
        adv_progress = adv_epoch / self.adversarial_epochs
        
        cycle = adv_epoch % 10
        rand = random.random()
        
        if adv_progress < 0.3:
            if cycle < 3:
                if rand < 0.7:
                    return [], "Normal flight"
                else:
                    attacks = [training_attack_models['early_light']]
                    return attacks, f"Light attack[1]"
            elif cycle < 5:
                if rand < 0.3:
                    return [], "Normal flight"
                else:
                    attacks = [
                        training_attack_models['early_heavy'],
                        training_attack_models['mid_sustained']
                    ]
                    return attacks, f"Medium combo[2]"
            elif cycle < 7:
                attacks = [
                    training_attack_models['early_light'],
                    training_attack_models['mid_protocol'],
                    training_attack_models['late_critical']
                ]
                return attacks, f"Heavy combo[3]"
            else:
                worst_type = max(self.recent_losses, key=self.recent_losses.get)
                if worst_type == 'normal':
                    return [], "Normal flight(supplement)"
                elif worst_type in ['light', 'medium']:
                    attacks = random.sample([
                        training_attack_models['early_light'],
                        training_attack_models['mid_sustained'],
                        training_attack_models['early_heavy']
                    ], 2)
                    return attacks, f"Supplement[{len(attacks)}]"
                else:
                    attacks = random.sample([
                        training_attack_models['mid_sustained'],
                        training_attack_models['late_critical'],
                        training_attack_models['late_selective'],
                        training_attack_models['coordinated_heavy']
                    ], 3)
                    return attacks, f"Hard supplement[{len(attacks)}]"
        
        elif adv_progress < 0.7:
            if cycle < 2:
                if rand < 0.5:
                    return [], "Normal flight"
                else:
                    attacks = [training_attack_models['early_light']]
                    return attacks, f"Light attack[1]"
            elif cycle < 4:
                attacks = random.sample([
                    training_attack_models['early_heavy'],
                    training_attack_models['mid_sustained'],
                    training_attack_models['mid_protocol']
                ], 2)
                return attacks, f"Medium combo[2]"
            elif cycle < 7:
                if rand < 0.5:
                    attacks = random.sample([
                        training_attack_models['early_heavy'],
                        training_attack_models['mid_sustained'],
                        training_attack_models['late_critical'],
                        training_attack_models['mid_protocol']
                    ], 3)
                    return attacks, f"Heavy combo[3]"
                else:
                    attacks = random.sample([
                        training_attack_models['early_light'],
                        training_attack_models['continuous_wave'],
                        training_attack_models['mid_protocol'],
                        training_attack_models['late_selective']
                    ], 4)
                    return attacks, f"Heavy combo[4]"
            else:
                num_attacks = 4 if rand < 0.5 else 5
                all_attacks = list(training_attack_models.values())
                attacks = random.sample(all_attacks, num_attacks)
                return attacks, f"Extreme combo[{num_attacks}]"
        
        else:
            if cycle < 2:
                if rand < 0.6:
                    return [], "Normal flight"
                else:
                    attacks = random.sample([
                        training_attack_models['early_light'],
                        training_attack_models['mid_sustained']
                    ], 1)
                    return attacks, f"Light attack[1]"
            elif cycle < 3:
                attacks = random.sample(list(training_attack_models.values()), 2)
                return attacks, f"Medium combo[2]"
            elif cycle < 6:
                attacks = random.sample(list(training_attack_models.values()), 3)
                return attacks, f"Heavy combo[3]"
            else:
                if rand < 0.3:
                    attacks = [
                        training_attack_models['early_heavy'],
                        training_attack_models['mid_sustained'],
                        training_attack_models['late_critical'],
                        training_attack_models['coordinated_heavy']
                    ]
                    return attacks, f"Extreme combo[4]"
                else:
                    attacks = [
                        training_attack_models['early_heavy'],
                        training_attack_models['mid_sustained'],
                        training_attack_models['mid_protocol'],
                        training_attack_models['late_critical'],
                        training_attack_models['late_selective']
                    ]
                    return attacks, f"Extreme combo[5]"
    
    def update_performance(self, scenario_type, loss_value):
        """Update performance record"""
        if "Basic" in scenario_type or scenario_type == "Normal flight" or "Normal" in scenario_type:
            key = 'normal'
        elif "Light" in scenario_type and "[1]" in scenario_type:
            key = 'light'
        elif "Medium" in scenario_type or "[2]" in scenario_type:
            key = 'medium'
        elif "Heavy" in scenario_type or "[3]" in scenario_type:
            key = 'heavy'
        elif "Extreme" in scenario_type or "[4]" in scenario_type or "[5]" in scenario_type:
            key = 'extreme'
        else:
            key = 'normal'
        
        self.performance_history[key].append(loss_value)
        
        alpha = 0.1
        if len(self.performance_history[key]) == 1:
            self.recent_losses[key] = loss_value
        else:
            self.recent_losses[key] = (1 - alpha) * self.recent_losses[key] + alpha * loss_value
    
    def get_statistics(self):
        """Get training statistics"""
        stats = {}
        for key, history in self.performance_history.items():
            if history:
                stats[key] = {
                    'count': len(history),
                    'avg_loss': sum(history) / len(history),
                    'recent_loss': self.recent_losses[key]
                }
        return stats


# Create output directories
os.makedirs("trained_models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Set parameters
min_dist, t_end, n_agents, x0, xbar, learning_rate, epochs, Q, \
    alpha_u, alpha_ca, alpha_side, alpha_form, alpha_x, alpha_vel, alpha_smooth, \
    alpha_obstacle, n_xi, l, n_traj, std_ini, Ts, formation_type = set_params(FORMATION_TYPE, device)

# Only keep cylinder obstacle avoidance weight, formation weights removed
alpha_cylinder = alpha_obstacle * 1.5   # Cylinder obstacle avoidance weight

print(f"Formation configuration complete:")
print(f"   Number of agents: {n_agents}")
print(f"   State dimension per agent: 6")
print(f"   Control dimension per agent: 3")
print(f"   Training epochs: {epochs}")
print(f"   Cylinder obstacle weight: {alpha_cylinder}")
print(f"   Note: Formation maintenance weight parameters removed")

# System model description
sys_model = f'DistributedRBN_{formation_type}_{n_agents}agents_obstacle_only'

# Create distributed quadcopter system
print("Initializing distributed quadcopter system...")
sys = QuadcopterSystem(
    target_states=xbar, 
    Ts=Ts,  
    formation_type=formation_type, 
    enable_communication=True, 
    heterogeneous_mode=HETEROGENEOUS_COMM, 
    device=device
)

# Print communication mode info
if HETEROGENEOUS_COMM:
    print(f"Communication mode: Heterogeneous (based on Wu & Negi EC theory)")
else:
    print(f"Communication mode: Homogeneous")

if HETEROGENEOUS_COMM and sys.enable_communication and sys.comm_network:
    print("Verifying heterogeneous communication configuration...")
    
    channel_types = {}
    missing_time_step = []
    
    for channel_key, channel in sys.comm_network.channels.items():
        ch_type = getattr(channel, 'channel_type', 'unknown')
        channel_types[ch_type] = channel_types.get(ch_type, 0) + 1
        
        if not hasattr(channel, 'time_step_size'):
            missing_time_step.append(channel_key)
            channel.time_step_size = Ts
    
    print(f"   Heterogeneous channel type distribution: {channel_types}")
    if missing_time_step:
        print(f"   Fixed channels missing time_step_size: {len(missing_time_step)}")
    
    from utils import verify_heterogeneous_equivalence
    equiv_result = verify_heterogeneous_equivalence(sys.comm_network)
    print(f"   Equivalence verification: {'Passed' if equiv_result['is_equivalent'] else 'Failed'} "
          f"(ratio={equiv_result['equivalence_ratio']:.3f})")
    
    if not equiv_result['is_equivalent']:
        print(f"   Warning: Heterogeneous network equivalence verification failed, may affect performance")

# Convert initial state to distributed states
initial_distributed_states = sys.convert_to_distributed_states(x0)
print(f"Distributed state initialization complete, per-agent state dimensions: {[s.shape for s in initial_distributed_states]}")

# Create distributed target states
distributed_targets = sys.target_states

# Display formation info
formation_info = sys.get_formation_info()
print(f"Formation details:")
for key, value in formation_info.items():
    print(f"   {key}: {value}")

# Display communication topology info
if sys.enable_communication and sys.comm_network:
    topology_info = sys.comm_network.get_topology_info()
    print(f"Communication topology info:")
    print(f"   Active links: {topology_info['active_links']}")
    print(f"   Connectivity ratio: {topology_info['connectivity_ratio']:.2f}")
    print(f"   Topology matrix: {topology_info['topology_matrix']}")

# Create distributed controller
print("Initializing distributed controller...")
ctl = DistributedController(n_agents, max_neighbors=3, device=device)

# Display controller info
controller_info = ctl.get_controller_info()
print(f"Distributed controller info:")
for key, value in controller_info.items():
    if key != 'agent_requirements':
        print(f"   {key}: {value}")

# Define optimizer
optimizer = torch.optim.Adam(ctl.parameters(), lr=learning_rate)

# Create difficulty-first scheduler
difficulty_scheduler = DifficultyFirstScheduler(epochs, BASIC_TRAINING_RATIO)

if ADVERSARIAL_TRAINING:
    print("Initializing combined attack adversarial training models...")
    print("Training strategy: Difficulty-first strategy")
    
    basic_training_epochs = int(epochs * BASIC_TRAINING_RATIO)
    
    training_attack_models = {
        'early_light': DOSAttackModel(
            attack_type=AttackType.BANDWIDTH_FLOODING,
            intensity=0.3,
            start_time=15,
            duration=20,
            target_agents=list(range(min(2, n_agents))),
            n_agents=n_agents
        ),
        'early_heavy': DOSAttackModel(
            attack_type=AttackType.BANDWIDTH_FLOODING,
            intensity=0.5,
            start_time=10,
            duration=25,
            target_agents=list(range(min(2, n_agents))),
            n_agents=n_agents
        ),
        'mid_sustained': DOSAttackModel(
            attack_type=AttackType.SELECTIVE_JAMMING,
            intensity=0.3,
            start_time=50,
            duration=35,
            target_agents=list(range(n_agents)),
            n_agents=n_agents
        ),
        'mid_protocol': DOSAttackModel(
            attack_type=AttackType.PROTOCOL_DISRUPTION,
            intensity=0.5,
            start_time=40,
            duration=30,
            target_agents=list(range(n_agents)),
            n_agents=n_agents
        ),
        'late_critical': DOSAttackModel(
            attack_type=AttackType.SELECTIVE_JAMMING,
            intensity=0.6,
            start_time=100,
            duration=25,
            target_agents=list(range(n_agents)),
            n_agents=n_agents
        ),
        'late_selective': DOSAttackModel(
            attack_type=AttackType.SELECTIVE_JAMMING,
            intensity=0.85,
            start_time=110,
            duration=20,
            target_agents=list(range(n_agents)),
            n_agents=n_agents
        ),
        'continuous_wave': DOSAttackModel(
            attack_type=AttackType.BANDWIDTH_FLOODING,
            intensity=0.8,
            start_time=30,
            duration=40,
            target_agents=list(range(min(2, n_agents))),
            n_agents=n_agents
        ),
        'coordinated_heavy': DOSAttackModel(
            attack_type=AttackType.COORDINATED_ATTACK,
            intensity=0.6,
            start_time=80,
            duration=30,
            target_agents=list(range(n_agents)),
            n_agents=n_agents
        )
    }
    
    print(f"Difficulty-first training configuration:")
    print(f"   First {basic_training_epochs} epochs: Basic training (no attacks)")
    print(f"   Remaining {epochs - basic_training_epochs} epochs: Difficulty-first adversarial training")

# Simulate open-loop trajectory (no control input)
print("Simulating open-loop trajectory (no control input)...")
x_log = torch.zeros((t_end, n_agents * 6), device=device)
u_log = torch.zeros((t_end, n_agents * 3), device=device)

distributed_states = [state.clone() for state in initial_distributed_states]
zero_controls = [torch.zeros(3, device=device) for _ in range(n_agents)]

for t in range(t_end):
    distributed_states = sys(t, distributed_states, zero_controls)
    global_state = sys.convert_to_global_state(distributed_states)
    global_control = torch.cat(zero_controls)
    x_log[t, :] = global_state.detach()
    u_log[t, :] = global_control.detach()

# Start distributed training
print(f"\n--------- Starting Difficulty-First Distributed Adversarial Training (Obstacle-Focus Mode) ---------")
print(f"Formation type: {formation_type.upper()} ({n_agents} agents)")
print(f"Problem: {sys_model} -- Time steps: {t_end} -- Learning rate: {learning_rate:.2e}")
print(f"Epochs: {epochs} -- Trajectories: {n_traj} -- Initial perturbation: {std_ini:.2f}")
print(f"Control weight: {alpha_u:.1f} -- Collision weight: {alpha_ca} -- Boundary weight: {alpha_side:.1e}")
print(f"Cylinder obstacle weight: {alpha_cylinder}")
print(f"Note: Formation maintenance weights and losses removed")
print("Strict distributed mode: each agent can only access local state and communication data")
print("--------- --------- ---------  ---------")

# Initialize loss recording arrays
lossl = np.zeros(epochs)
lossxl = np.zeros(epochs)
lossul = np.zeros(epochs)
losscal = np.zeros(epochs)
lossvel = np.zeros(epochs)
losssidel = np.zeros(epochs)
losscyll = np.zeros(epochs)  # Cylinder obstacle avoidance loss

# Attack statistics
attack_epochs = []
attack_types_used = []
attack_combinations_used = []
scenario_type_history = []

# Mode switching statistics
mode_switch_stats_per_epoch = []

# Create position mask
position_mask = create_position_mask(n_agents, device)

# Save best epoch data
best_loss = float('inf')
best_epoch = -1
best_epoch_trajectories = []
last_epoch_trajectories = []

def save_model_with_formation_info(model_state, filename_base):
    """Save model with formation information"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trained_models/{filename_base}_{formation_type}_{n_agents}agents_{timestamp}.pt"
    torch.save(model_state, filename)
    print(f"Model saved: {filename}")
    return filename

# ===== Difficulty-First Distributed Adversarial Training Loop =====
training_start_time = time.time()

# Display communication topology info
if sys.enable_communication and sys.comm_network:
    topology_info = sys.comm_network.get_topology_info()
    print(f"Communication topology info:")
    print(f"   Active links: {topology_info['active_links']}")
    print(f"   Connectivity ratio: {topology_info['connectivity_ratio']:.2f}")

for epoch in range(epochs):
    attack_session_started = False
    epoch_start_time = time.time()
    optimizer.zero_grad()
    
    # Use difficulty-first strategy to determine attack scenario
    active_attack_models, attack_info = difficulty_scheduler.get_scenario(epoch, training_attack_models)
    attack_enabled = len(active_attack_models) > 0
    
    scenario_type_history.append(attack_info)
    
    if attack_enabled:
        for attack_model in active_attack_models:
            attack_epochs.append(epoch)
            attack_types_used.append(attack_model.attack_type.value)
        
        combination_str = f"{attack_info}_{len(active_attack_models)}attacks"
        attack_combinations_used.append(combination_str)
    
    sys.stop_communication_attack()
    ctl.reset_all_states()
    
    # Initialize loss terms - formation-related losses removed
    loss_x, loss_u, loss_ca, loss_side, loss_vel = 0, 0, 0, 0, 0
    loss_cylinder = torch.tensor(0.0, device=device)
    
    current_epoch_trajectories = []
    
    epoch_mode_switches = {'total_switches': 0, 'agent_switches': {i: 0 for i in range(n_agents)}}
    epoch_emergency_time = {'total_steps': 0, 'agent_emergency_steps': {i: 0 for i in range(n_agents)}}
    
    if epoch > epochs // 2:
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * 0.5
    
    # Collect multiple trajectories
    for kk in range(n_traj):
        distributed_states = []
        for i in range(n_agents):
            agent_initial = initial_distributed_states[i].clone()
            agent_perturbation = std_ini * torch.randn(6, device=device)
            agent_perturbation[3:] = 0
            agent_state = agent_initial + agent_perturbation
            distributed_states.append(agent_state)
        
        initial_states_copy = [s.clone() for s in distributed_states]
        
        traj_states = []
        traj_controls = []
        
        persistent_hover_status = [False] * n_agents
       
        # Distributed trajectory simulation
        for t in range(t_end):
            # Apply combined training attacks
            if attack_enabled and active_attack_models:
                for attack_model in active_attack_models:
                    if attack_model.is_attack_active(t):
                        sys.apply_communication_attack(attack_model, t)
            
            # Communication update
            communication_data = sys.distributed_communication_update(distributed_states, t, "training")
            
            if MODE_SWITCHING_ENABLED:
                comm_quality_scores = sys.evaluate_communication_quality(communication_data, t)
            else:
                comm_quality_scores = None
            
            # Distributed control computation
            distributed_controls = ctl(distributed_states, communication_data, comm_quality_scores)
            
            if MODE_SWITCHING_ENABLED and t % 20 == 0:
                sys.update_agent_modes(ctl)
                emergency_count = sum(1 for mode in sys.agent_modes if mode == 'emergency')
                for i, mode in enumerate(sys.agent_modes):
                    if mode == 'emergency':
                        epoch_emergency_time['agent_emergency_steps'][i] += 1
                        epoch_emergency_time['total_steps'] += 1
            
            # System evolution
            distributed_states = sys(t, distributed_states, distributed_controls)
            
            # Handle agents that have reached their targets
            for i in range(n_agents):
                if persistent_hover_status[i]:
                    target_state = sys.target_states[i]
                    distributed_states[i][:3] = target_state[:3]
                    distributed_states[i][3:] = target_state[3:]
                    
                    hover_control = torch.zeros(3, device=device)
                    hover_control[2] = sys.mass * sys.g
                    distributed_controls[i] = hover_control
                    continue
                
                current_pos = distributed_states[i][:3]
                target_pos = sys.target_states[i][:3]
                dist = torch.sqrt(torch.sum((current_pos - target_pos) ** 2))
                
                if dist < 0.1:
                    persistent_hover_status[i] = True
                    distributed_states[i][:3] = target_pos
                    distributed_states[i][3:] = sys.target_states[i][3:]
                    
                    hover_control = torch.zeros(3, device=device)
                    hover_control[2] = sys.mass * sys.g
                    distributed_controls[i] = hover_control
            
            # Record trajectory
            global_state = sys.convert_to_global_state(distributed_states)
            global_control = torch.cat(distributed_controls)
            
            traj_states.append(global_state.clone())
            traj_controls.append(global_control.clone())
            
            # Compute losses - formation-related losses removed
            time_weight = 1.0 + max(0, (t - (t_end - 30)) / 10.0) * 4.0
            loss_x = loss_x + alpha_x * time_weight * f_loss_states(t, distributed_states, distributed_targets, sys, Q)
            loss_u = loss_u + alpha_u * f_loss_u(t, distributed_controls)
            loss_ca = loss_ca + alpha_ca * f_loss_ca(distributed_states, communication_data, sys, min_dist)
            loss_side = loss_side + alpha_side * f_loss_side(distributed_states, sys)
            loss_vel = loss_vel + alpha_vel * f_loss_velocity(distributed_states, sys)
            
            # Only compute cylinder obstacle loss
            loss_cylinder = loss_cylinder + alpha_cylinder * f_loss_cylinder_obstacle(distributed_states, distributed_targets, sys)
        
        current_epoch_trajectories.append({
            "states": torch.stack(traj_states),
            "controls": torch.stack(traj_controls),
            "initial_states": sys.convert_to_global_state(initial_states_copy)
        })
    
    # End-of-epoch: clean up attack state
    if attack_enabled:
        for i, attack_model in enumerate(active_attack_models):
            attack_model.restore_network(sys.comm_network)
        sys.stop_communication_attack()
    
    if MODE_SWITCHING_ENABLED:
        mode_stats = ctl.get_mode_statistics()
        for agent_id, stats in mode_stats.items():
            epoch_mode_switches['agent_switches'][stats['agent_id']] = stats['switch_count']
            epoch_mode_switches['total_switches'] += stats['switch_count']
        
        mode_switch_stats_per_epoch.append({
            'epoch': epoch,
            'switches': epoch_mode_switches,
            'emergency_time': epoch_emergency_time,
            'attack_info': attack_info
        })
    
    # Compute total loss - only obstacle avoidance and basic losses
    loss = loss_x + loss_u + loss_ca + loss_side + loss_vel + loss_cylinder
    
    current_loss_value = loss.item()
    difficulty_scheduler.update_performance(attack_info, current_loss_value / t_end)
    
    if current_loss_value < best_loss:
        best_loss = current_loss_value
        best_epoch = epoch
        
        save_model_with_formation_info(ctl.state_dict(), f"{sys_model}_best_epoch")
        
        best_epoch_trajectories = [
            {
                "states": traj["states"].clone(),
                "controls": traj["controls"].clone(),
                "initial_states": traj["initial_states"].clone()
            }
            for traj in current_epoch_trajectories
        ]
        print(f"New best epoch: {epoch} - Loss: {current_loss_value / t_end:.4f} - {attack_info}")
    
    if epoch == epochs - 1:
        last_epoch_trajectories = [
            {
                "states": traj["states"].clone(),
                "controls": traj["controls"].clone(),
                "initial_states": traj["initial_states"].clone()
            }
            for traj in current_epoch_trajectories
        ]
    
    # Print training info - formation loss removed
    epoch_time = time.time() - epoch_start_time
    if epoch % 1 == 0 or epoch == epochs - 1:
        print(f"Epoch: {epoch:4d} - {attack_info}")
        print(f"       Total loss: {current_loss_value / t_end:.4f}")
        print(f"       State loss: {loss_x.item():.2f} - Control loss: {loss_u.item():.2f} - "
              f"Collision loss: {loss_ca.item():.2f}")
        print(f"       Boundary loss: {loss_side.item():.2f} - Velocity loss: {loss_vel.item():.2f}")
        print(f"       Cylinder obstacle loss: {loss_cylinder.item():.2f} - Time: {epoch_time:.2f}s")
        print(f"       Controller gain: {ctl.amplifier.item():.4f}")
        
        if MODE_SWITCHING_ENABLED:
            system_status = ctl.get_system_mode_summary()
            print(f"       Mode status: {system_status['normal_agents']} normal/{system_status['emergency_agents']} emergency")
        
        stats = difficulty_scheduler.get_statistics()
        if stats:
            print(f"       Scenario stats: ", end="")
            for key, info in stats.items():
                print(f"{key}:{info['count']}x ", end="")
            print()
            
    elif epoch % 10 == 0:
        attack_count = len(active_attack_models)
        attack_short = f"[{attack_count}]" if attack_enabled else "[0]"
        print(f"Epoch: {epoch:4d}{attack_short} - {attack_info[:20]:20s} - Loss: {current_loss_value / t_end:.4f} - Time: {epoch_time:.2f}s")
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    # Record losses - formation loss removed
    lossl[epoch] = loss.detach().cpu().numpy()
    lossxl[epoch] = loss_x.detach().cpu().numpy()
    lossul[epoch] = loss_u.detach().cpu().numpy()
    losscal[epoch] = loss_ca.detach().cpu().numpy()
    losssidel[epoch] = loss_side.detach().cpu().numpy()
    lossvel[epoch] = loss_vel.detach().cpu().numpy()
    losscyll[epoch] = loss_cylinder.detach().cpu().numpy()

training_time = time.time() - training_start_time

# Training statistics
if ADVERSARIAL_TRAINING:
    final_stats = difficulty_scheduler.get_statistics()
    
    print(f"\nDifficulty-first adversarial training statistics:")
    print(f"   Basic training epochs: {int(epochs * BASIC_TRAINING_RATIO)}")
    print(f"   Adversarial training epochs: {epochs - int(epochs * BASIC_TRAINING_RATIO)}")
    
    print(f"\n   Scenario distribution statistics:")
    total_scenarios = sum(info['count'] for info in final_stats.values())
    for scenario_type, info in final_stats.items():
        percentage = (info['count'] / total_scenarios * 100) if total_scenarios > 0 else 0
        print(f"     {scenario_type:8s}: {info['count']:4d}x ({percentage:5.1f}%) - Avg loss: {info['avg_loss']:.4f}")

if MODE_SWITCHING_ENABLED:
    print(f"\n--------- Mode Switching Analysis ---------")
    final_mode_stats = ctl.get_mode_statistics()
    
    print(f"Per-agent mode switching statistics:")
    for agent_id, stats in final_mode_stats.items():
        print(f"  {agent_id}:")
        print(f"    Current mode: {stats['current_mode']}")
        print(f"    Total switches: {stats['switch_count']}")
    
    system_summary = ctl.get_system_mode_summary()
    print(f"\nSystem-level mode statistics:")
    print(f"  Current status: {system_summary['system_status']}")
    print(f"  Total switches: {system_summary['total_switches']}")

print(f"\nTraining complete - Total time: {training_time:.2f}s")
print(f"Best epoch: {best_epoch} - Best loss: {best_loss / t_end:.4f}")

# Save training data - formation-related data removed
print("Saving training results...")
best_trajectories_data = {
    "formation_type": formation_type,
    "n_agents": n_agents,
    "epoch": best_epoch,
    "loss": best_loss,
    "x_logs": np.array([traj["states"].cpu().detach().numpy() for traj in best_epoch_trajectories]),
    "u_logs": np.array([traj["controls"].cpu().detach().numpy() for traj in best_epoch_trajectories]),
    "initial_states": np.array([traj["initial_states"].cpu().detach().numpy() for traj in best_epoch_trajectories]),
    "xbar": xbar.cpu().numpy(),
    "adversarial_training": ADVERSARIAL_TRAINING,
    "mode_switching_enabled": MODE_SWITCHING_ENABLED,
    "cylinder_only": True,
    "formation_weights_removed": True,  # Flag: formation weights removed
    "cylinder_obstacles": [{'x': cyl['x'], 'y': cyl['y'], 'z_min': cyl['z_min'], 'z_max': cyl['z_max'], 'radius': cyl['radius']}
                          for cyl in CYLINDER_OBSTACLES],
    "attack_epochs": attack_epochs,
    "attack_types_used": attack_types_used,
    "attack_combinations_used": attack_combinations_used,
    "scenario_type_history": scenario_type_history,
    "difficulty_first_strategy": True
}
best_filename = f'results/best_epoch_trajectories_{formation_type}_{n_agents}agents_obstacle_only.mat'
sio.savemat(best_filename, best_trajectories_data)

# Save loss history - formation loss removed
loss_history = {
    "formation_type": formation_type,
    "n_agents": n_agents,
    "total_loss": lossl,
    "state_loss": lossxl,
    "control_loss": lossul,
    "collision_loss": losscal,
    "side_loss": losssidel,
    "velocity_loss": lossvel,
    "cylinder_obstacle_loss": losscyll,
    "epochs": epochs,
    "best_epoch": best_epoch,
    "training_time": training_time,
    "distributed_mode": True,
    "adversarial_training": ADVERSARIAL_TRAINING,
    "mode_switching_enabled": MODE_SWITCHING_ENABLED,
    "cylinder_only": True,
    "formation_weights_removed": True,  # Flag: formation weights removed
    "basic_training_ratio": BASIC_TRAINING_RATIO,
    "attack_epochs": attack_epochs,
    "attack_types_used": attack_types_used,
    "attack_combinations_used": attack_combinations_used,
    "scenario_statistics": final_stats if ADVERSARIAL_TRAINING else {},
    "difficulty_first_strategy": True
}
loss_filename = f'results/loss_history_{formation_type}_{n_agents}agents_obstacle_only.mat'
sio.savemat(loss_filename, loss_history)

if MODE_SWITCHING_ENABLED:
    mode_switch_history_data = {
        "formation_type": formation_type,
        "n_agents": n_agents,
        "mode_switch_stats_per_epoch": mode_switch_stats_per_epoch,
        "final_mode_statistics": final_mode_stats,
        "system_summary": system_summary
    }
    mode_history_filename = f'results/mode_switch_history_{formation_type}_{n_agents}agents.mat'
    sio.savemat(mode_history_filename, mode_switch_history_data)
    print(f"Mode switching history saved: {mode_history_filename}")

# ===== Baseline Performance Test =====
print(f"\n--------- Baseline Performance Test ---------")
sys.reset_communication_phase()
ctl.reset_all_states()
sys.reset_communication_stats()

distributed_states = [state.clone() for state in initial_distributed_states]
x_log = torch.zeros((t_end, n_agents * 6), device=device)
u_log = torch.zeros((t_end, n_agents * 3), device=device)

test_mode_changes = []

test_start_time = time.time()
for t in range(t_end):
    communication_data = sys.distributed_communication_update(distributed_states, t, "testing")
    
    if MODE_SWITCHING_ENABLED:
        comm_quality_scores = sys.evaluate_communication_quality(communication_data, t)
    else:
        comm_quality_scores = None
    
    distributed_controls = ctl(distributed_states, communication_data, comm_quality_scores)
    
    if MODE_SWITCHING_ENABLED and t % 20 == 0:
        sys.update_agent_modes(ctl)
        test_mode_changes.append({
            't': t,
            'modes': sys.agent_modes.copy(),
            'comm_quality': comm_quality_scores.copy() if comm_quality_scores else []
        })
    
    distributed_states = sys(t, distributed_states, distributed_controls)
    
    global_state = sys.convert_to_global_state(distributed_states)
    global_control = torch.cat(distributed_controls)
    x_log[t, :] = global_state.detach()
    u_log[t, :] = global_control.detach()

test_time = time.time() - test_start_time
print(f"Baseline test time: {test_time:.2f}s")

n_coll = calculate_collisions(x_log, sys, min_dist)
print(f"Number of collisions: {n_coll}")

final_distances = []
for i in range(n_agents):
    pos_idx = i * 6
    final_pos = x_log[-1, pos_idx:pos_idx+3]
    target_pos = xbar[pos_idx:pos_idx+3]
    dist = torch.sqrt(torch.sum((final_pos - target_pos) ** 2))
    final_distances.append(dist.item())
    print(f"UAV {i+1} final distance to target: {dist.item():.4f}m")

avg_final_distance = sum(final_distances) / len(final_distances)
print(f"Average final distance to target: {avg_final_distance:.4f}m")

if MODE_SWITCHING_ENABLED and test_mode_changes:
    emergency_steps = sum(1 for record in test_mode_changes 
                         if 'emergency' in record['modes'])
    total_steps = len(test_mode_changes)
    print(f"\nTest mode statistics:")
    print(f"  Steps with emergency mode: {emergency_steps}/{total_steps}")

# Save baseline test data
test_data = {
    "formation_type": formation_type,
    "n_agents": n_agents,
    "x_log": x_log.cpu().detach().numpy(),
    "u_log": u_log.cpu().detach().numpy(),
    "xbar": xbar.cpu().detach().numpy(),
    "t_end": t_end,
    "min_dist": min_dist,
    "n_collisions": n_coll,
    "avg_final_distance": avg_final_distance,
    "test_time": test_time,
    "distributed_mode": True,
    "adversarial_training": ADVERSARIAL_TRAINING,
    "mode_switching_enabled": MODE_SWITCHING_ENABLED,
    "cylinder_only": True,
    "formation_weights_removed": True,  # Flag: formation weights removed
    "difficulty_first_strategy": True,
    "test_mode_changes": test_mode_changes if MODE_SWITCHING_ENABLED else []
}
test_filename = f'results/test_trajectory_{formation_type}_{n_agents}agents_obstacle_only.mat'
sio.savemat(test_filename, test_data)

# Generate final report
print(f"\n--------- Obstacle-Focus Training Summary Report ---------")
print(f"Formation type: {formation_type.upper()}")
print(f"Number of agents: {n_agents}")
print(f"Control mode: Fully distributed control")
print(f"Training mode: Difficulty-first adversarial training (obstacle-focus)")
print(f"Obstacle configuration:")
print(f"   Cylinder 1: x=-2.5, y=1, z_range=[-10, 10], radius=0.5")
print(f"   Cylinder 2: x=-2.5, y=-1, z_range=[-10, 10], radius=0.5")
print(f"   Cylinder 3: x=2.5, y=0, z_range=[-10, 10], radius=0.5")
print(f"Training epochs: {epochs}")
print(f"Total training time: {training_time:.2f}s")
print(f"Best epoch: {best_epoch}")
print(f"Best loss: {best_loss / t_end:.6f}")
print(f"Test collisions: {n_coll}")
print(f"Average arrival accuracy: {avg_final_distance:.4f}m")

print(f"\nTraining strategy notes:")
print(f"  1. All formation maintenance weight parameters removed")
print(f"  2. Focus on obstacle avoidance and target reaching")
print(f"  3. Cylinder obstacles: navigate around sides, avoid direct traversal")

saved_files = [best_filename, loss_filename, test_filename]
if MODE_SWITCHING_ENABLED:
    saved_files.append(mode_history_filename)

print(f"\nSaved files:")
for filename in saved_files:
    print(f"   {filename}")

print(f"\n--------- Training Complete ---------")
