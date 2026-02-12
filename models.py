#!/usr/bin/env python
"""
Adaptive Switching Control Network (ASCN) for Distributed Multi-UAV Formation Control

This module implements the core ASCN framework proposed in:
    "Distributed Formation Control Driven by Recursive Equilibrium Networks 
     Under Communication Attacks"
    T. Luan, F. Huang, J. Li, Y. Yang
    Knowledge-Based Systems, 2026.

Key innovations:
    1. Recursive Balanced Network (RBN) with acyclic feedforward structure,
       reducing computational complexity from O(q^3) to O(lq^2). (Section 3.1)
    2. Communication quality-aware adaptive dual-mode switching controller
       with hysteresis thresholds for anti-jitter switching. (Section 2.2)
    3. Distributed control architecture where each agent makes independent
       mode decisions based on local communication quality assessment. (Section 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ==============================================================================
# Recursive Balanced Network (RBN) - Acyclic Feedforward Structure
# Paper Section 3.1, Equations (17)-(21)
# ==============================================================================

class RBNRG(nn.Module):
    """Recursive Balanced Network with acyclic feedforward structure.
    
    Implements the non-cyclic feedforward recursive balanced network mapping 
    function f_theta described in Eq. (17)-(18). The network guarantees 
    contraction mapping properties (Eq. 24-25) through constrained weight 
    parameterization, enabling strict exponential stability analysis.
    
    Computational complexity: O(L * m^2) per forward pass (Eq. 19),
    significantly lower than O(m^3) for traditional fully connected networks.
    
    Args:
        n: Input dimension (state difference dimension).
        m: Output dimension (control dimension).
        n_xi: Internal state dimension of the recursive network.
        l: Number of nonlinear activation functions (network depth).
        device: Computation device.
    """
    
    def __init__(self, n, m, n_xi, l, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.n = n      # input dimension (state dimension)
        self.n_xi = n_xi  # internal state dimension
        self.l = l      # number of nonlinear activation functions
        self.m = m      # output dimension (control dimension)

        # Trainable parameters for constrained weight construction
        std = 1
        self.X = nn.Parameter((torch.randn(2 * n_xi + l, 2 * n_xi + l, device=device) * std))
        self.Y = nn.Parameter((torch.randn(n_xi, n_xi, device=device) * std))
        # NN state dynamics:
        self.B2 = nn.Parameter((torch.randn(n_xi, n, device=device) * std))
        # NN output:
        self.C2 = nn.Parameter((torch.randn(m, n_xi, device=device) * std))
        self.D21 = nn.Parameter((torch.randn(m, l, device=device) * std))
        if m >= n:
            self.Z3 = nn.Parameter(torch.randn(m - n, n, device=device) * std)
            self.X3 = nn.Parameter(torch.randn(n, n, device=device) * std)
            self.Y3 = nn.Parameter(torch.randn(n, n, device=device) * std)
        else:
            self.Z3 = nn.Parameter(torch.randn(n - m, m, device=device) * std)
            self.X3 = nn.Parameter(torch.randn(m, m, device=device) * std)
            self.Y3 = nn.Parameter(torch.randn(m, m, device=device) * std)
        # v signal:
        self.D12 = nn.Parameter((torch.randn(l, n, device=device) * std))
        # bias:
        self.bxi = nn.Parameter(torch.randn(n_xi, device=device))
        self.bv = nn.Parameter(torch.randn(l, device=device))
        self.bu = nn.Parameter(torch.randn(m, device=device))
        
        # Non-trainable parameters (derived from trainable ones in forward pass)
        self.epsilon = 0.001
        self.F = torch.zeros(n_xi, n_xi, device=device)
        self.B1 = torch.zeros(n_xi, l, device=device)
        self.E = torch.zeros(n_xi, n_xi, device=device)
        self.Lambda = torch.ones(l, device=device)
        self.C1 = torch.zeros(l, n_xi, device=device)
        self.D11 = torch.zeros(l, l, device=device)
        self.Lq = torch.zeros(m, m, device=device)
        self.Lr = torch.zeros(n, n, device=device)
        self.D22 = torch.zeros(m, n, device=device)

    def forward(self, t, w, xi, gammap, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """Forward pass of the Recursive Balanced Network.
        
        Implements the L-layer acyclic feedforward computation (Eq. 18):
            z^(l) = sigma_a(W^(l) * z^(l-1) + b^(l))
        where z^(0) = x_i - x_j and z^(L) = f_theta(x_i - x_j).
        
        The weight matrices are constructed from trainable parameters to 
        guarantee the contraction mapping property: ||f(x)-f(y)|| <= alpha*||x-y||
        with alpha = rho^L < 1 (Eq. 25).
        
        Args:
            t: Current time step.
            w: Input state vector (state difference x_i - x_j).
            xi: Internal state of the recursive network.
            gammap: Mode-dependent control gain (gamma).
            device: Computation device.
            
        Returns:
            u: Control output vector u_i (Eq. 17).
            xi_: Updated internal state.
        """
        # --- Constrained parameter construction ---
        n_xi = self.n_xi
        l = self.l
        n = self.n
        m = self.m
        R = gammap * torch.eye(n, n, device=device)
        Q = (-1 / gammap) * torch.eye(m, m, device=device)
        M = F.linear(self.X3.T, self.X3.T) + self.Y3 - self.Y3.T + F.linear(self.Z3.T,
                                                                            self.Z3.T) + self.epsilon * torch.eye(min(n, m), device=device)
        if m >= n:
            N = torch.vstack((F.linear(torch.eye(min(n, m), device=device) - M,
                                       torch.inverse(torch.eye(min(n, m), device=device) + M).T),
                              -2 * F.linear(self.Z3, torch.inverse(torch.eye(min(n, m), device=device) + M).T)))
        else:
            N = torch.hstack((F.linear(torch.inverse(torch.eye(min(n, m), device=device) + M),
                                       (torch.eye(min(n, m), device=device) - M).T),
                              -2 * F.linear(torch.inverse(torch.eye(min(n, m), device=device) + M), self.Z3)))

        self.D22 = gammap * N
        R_capital = R - (1 / gammap) * F.linear(self.D22.T, self.D22.T)
        C2_capital = torch.matmul(torch.matmul(self.D22.T, Q), self.C2)
        D21_capital = torch.matmul(torch.matmul(self.D22.T, Q), self.D21) - self.D12.T
        vec_R = torch.cat([C2_capital.T, D21_capital.T, self.B2], 0)
        vec_Q = torch.cat([self.C2.T, self.D21.T, torch.zeros(n_xi, m, device=device)], 0)
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * n_xi + l, device=device) + torch.matmul(
            torch.matmul(vec_R, torch.inverse(R_capital)), vec_R.T) - torch.matmul(
            torch.matmul(vec_Q, Q), vec_Q.T)
        h1, h2, h3 = torch.split(H, (n_xi, l, n_xi), dim=0)
        H11, H12, H13 = torch.split(h1, (n_xi, l, n_xi), dim=1)
        H21, H22, _ = torch.split(h2, (n_xi, l, n_xi), dim=1)
        H31, H32, H33 = torch.split(h3, (n_xi, l, n_xi), dim=1)
        P = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + P + self.Y - self.Y.T)
        # v signal:
        self.Lambda = torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21
        
        # --- Acyclic feedforward computation (Eq. 18) ---
        # Sequential layer-by-layer evaluation with ReLU activation
        vec = torch.zeros(self.l, device=device)
        vec[0] = 1
        epsilon = torch.zeros(self.l, device=device)
        v = F.linear(xi, self.C1[0, :]) + F.linear(w, self.D12[0, :]) + self.bv[0]
        epsilon = epsilon + vec * torch.relu(v / self.Lambda[0])
        for i in range(1, self.l):
            vec = torch.zeros(self.l, device=device)
            vec[i] = 1
            v = F.linear(xi, self.C1[i, :]) + F.linear(epsilon, self.D11[i, :]) + F.linear(w, self.D12[i, :]) + self.bv[i]
            epsilon = epsilon + vec * torch.relu(v / self.Lambda[i])
        
        # Internal state update and control output
        E_xi_ = F.linear(xi, self.F) + F.linear(epsilon, self.B1) + F.linear(w, self.B2) + self.bxi
        xi_ = F.linear(E_xi_, self.E.inverse())
        u = F.linear(xi, self.C2) + F.linear(epsilon, self.D21) + F.linear(w, self.D22) + self.bu
        return u, xi_


# ==============================================================================
# Single Agent ASCN Controller - Dual-Mode Adaptive Switching
# Paper Section 2.2 (Switching Mechanism) + Section 3.1 (Controller Architecture)
# ==============================================================================

class SingleAgentController(nn.Module):
    """ASCN controller for a single UAV agent with adaptive dual-mode switching.
    
    Implements the communication quality-aware adaptive mode switching mechanism 
    (Eq. 12) with hysteresis thresholds to prevent frequent oscillations.
    
    Dual-mode architecture:
        - Normal mode (sigma=1): Cooperative control using neighbor information
          via full-scale RBN network with n_xi neurons. (Eq. 17)
        - Emergency mode (sigma=2): Local state-only control via reduced-scale
          RBN network with n_xi/2 neurons, achieving 75% complexity reduction. (Eq. 21)
    
    The mode switching is governed by the composite quality indicator Q(t) (Eq. 8)
    integrating channel quality Q_ch (60%) and topology quality Q_top (40%).
    
    Args:
        agent_id: Unique identifier for this agent.
        n_agents: Total number of agents in the formation.
        state_dim: State dimension per agent (default: 6 for [x,y,z,vx,vy,vz]).
        control_dim: Control dimension per agent (default: 3 for [ux,uy,uz]).
        n_xi: Internal state dimension for the normal mode RBN (default: 30).
        l: Number of nonlinear activation functions (default: 30).
        max_neighbors: Maximum number of communication neighbors (default: 3).
        device: Computation device.
    """
    
    def __init__(self, agent_id, n_agents, state_dim=6, control_dim=3, 
                 n_xi=30, l=30, max_neighbors=3, device=torch.device('cpu')):
        super().__init__()
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.n_xi = n_xi
        self.max_neighbors = min(max_neighbors, n_agents - 1)
        self.device = device
        
        # === Mode state management ===
        self.current_mode = 'normal'  # 'normal' or 'emergency'
        self.mode_switch_count = 0
        self.mode_switch_history = []
        
        # Input dimension: local state + neighbor communication data
        comm_dim = self.max_neighbors * state_dim
        input_dim = state_dim + comm_dim
        
        # === Normal mode controller (sigma=1) ===
        # Full-scale RBN: complexity O(l * n_xi^2)  (Eq. 20)
        self.local_rbn = RBNRG(input_dim, control_dim, n_xi, l, device)
        
        # === Emergency mode controller (sigma=2) ===
        # Reduced-scale RBN: complexity O(l * (n_xi/2)^2), 75% reduction (Eq. 21)
        self.emergency_rbn = RBNRG(state_dim, control_dim, n_xi // 2, l // 2, device)
        
        # Internal state management
        self._xi_current = None
        self._xi_emergency = None
        self._initialize_state()
        
        # === Communication quality tracking ===
        self.comm_quality_history = []
        self.mode_duration = {'normal': 0, 'emergency': 0}
        
    def _initialize_state(self):
        """Initialize internal states for both normal and emergency mode RBNs."""
        self._xi_current = torch.zeros(self.n_xi, device=self.device, requires_grad=False).detach()
        self._xi_emergency = torch.zeros(self.n_xi // 2, device=self.device, requires_grad=False).detach()
        
    def forward(self, local_state: torch.Tensor, 
                communication_data: Dict[int, torch.Tensor], 
                comm_quality: float = 1.0) -> torch.Tensor:
        """Forward pass with adaptive mode selection based on communication quality.
        
        Implements the ASCN control law (Eq. 17):
            u_i = K_{sigma(t)} * sum_{j in N_i} a_ij * f_theta(x_i - x_j)
        where sigma(t) is determined by the hysteresis switching rule (Eq. 12).
        
        Args:
            local_state: Agent's own state vector [x,y,z,vx,vy,vz], shape [state_dim].
            communication_data: Received neighbor states {neighbor_id: state_tensor}.
                May be incomplete or empty under DoS attacks.
            comm_quality: Communication quality score Q(t) in [0,1] (Eq. 8).
            
        Returns:
            Control input vector [ux,uy,uz], shape [control_dim].
        """
        # === Record communication quality ===
        self.comm_quality_history.append(comm_quality)
        if len(self.comm_quality_history) > 100:
            self.comm_quality_history.pop(0)
        
        # === Determine control mode based on quality assessment (Eq. 12) ===
        new_mode = self._determine_mode(comm_quality, communication_data)
        
        # === Mode switch detection and logging ===
        if new_mode != self.current_mode:
            self.mode_switch_count += 1
            self.mode_switch_history.append({
                'agent_id': self.agent_id,
                'switch_count': self.mode_switch_count,
                'from_mode': self.current_mode,
                'to_mode': new_mode,
                'comm_quality': comm_quality,
                'neighbor_count': len(communication_data) if communication_data else 0
            })
            self.current_mode = new_mode
        
        # === Update mode duration counter ===
        self.mode_duration[self.current_mode] += 1
        
        # === Select control strategy based on current mode ===
        if self.current_mode == 'emergency':
            control = self._emergency_mode_control(local_state)
        else:
            control = self._normal_mode_control(local_state, communication_data)
        
        return control
    
    def _determine_mode(self, comm_quality: float, 
                        communication_data: Dict[int, torch.Tensor]) -> str:
        """Determine the control mode using hysteresis switching rule.
        
        Implements the switching rule with anti-jitter characteristics (Eq. 12):
            sigma(t) = 2 (emergency),  if Q(t) < T_h^low
            sigma(t) = 1 (normal),     if Q(t) > T_h^high
            sigma(t) = sigma(t-),      otherwise (hysteresis band)
        
        The hysteresis bandwidth Delta_T_h = T_h^high - T_h^low = 0.3 ensures
        a minimum switching interval T_min (Eq. 13) and guarantees the average
        dwell time condition tau_a >= Delta_T_h / (2 * Q_dot_max) (Eq. 16).
        
        The composite quality indicator integrates (Eq. 8):
            Q(t) = w1 * Q_ch(t) + w2 * Q_top(t)  with w1=0.6, w2=0.4
        
        Args:
            comm_quality: Channel quality score Q_ch in [0,1].
            communication_data: Received neighbor data for topology assessment.
            
        Returns:
            Mode string: 'normal' or 'emergency'.
        """
        # Hysteresis thresholds (Eq. 12)
        EMERGENCY_THRESHOLD = 0.4   # T_h^low: switch to emergency below this
        RECOVERY_THRESHOLD = 0.7    # T_h^high: recover to normal above this
        NEIGHBOR_EMERGENCY_THRESHOLD = 0.3  # Neighbor availability emergency threshold
        NEIGHBOR_RECOVERY_THRESHOLD = 0.6   # Neighbor availability recovery threshold
        
        # Compute effective neighbor ratio as proxy for topology quality Q_top (Eq. 11)
        effective_neighbors = len(communication_data) if communication_data else 0
        expected_neighbors = min(self.max_neighbors, self.n_agents - 1)
        neighbor_ratio = effective_neighbors / expected_neighbors if expected_neighbors > 0 else 0
        
        # Composite quality indicator Q(t) (Eq. 8): w1=0.6 (channel), w2=0.4 (topology)
        overall_quality = 0.6 * comm_quality + 0.4 * neighbor_ratio
        
        # Hysteresis switching logic (Eq. 12)
        if self.current_mode == 'normal':
            # Normal -> Emergency: quality drops below T_h^low
            if overall_quality < EMERGENCY_THRESHOLD or neighbor_ratio < NEIGHBOR_EMERGENCY_THRESHOLD:
                return 'emergency'
        else:  # emergency mode
            # Emergency -> Normal: quality recovers above T_h^high
            if overall_quality > RECOVERY_THRESHOLD and neighbor_ratio > NEIGHBOR_RECOVERY_THRESHOLD:
                return 'normal'
        
        # Stay in current mode (hysteresis band)
        return self.current_mode
    
    def _normal_mode_control(self, local_state: torch.Tensor, 
                             communication_data: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Normal mode control (sigma=1): cooperative control using neighbor information.
        
        Utilizes full-scale RBN network with complete neighbor state information
        for high-precision formation tracking. Complexity: O(l * n_xi^2) (Eq. 20).
        
        Args:
            local_state: Agent's own state vector, shape [state_dim].
            communication_data: Received neighbor states from communication network.
            
        Returns:
            Control output vector, shape [control_dim].
        """
        # Aggregate neighbor information into fixed-dimension vector
        neighbor_info = self._process_communication_data(communication_data)
        
        # Concatenate local state and neighbor information as network input
        full_input = torch.cat([local_state, neighbor_info])
        
        # Create detached copy of internal state for forward pass
        xi_input = torch.zeros_like(self._xi_current, device=self.device, requires_grad=False)
        xi_input.data.copy_(self._xi_current.data)
        
        # Full-scale RBN computation with normal gain
        gamma = 1.0  # Normal mode gain K_{sigma=1}
        control, xi_new = self.local_rbn(0, full_input, xi_input, gamma)
        
        # Update internal state (detached from computation graph)
        with torch.no_grad():
            new_state = torch.zeros_like(self._xi_current, device=self.device, requires_grad=False)
            new_state.data.copy_(xi_new.detach().data)
            self._xi_current = new_state
        
        return control
    
    def _emergency_mode_control(self, local_state: torch.Tensor) -> torch.Tensor:
        """Emergency mode control (sigma=2): local state-only robust control.
        
        Uses reduced-scale RBN network (n_xi/2 neurons, l/2 layers) relying 
        only on local state when communication is degraded. Achieves 75% 
        complexity reduction compared to normal mode (Eq. 21):
            eta = 1 - (m2/m1)^2 = 1 - (1/2)^2 = 75%
        
        Applies conservative gain and control amplitude limiting to ensure
        stability under communication interruption scenarios.
        
        Args:
            local_state: Agent's own state vector, shape [state_dim].
            
        Returns:
            Control output vector (amplitude-limited), shape [control_dim].
        """
        # Use only local state (no neighbor information needed)
        xi_input = torch.zeros_like(self._xi_emergency, device=self.device, requires_grad=False)
        xi_input.data.copy_(self._xi_emergency.data)
        
        # Reduced-scale RBN with conservative gain K_{sigma=2}
        gamma = 0.5  # Reduced gain for conservative control
        control, xi_new = self.emergency_rbn(0, local_state, xi_input, gamma)
        
        # Update emergency internal state
        with torch.no_grad():
            new_state = torch.zeros_like(self._xi_emergency, device=self.device, requires_grad=False)
            new_state.data.copy_(xi_new.detach().data)
            self._xi_emergency = new_state
        
        # === Limit control amplitude in emergency mode for safety ===
        max_control = 8.0
        control = torch.clamp(control, -max_control, max_control)
        
        return control
    
    def _process_communication_data(self, communication_data: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Process received communication data into a fixed-dimension neighbor vector.
        
        Constructs the neighbor information vector for the ASCN control law (Eq. 17).
        Under DoS attacks, missing neighbor data results in zero-padding, directly
        degrading control performance and triggering mode switching.
        
        Args:
            communication_data: Mapping from neighbor_id to state tensor.
                May be incomplete or empty due to DoS attacks.
            
        Returns:
            Fixed-dimension neighbor vector, shape [max_neighbors * state_dim].
            Missing neighbors are zero-padded.
        """
        neighbor_info = torch.zeros(self.max_neighbors * self.state_dim, device=self.device)
        
        if communication_data is None or len(communication_data) == 0:
            # No communication data available - direct consequence of DoS attack
            return neighbor_info
        
        # Extract neighbor states from communication data
        neighbor_items = list(communication_data.items())
        
        # Fill neighbor information slots
        for idx, (neighbor_id, neighbor_state) in enumerate(neighbor_items):
            if idx >= self.max_neighbors:
                break
            
            start_idx = idx * self.state_dim
            end_idx = start_idx + self.state_dim
            
            if isinstance(neighbor_state, torch.Tensor) and neighbor_state.numel() >= self.state_dim:
                neighbor_info[start_idx:end_idx] = neighbor_state[:self.state_dim]
            # Incomplete or corrupted data remains zero-padded
        
        return neighbor_info
    
    def reset_state(self):
        """Reset all internal states and mode tracking."""
        del self._xi_current
        del self._xi_emergency
        self._initialize_state()
        self.current_mode = 'normal'
        self.mode_switch_count = 0
        self.comm_quality_history.clear()
        self.mode_switch_history.clear()
        self.mode_duration = {'normal': 0, 'emergency': 0}
    
    def get_mode_statistics(self) -> dict:
        """Get mode switching statistics for this agent.
        
        Returns:
            Dictionary containing current mode, switch count, quality history,
            and time distribution across modes.
        """
        avg_quality = sum(self.comm_quality_history) / len(self.comm_quality_history) if self.comm_quality_history else 0.0
        total_duration = sum(self.mode_duration.values())
        
        return {
            'agent_id': self.agent_id,
            'current_mode': self.current_mode,
            'switch_count': self.mode_switch_count,
            'mode_history': self.mode_switch_history.copy(),
            'avg_comm_quality': avg_quality,
            'mode_duration': self.mode_duration.copy(),
            'normal_pct': (self.mode_duration['normal'] / total_duration * 100) if total_duration > 0 else 0,
            'emergency_pct': (self.mode_duration['emergency'] / total_duration * 100) if total_duration > 0 else 0
        }


# ==============================================================================
# Distributed ASCN Controller - Multi-Agent Coordination
# Paper Section 3 (Overall Architecture)
# ==============================================================================

class DistributedController(nn.Module):
    """Distributed ASCN controller for multi-UAV formation systems.
    
    Coordinates N independent SingleAgentControllers, where each agent makes
    autonomous control decisions based solely on:
        1. Its own local state x_i (always available)
        2. Neighbor states received via communication (may be degraded/lost)
        3. Local communication quality score Q_i(t)
    
    No agent has access to global state, ensuring strict distributed operation.
    Each agent independently determines its control mode (normal/emergency) 
    based on local communication quality assessment (Eq. 12).
    
    Args:
        n_agents: Number of UAV agents in the formation (N).
        max_neighbors: Maximum communication neighbors per agent.
        device: Computation device.
    """
    
    def __init__(self, n_agents, max_neighbors=3, device=torch.device('cpu')):
        super().__init__()
        self.n_agents = n_agents
        self.device = device
        
        # Create independent controller for each agent
        self.agents = nn.ModuleList([
            SingleAgentController(i, n_agents, max_neighbors=max_neighbors, device=device) 
            for i in range(n_agents)
        ])
        
        # Global control gain amplifier (trainable)
        self.amplifier = nn.Parameter(torch.ones(1, device=device))
    
    def forward(self, agent_states: List[torch.Tensor], 
                communication_data: List[Dict[int, torch.Tensor]], 
                comm_quality_scores: Optional[List[float]] = None) -> List[torch.Tensor]:
        """Distributed control computation - strictly no global state access.
        
        Each agent independently computes its control input using only locally
        available information, implementing the distributed control protocol (Eq. 17):
            u_i = K_{sigma(t)} * sum_{j in N_i} a_ij * f_theta(x_i - x_j)
        
        Args:
            agent_states: List of N local state tensors, each shape [6].
            communication_data: List of N dicts, each mapping neighbor_id to state.
                Under DoS attacks, dicts may be incomplete or empty.
            comm_quality_scores: List of N quality scores Q_i(t) in [0,1] (Eq. 8).
                Defaults to 1.0 (perfect communication) if not provided.
            
        Returns:
            List of N control input tensors, each shape [3].
        """
        if len(agent_states) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} agent states, got {len(agent_states)}")
        
        if len(communication_data) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} communication data, got {len(communication_data)}")
        
        # Default to perfect communication if quality scores not provided
        if comm_quality_scores is None:
            comm_quality_scores = [1.0] * self.n_agents
        
        distributed_controls = []
        
        for i in range(self.n_agents):
            # Each agent accesses ONLY:
            # 1. Its own state
            # 2. Communication-received data (may be incomplete/delayed/lost)
            # 3. Local communication quality score (for mode switching)
            local_state = agent_states[i]
            comm_info = communication_data[i]
            quality = comm_quality_scores[i]
            
            # Local autonomous control decision
            local_control = self.agents[i](local_state, comm_info, quality)
            distributed_controls.append(local_control * self.amplifier)
        
        return distributed_controls
    
    def reset_all_states(self):
        """Reset internal states of all agents."""
        for agent in self.agents:
            agent.reset_state()
    
    def get_mode_statistics(self) -> Dict[str, dict]:
        """Get mode switching statistics for all agents."""
        return {f'agent_{i}': agent.get_mode_statistics() for i, agent in enumerate(self.agents)}
    
    def get_system_mode_summary(self) -> dict:
        """Get system-level mode status summary.
        
        Returns:
            Dictionary with counts of normal/emergency agents, total switches,
            and overall system status ('normal', 'degraded', or 'critical').
        """
        mode_stats = self.get_mode_statistics()
        
        normal_count = sum(1 for s in mode_stats.values() if s['current_mode'] == 'normal')
        emergency_count = self.n_agents - normal_count
        total_switches = sum(s['switch_count'] for s in mode_stats.values())
        
        return {
            'normal_agents': normal_count,
            'emergency_agents': emergency_count,
            'total_switches': total_switches,
            'system_status': 'normal' if emergency_count == 0 
                            else 'degraded' if emergency_count < self.n_agents 
                            else 'critical'
        }
