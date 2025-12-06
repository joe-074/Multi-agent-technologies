from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

from topology_utils import random_connected_topology, print_adjacency


# --- Parameters (you can tune for experiments) ---

NUM_AGENTS = 8
SEED = 2

TIME_STEPS = 100            # number of iterations
ALPHA = 0.1                 # step size in local voting protocol

MEAS_NOISE_STD = 0.5        # standard deviation of measurement noise
MAX_DELAY = 2               # maximum discrete delay (0..MAX_DELAY)
LINK_ACTIVE_PROB = 0.8      # probability that a link is active at a step

NEIGHBOR_MESSAGE_COST = 10
SUPERVISOR_MESSAGE_COST = 1000


def gauss_noise(std: float) -> float:
    """Zero-mean Gaussian noise."""
    return random.gauss(0.0, std)


@dataclass
class VotingAgent:
    """
    Agent that participates in a local voting consensus protocol.
    Stores its own state x and a history to simulate delays.
    """

    idx: int
    x: float
    neighbors: List["VotingAgent"] = field(default_factory=list)

    history: List[float] = field(default_factory=list)
    spent: int = 0  # accumulated cost for neighbor communications

    def __post_init__(self) -> None:
        self.history = [self.x]

    def get_delayed_state(self, other: "VotingAgent") -> float:
        """Return delayed state of another agent using its history."""
        d = random.randint(0, MAX_DELAY)
        hist = other.history
        # index from the end, clamp at 0
        idx = max(0, len(hist) - 1 - d)
        return hist[idx]

    def step(self) -> tuple[int, int]:
        """
        Perform one local voting step.

        Returns:
            (messages_sent, communication_cost)
        """
        # Noisy self measurement
        y_ii = self.x + gauss_noise(MEAS_NOISE_STD)

        control = 0.0
        messages = 0
        cost = 0

        for nbr in self.neighbors:
            # switching link: may be inactive this time step
            if random.random() > LINK_ACTIVE_PROB:
                continue

            # delayed neighbor state + noise
            xj_delayed = self.get_delayed_state(nbr)
            y_ij = xj_delayed + gauss_noise(MEAS_NOISE_STD)

            control += (y_ij - y_ii)
            messages += 1
            cost += NEIGHBOR_MESSAGE_COST

        # local voting update
        u = ALPHA * control
        self.x = self.x + u
        self.history.append(self.x)
        self.spent += cost

        return messages, cost


def run_simulation() -> None:
    random.seed(SEED)

    # Initial values (for example, random integers)
    initial_values = [random.randint(-50, 50) for _ in range(NUM_AGENTS)]
    true_mean = sum(initial_values) / NUM_AGENTS

    print("Initial agent values:")
    for i, v in enumerate(initial_values):
        print(f"  Agent {i}: {v}")
    print(f"\nTrue mean (for reference): {true_mean:.4f}\n")

    # Base (time-invariant) connected topology. Links will switch on/off on top of this.
    base_adj = random_connected_topology(NUM_AGENTS, extra_edges=2, seed=SEED)
    print("Base adjacency matrix (connected graph):")
    print_adjacency(base_adj)
    print()

    # Create agents
    agents: List[VotingAgent] = [
        VotingAgent(idx=i, x=float(initial_values[i])) for i in range(NUM_AGENTS)
    ]

    # Set neighbor lists according to base adjacency
    for i, ag in enumerate(agents):
        ag.neighbors = [agents[j] for j, connected in enumerate(base_adj[i]) if connected]

    total_neighbor_messages = 0
    total_cost = 0

    # --- Time loop ---
    for t in range(TIME_STEPS):
        print(f"\n=== Time step {t + 1} ===")

        step_messages = 0
        step_cost = 0

        # Each agent performs local voting step
        for ag in agents:
            msgs, cost = ag.step()
            step_messages += msgs
            step_cost += cost

        total_neighbor_messages += step_messages
        total_cost += step_cost

        print(f"Neighbor messages this step: {step_messages}, cost: {step_cost}")

        # Print current states (optional, useful for debugging and understanding)
        for ag in agents:
            print(f"  Agent {ag.idx}: x = {ag.x:.4f}")

    # --- Results ---

    final_values = [ag.x for ag in agents]
    mean_final = sum(final_values) / NUM_AGENTS
    residual_mean_sq = (true_mean - mean_final) ** 2

    print("\n=== Simulation summary ===")
    print("Final agent states:")
    for i, x in enumerate(final_values):
        print(f"  Agent {i}: x_T = {x:.4f}")

    print(f"\nTrue mean: {true_mean:.4f}")
    print(f"Mean of final states: {mean_final:.4f}")
    print(f"Squared error between means: {residual_mean_sq:.6f}")

    # Measure how well agents agreed (consensus quality)
    max_dev = max(abs(x - mean_final) for x in final_values)
    print(f"Max deviation of agent from mean_final: {max_dev:.6f}")

    print(f"\nTotal neighbor messages: {total_neighbor_messages}")
    print(f"Total neighbor communication cost: {total_cost}")

    # If one final report to a supervisor is needed:
    total_cost_with_report = total_cost + SUPERVISOR_MESSAGE_COST
    print(f"Total cost including one final report: {total_cost_with_report}")

    print("\nPer-agent spent cost (neighbor communication only):")
    for ag in agents:
        print(f"  Agent {ag.idx}: cost = {ag.spent}")


if __name__ == "__main__":
    run_simulation()
