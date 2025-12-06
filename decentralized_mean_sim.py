"""
decentralized_mean_sim.py

Module 4: Decentralized computation of the arithmetic mean in a multi-agent system.

Each agent initially knows only its own scalar value. Agents exchange information
only with neighbors according to a fixed communication topology (adjacency matrix).
Information spreads (flooding) through the network until at least one agent knows
all values and can compute the global mean.

We record:
- true mean (for reference only)
- estimated mean from the decentralized algorithm
- residual error
- number of iterations
- number of inter-agent messages
- number of messages to the fusion center
- total communication cost
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

from topology_utils import line_topology, print_adjacency


NUM_AGENTS = 8
SEED = 1

# Communication cost model (can be adjusted for experiments)
NEIGHBOR_MESSAGE_COST = 10      # cost of one message between neighbors
SUPERVISOR_MESSAGE_COST = 1000  # cost of one message to the fusion center

MAX_ITERATIONS = 1000  # safety limit


@dataclass
class Agent:
    """Agent that gradually learns all values via local communication."""

    idx: int
    value: float
    neighbors: List["Agent"] = field(default_factory=list)

    knowledge: Dict[int, float] = field(default_factory=dict)
    inbox: List[Dict[int, float]] = field(default_factory=list)
    active: bool = True  # becomes False after silence
    has_reported: bool = False

    def __post_init__(self) -> None:
        # Initially, the agent knows only its own value.
        self.knowledge = {self.idx: self.value}

    def send_to_neighbors(self) -> int:
        """Send current knowledge to all neighbors. Returns communication cost."""
        if not self.active:
            return 0

        message = dict(self.knowledge)  # snapshot
        cost = 0
        for nbr in self.neighbors:
            nbr.inbox.append(message)
            cost += NEIGHBOR_MESSAGE_COST
        return cost

    def process_inbox(self) -> bool:
        """
        Merge all messages from inbox into knowledge.
        Returns True if any new information was learned.
        """
        if not self.active:
            self.inbox.clear()
            return False

        updated = False

        while self.inbox:
            msg = self.inbox.pop()
            for k, v in msg.items():
                if k not in self.knowledge:
                    self.knowledge[k] = v
                    updated = True

        return updated

    def knows_everyone(self, n: int) -> bool:
        """Check if this agent has values for all n agents."""
        return len(self.knowledge) == n


def run_simulation() -> None:
    random.seed(SEED)

    # Assign initial values
    initial_values = [random.randint(-50, 50) for _ in range(NUM_AGENTS)]
    true_mean = sum(initial_values) / NUM_AGENTS

    print("Initial agent values:")
    for i, v in enumerate(initial_values):
        print(f"  Agent {i}: {v}")
    print(f"\nTrue mean (for reference): {true_mean:.4f}\n")

    # Communication topology: here we use a simple line, but this can be changed
    adj = line_topology(NUM_AGENTS)
    print("Adjacency matrix (line topology):")
    print_adjacency(adj)
    print()

    # Create agents
    agents: List[Agent] = [Agent(idx=i, value=float(initial_values[i])) for i in range(NUM_AGENTS)]

    # Set neighbors according to adjacency matrix
    for i, agent in enumerate(agents):
        agent.neighbors = [agents[j] for j, connected in enumerate(adj[i]) if connected]

    total_neighbor_messages = 0
    total_supervisor_messages = 0
    total_cost = 0
    iterations = 0

    reporter_idx = None
    estimated_mean = None

    # Main synchronous loop
    while iterations < MAX_ITERATIONS:
        iterations += 1
        print(f"\n=== Iteration {iterations} ===")

        # Phase 1: send
        step_cost = 0
        step_messages = 0
        for ag in agents:
            cost = ag.send_to_neighbors()
            step_cost += cost
            # each neighbor message has fixed cost; count messages explicitly
            if ag.active:
                # outgoing messages = number of neighbors
                step_messages += len(ag.neighbors)

        total_neighbor_messages += step_messages
        total_cost += step_cost

        print(f"Neighbor messages this iteration: {step_messages}, cost: {step_cost}")

        # Phase 2: receive/process
        any_updates = False
        for ag in agents:
            updated = ag.process_inbox()
            any_updates = any_updates or updated

        # Check if some agent now knows everyone
        if reporter_idx is None:
            for ag in agents:
                if ag.knows_everyone(NUM_AGENTS):
                    reporter_idx = ag.idx
                    estimated_mean = sum(ag.knowledge.values()) / NUM_AGENTS
                    print(
                        f"\nAgent {ag.idx} has collected all values and will report "
                        f"the mean to the fusion center."
                    )
                    total_supervisor_messages += 1
                    total_cost += SUPERVISOR_MESSAGE_COST
                    ag.has_reported = True
                    break

        # If mean already reported, we can stop (everyone can be silenced)
        if reporter_idx is not None:
            print("Silencing all agents after reporting.")
            for ag in agents:
                ag.active = False
            break

        # If no updates in this iteration, but nobody knows everyone -> stuck
        if not any_updates:
            print("No updates occurred and no agent knows all values. Stopping (disconnected graph?).")
            break

    # Summary
    print("\n=== Simulation summary ===")
    print(f"Iterations performed: {iterations}")
    if reporter_idx is not None and estimated_mean is not None:
        print(f"Reporting agent: {reporter_idx}")
        print(f"Estimated mean: {estimated_mean:.4f}")
        residual = (true_mean - estimated_mean) ** 2
        print(f"Residual error (squared difference): {residual:.6f}")
    else:
        print("No agent managed to collect all values. Estimated mean is undefined.")
        estimated_mean = None

    print(f"\nNeighbor messages total: {total_neighbor_messages}")
    print(f"Messages to fusion center: {total_supervisor_messages}")
    print(f"Total communication cost: {total_cost}")

    # Optional: show how much each agent learned (for report / analysis)
    print("\nKnowledge sizes per agent:")
    for ag in agents:
        print(f"  Agent {ag.idx}: knows {len(ag.knowledge)} values")


if __name__ == "__main__":
    run_simulation()