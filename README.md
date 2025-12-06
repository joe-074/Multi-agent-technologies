# Multi-Agent Technologies — SPbU (Master’s Program)

This repository contains Python implementations of coursework tasks for the Multi-Agent Technologies subject at SPbU.
The project focuses on distributed computation of the arithmetic mean in multi-agent systems under different communication models and uncertainties.


# Module 4 — Decentralized Averaging

Flooding-based algorithm where agents exchange values with neighbors until one agent collects all values and computes the mean.

Reports:

estimated mean

error vs. true mean

number of iterations

message counts

communication cost


# Module 6 — Local Voting Protocol

Consensus algorithm with noisy measurements, random delays, and switching communication links.

Reports:

final agent states

consensus accuracy

error vs. true mean

total communication cost
---

# Summary of Implemented Work
Centralized Baseline

A reference model where all agents send their values to a fusion center that computes the global mean.


## Repository Structure

Planned / typical structure:

```text
.
├── README.md                  # Project description and usage
├── topology_utils.py          # Functions to create communication graphs (adjacency matrices)
├── centralized_baseline.py    # Centralized averaging (reference solution)
├── decentralized_mean_sim.py  # Module 4: decentralized averaging and communication-cost analysis
└── local_voting_sim.py        # Module 6: local voting with noise, delays, and switching links

