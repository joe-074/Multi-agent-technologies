# Multi-Agent Technologies — SPbU (Master’s Program)

This repository contains Python implementations of coursework tasks for the **Multi-Agent Technologies** subject at SPbU.  
The focus is on distributed computation of the arithmetic mean in multi-agent systems under different communication models and uncertainties.

---

## Repository Structure

Planned / typical structure:

```text
.
├── README.md                  # Project description and usage
├── topology_utils.py          # Functions to create communication graphs (adjacency matrices)
├── centralized_baseline.py    # Centralized averaging (reference solution)
├── decentralized_mean_sim.py  # Module 4: decentralized averaging and communication-cost analysis
└── local_voting_sim.py        # Module 6: local voting with noise, delays, and switching links
