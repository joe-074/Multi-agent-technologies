import random

NUM_AGENTS = 10
MESSAGE_COST = 1000  # cost of one message to the fusion center
SEED = 0


def main():
    random.seed(SEED)

    # Each agent gets a random integer value
    values = [random.randint(-100, 100) for _ in range(NUM_AGENTS)]

    print("Agent values:")
    for i, v in enumerate(values):
        print(f"  Agent {i}: {v}")

    # Fusion center receives all values (NUM_AGENTS messages)
    true_mean = sum(values) / NUM_AGENTS
    total_cost = NUM_AGENTS * MESSAGE_COST

    print("\n=== Centralized Averaging Result ===")
    print(f"True arithmetic mean: {true_mean:.4f}")
    print(f"Total messages sent to fusion center: {NUM_AGENTS}")
    print(f"Communication cost: {total_cost}")

    print("\nThis result will be used to compare with decentralized algorithms.")


if __name__ == "__main__":
    main()