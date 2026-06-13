import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_FILE = "results/runtime_log.csv"

def main():
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(RESULTS_FILE)

    # Plot 1: execution time per frame
    plt.figure()
    plt.plot(df["frame_id"], df["execution_time_ms"], marker="o", linewidth=1)
    plt.axhline(df["normal_budget_ms"].iloc[0], linestyle="--", label="Normal budget")
    plt.axhline(df["deadline_ms"].iloc[0], linestyle="--", label="Deadline")
    plt.xlabel("Frame ID")
    plt.ylabel("Execution Time (ms)")
    plt.title("Execution Time per Image Frame")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/runtime_per_frame.png", dpi=200)

    # Plot 2: average execution time by difficulty
    avg_by_difficulty = df.groupby("difficulty")["execution_time_ms"].mean()

    plt.figure()
    avg_by_difficulty.plot(kind="bar")
    plt.xlabel("Difficulty")
    plt.ylabel("Average Execution Time (ms)")
    plt.title("Average Execution Time by Difficulty")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("results/average_runtime_by_difficulty.png", dpi=200)

    # Plot 3: overruns by difficulty
    overruns = df.groupby("difficulty")["overrun"].sum()

    plt.figure()
    overruns.plot(kind="bar")
    plt.xlabel("Difficulty")
    plt.ylabel("Number of Overruns")
    plt.title("Overruns by Difficulty")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("results/overruns_by_difficulty.png", dpi=200)

    print("Saved plots:")
    print("results/runtime_per_frame.png")
    print("results/average_runtime_by_difficulty.png")
    print("results/overruns_by_difficulty.png")


if __name__ == "__main__":
    main()