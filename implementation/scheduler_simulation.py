import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "results/runtime_log.csv"
OUTPUT_FILE = "results/scheduler_comparison.csv"

# Fast control loop period used by normal/adaptive strategies.
FAST_PERIOD_MS = 2.0

# Normal expected budget.
NORMAL_BUDGET_MS = 0.5

# Safety factor added to measured WCET.
WCET_SAFETY_FACTOR = 1.10


def simulate_strategy(df, strategy_name, wcet_estimate_ms):
    rows = []

    if strategy_name == "WCET":
        # Proper WCET scheduling:
        # The task period/deadline must be large enough for the worst case.
        period_ms = wcet_estimate_ms * WCET_SAFETY_FACTOR
        reserved_budget_ms = wcet_estimate_ms

    else:
        # Normal and overrun-handled strategies use the faster control loop.
        period_ms = FAST_PERIOD_MS
        reserved_budget_ms = NORMAL_BUDGET_MS

    for _, row in df.iterrows():
        frame_id = int(row["frame_id"])
        difficulty = row["difficulty"]
        execution_time = float(row["execution_time_ms"])

        release_time = frame_id * period_ms
        start_time = release_time
        finish_time = start_time + execution_time

        original_deadline = release_time + period_ms
        adjusted_deadline = original_deadline

        overrun = execution_time > reserved_budget_ms

        if strategy_name == "OverrunHandled" and execution_time > NORMAL_BUDGET_MS:
            # CBS-inspired simplified deadline extension.
            bandwidth = NORMAL_BUDGET_MS / FAST_PERIOD_MS
            extra = execution_time - NORMAL_BUDGET_MS
            deadline_extension = extra / bandwidth
            adjusted_deadline = original_deadline + deadline_extension

        deadline_missed = finish_time > adjusted_deadline

        slack_ms = adjusted_deadline - finish_time

        rows.append({
            "strategy": strategy_name,
            "frame_id": frame_id,
            "difficulty": difficulty,
            "execution_time_ms": execution_time,
            "period_ms": period_ms,
            "release_time_ms": release_time,
            "start_time_ms": start_time,
            "finish_time_ms": finish_time,
            "original_deadline_ms": original_deadline,
            "adjusted_deadline_ms": adjusted_deadline,
            "reserved_budget_ms": reserved_budget_ms,
            "overrun": overrun,
            "deadline_missed": deadline_missed,
            "slack_ms": slack_ms,
        })

    return rows


def plot_deadline_misses(result_df):
    misses = result_df.groupby("strategy")["deadline_missed"].sum()

    plt.figure()
    misses.plot(kind="bar")
    plt.xlabel("Scheduling Strategy")
    plt.ylabel("Number of Deadline Misses")
    plt.title("Deadline Misses by Scheduling Strategy")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("results/deadline_misses_by_strategy.png", dpi=200)


def plot_effective_frequency(result_df):
    periods = result_df.groupby("strategy")["period_ms"].first()
    frequencies_hz = 1000.0 / periods

    plt.figure()
    frequencies_hz.plot(kind="bar")
    plt.xlabel("Scheduling Strategy")
    plt.ylabel("Effective Frequency (Hz)")
    plt.title("Effective Control Frequency by Strategy")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("results/effective_frequency_by_strategy.png", dpi=200)


def plot_slack(result_df):
    avg_slack = result_df.groupby("strategy")["slack_ms"].mean()

    plt.figure()
    avg_slack.plot(kind="bar")
    plt.xlabel("Scheduling Strategy")
    plt.ylabel("Average Slack Time (ms)")
    plt.title("Average Deadline Slack by Strategy")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("results/average_slack_by_strategy.png", dpi=200)


def plot_runtime_with_strategy_deadlines(result_df):
    for strategy in result_df["strategy"].unique():
        subset = result_df[result_df["strategy"] == strategy]

        plt.figure()
        plt.plot(
            subset["frame_id"],
            subset["execution_time_ms"],
            marker="o",
            linewidth=1,
            label="Execution time"
        )

        # Deadline relative to release time
        relative_deadline = (
            subset["adjusted_deadline_ms"] - subset["release_time_ms"]
        )

        plt.plot(
            subset["frame_id"],
            relative_deadline,
            linestyle="--",
            linewidth=1,
            label="Allowed deadline"
        )

        plt.axhline(
            subset["reserved_budget_ms"].iloc[0],
            linestyle=":",
            linewidth=1,
            label="Reserved budget"
        )

        plt.xlabel("Frame ID")
        plt.ylabel("Time (ms)")
        plt.title(f"Execution Time and Deadline: {strategy}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/runtime_deadline_{strategy}.png", dpi=200)


def plot_overruns_by_strategy(result_df):
    overruns = result_df.groupby("strategy")["overrun"].sum()

    plt.figure()
    overruns.plot(kind="bar")
    plt.xlabel("Scheduling Strategy")
    plt.ylabel("Number of Overruns")
    plt.title("Detected Overruns by Scheduling Strategy")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("results/overruns_by_strategy.png", dpi=200)


def main():
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    wcet_estimate_ms = df["execution_time_ms"].max()

    all_rows = []
    for strategy in ["WCET", "NormalOnly", "OverrunHandled"]:
        all_rows.extend(simulate_strategy(df, strategy, wcet_estimate_ms))

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Measured WCET estimate: {wcet_estimate_ms:.4f} ms")
    print(f"WCET period with safety factor: {wcet_estimate_ms * WCET_SAFETY_FACTOR:.4f} ms")
    print(f"Fast normal/adaptive period: {FAST_PERIOD_MS:.4f} ms")
    print(f"Normal budget: {NORMAL_BUDGET_MS:.4f} ms")
    print()

    print("Deadline misses:")
    print(result_df.groupby("strategy")["deadline_missed"].sum())
    print()

    print("Detected overruns:")
    print(result_df.groupby("strategy")["overrun"].sum())
    print()

    print("Effective frequency in Hz:")
    periods = result_df.groupby("strategy")["period_ms"].first()
    print(1000.0 / periods)

    plot_deadline_misses(result_df)
    plot_effective_frequency(result_df)
    plot_slack(result_df)
    plot_runtime_with_strategy_deadlines(result_df)
    plot_overruns_by_strategy(result_df)

    print()
    print("Saved plots:")
    print("results/deadline_misses_by_strategy.png")
    print("results/effective_frequency_by_strategy.png")
    print("results/average_slack_by_strategy.png")
    print("results/overruns_by_strategy.png")
    print("results/runtime_deadline_WCET.png")
    print("results/runtime_deadline_NormalOnly.png")
    print("results/runtime_deadline_OverrunHandled.png")


if __name__ == "__main__":
    main()