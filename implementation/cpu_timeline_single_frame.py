import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "results/runtime_log.csv"
OUTPUT_DIR = "results"

FAST_PERIOD_MS = 2.0
NORMAL_BUDGET_MS = 0.5
WCET_SAFETY_FACTOR = 1.10

# This is only for the conceptual diagram.
# It represents other system work that should not be blocked forever.
OTHER_TASKS_TIME_MS = 0.7

# This is used only to visually show CBS budget blocks clearly.
# It represents full server recharges after the normal budget is consumed.
CBS_BLOCK_MS = 2.0


def add_segment(segments, strategy, task, start, duration):
    if duration <= 0:
        return

    segments.append({
        "strategy": strategy,
        "task": task,
        "start_ms": start,
        "duration_ms": duration,
        "end_ms": start + duration,
    })


def choose_representative_overrun_frame(df):
    """
    Choose one hard or missing frame that clearly overruns,
    but avoid choosing only an extreme spike if possible.
    """
    candidates = df[
        (df["execution_time_ms"] > 5.0)
        & (df["execution_time_ms"] < 10.0)
    ]

    if len(candidates) == 0:
        candidates = df[df["execution_time_ms"] > NORMAL_BUDGET_MS]

    if len(candidates) == 0:
        raise ValueError("No overrun frame found in runtime_log.csv")

    median_runtime = candidates["execution_time_ms"].median()
    selected = candidates.iloc[
        (candidates["execution_time_ms"] - median_runtime).abs().argsort().iloc[0]
    ]

    return selected


def simulate_conceptual_allocation(image_time, wcet_estimate):
    """
    This is a conceptual allocation diagram, not a full OS scheduler trace.

    It shows the main difference between:
    - NormalOnly
    - WCET
    - CBS-inspired handling
    - CBShd-inspired handling
    """
    all_segments = []

    # ------------------------------------------------------------
    # NormalOnly
    # Image task runs until finished.
    # Other tasks are delayed until after the image overrun.
    # ------------------------------------------------------------
    t = 0.0
    add_segment(all_segments, "NormalOnly", "Image task", t, image_time)
    t += image_time
    add_segment(all_segments, "NormalOnly", "Other tasks delayed", t, OTHER_TASKS_TIME_MS)

    # ------------------------------------------------------------
    # WCET
    # Worst-case time is reserved.
    # Safe, but unused reserved time is visible.
    # ------------------------------------------------------------
    t = 0.0
    wcet_slot = wcet_estimate * WCET_SAFETY_FACTOR

    add_segment(all_segments, "WCET", "Image task", t, image_time)
    t += image_time

    add_segment(all_segments, "WCET", "Other tasks", t, OTHER_TASKS_TIME_MS)
    t += OTHER_TASKS_TIME_MS

    unused = max(0.0, wcet_slot - t)
    add_segment(all_segments, "WCET", "Unused WCET reservation", t, unused)

    # ------------------------------------------------------------
    # CBS-inspired
    # Image gets its normal budget first.
    # Other tasks are protected.
    # The overrun continues in full CBS budget blocks.
    # The final block may contain unused budget.
    # ------------------------------------------------------------
    t = 0.0
    remaining = image_time

    first_chunk = min(remaining, NORMAL_BUDGET_MS)
    add_segment(all_segments, "CBS-inspired", "Image normal budget", t, first_chunk)
    t += first_chunk
    remaining -= first_chunk

    add_segment(all_segments, "CBS-inspired", "Other tasks", t, OTHER_TASKS_TIME_MS)
    t += OTHER_TASKS_TIME_MS

    while remaining > 0:
        used = min(remaining, CBS_BLOCK_MS)
        unused_budget = CBS_BLOCK_MS - used

        add_segment(all_segments, "CBS-inspired", "CBS budget used", t, used)
        t += used
        remaining -= used

        if unused_budget > 0:
            add_segment(all_segments, "CBS-inspired", "CBS unused final budget", t, unused_budget)
            t += unused_budget

        if remaining > 0:
            # Small visual gap between conceptual CBS recharges.
            t += 0.05

    # ------------------------------------------------------------
    # CBShd-inspired
    # Image gets its normal budget first.
    # Other tasks are protected.
    # Remaining computation is handled precisely.
    # Final block is only as long as the remaining work.
    # ------------------------------------------------------------
    t = 0.0
    remaining = image_time

    first_chunk = min(remaining, NORMAL_BUDGET_MS)
    add_segment(all_segments, "CBShd-inspired", "Image normal budget", t, first_chunk)
    t += first_chunk
    remaining -= first_chunk

    add_segment(all_segments, "CBShd-inspired", "Other tasks", t, OTHER_TASKS_TIME_MS)
    t += OTHER_TASKS_TIME_MS

    while remaining > 0:
        used = min(remaining, CBS_BLOCK_MS)

        add_segment(all_segments, "CBShd-inspired", "CBShd precise work", t, used)
        t += used
        remaining -= used

        if remaining > 0:
            # Small visual gap between conceptual recharges.
            t += 0.05

    return pd.DataFrame(all_segments)


def plot_conceptual_timeline(timeline_df, image_time, wcet_estimate, selected_frame_id):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    strategy_order = [
        "NormalOnly",
        "WCET",
        "CBS-inspired",
        "CBShd-inspired",
    ]

    task_colors = {
        "Image task": "#1f77b4",
        "Image normal budget": "#1f77b4",

        "Other tasks": "#2ca02c",
        "Other tasks delayed": "#2ca02c",

        "Unused WCET reservation": "#bdbdbd",

        "CBS budget used": "#ff7f0e",
        "CBS unused final budget": "#f7c97f",

        "CBShd precise work": "#ff7f0e",
    }

    y_positions = {
        strategy: i for i, strategy in enumerate(reversed(strategy_order))
    }

    plt.figure(figsize=(13, 5))

    for _, row in timeline_df.iterrows():
        strategy = row["strategy"]
        task = row["task"]
        y = y_positions[strategy]

        plt.barh(
            y=y,
            width=row["duration_ms"],
            left=row["start_ms"],
            height=0.55,
            color=task_colors.get(task, "#333333"),
            edgecolor="black",
            linewidth=0.5,
            label=task
        )

    # Mark the normal image budget and the nominal period.
    plt.axvline(
        NORMAL_BUDGET_MS,
        linestyle=":",
        linewidth=1.5,
        color="black",
        label="0.5 ms normal budget"
    )

    plt.axvline(
        FAST_PERIOD_MS,
        linestyle="--",
        linewidth=1.5,
        color="black",
        label="2 ms nominal period"
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    plt.yticks(
        list(y_positions.values()),
        list(y_positions.keys())
    )

    plt.xlabel("Time after frame release (ms)")
    plt.ylabel("Handling method")
    plt.title(
        f"Conceptual CPU Allocation During One Overrun "
        f"(frame {selected_frame_id}, image time = {image_time:.2f} ms)"
    )

    plt.grid(axis="x", alpha=0.25)

    plt.legend(
        unique.values(),
        unique.keys(),
        loc="upper right",
        fontsize=8
    )

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "timeline_conceptual_overrun_handling.png")
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")


def main():
    df = pd.read_csv(INPUT_FILE)

    selected = choose_representative_overrun_frame(df)

    selected_frame_id = int(selected["frame_id"])
    image_time = float(selected["execution_time_ms"])
    wcet_estimate = float(df["execution_time_ms"].max())

    print(f"Selected frame: {selected_frame_id}")
    print(f"Difficulty: {selected['difficulty']}")
    print(f"Image execution time: {image_time:.4f} ms")
    print(f"Measured WCET estimate: {wcet_estimate:.4f} ms")
    print(f"WCET slot with safety factor: {wcet_estimate * WCET_SAFETY_FACTOR:.4f} ms")
    print(f"Normal image budget: {NORMAL_BUDGET_MS:.4f} ms")
    print(f"Nominal period: {FAST_PERIOD_MS:.4f} ms")

    timeline_df = simulate_conceptual_allocation(image_time, wcet_estimate)
    timeline_df.to_csv("results/cpu_timeline_conceptual.csv", index=False)

    plot_conceptual_timeline(
        timeline_df,
        image_time,
        wcet_estimate,
        selected_frame_id
    )

    print("Saved results/cpu_timeline_conceptual.csv")


if __name__ == "__main__":
    main()