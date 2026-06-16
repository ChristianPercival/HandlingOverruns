import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "results/runtime_log.csv"
OUTPUT_DIR = "results"

# Simulated task parameters in milliseconds
FAST_PERIOD_MS = 2.0
NORMAL_BUDGET_MS = 0.5
WCET_SAFETY_FACTOR = 1.10

# Other periodic system tasks
CONTROL_TASK_TIME_MS = 0.4
COMM_TASK_TIME_MS = 0.3

# How many frames to show in timeline plots
# We select frames around the first hard/missing overrun area.
START_FRAME = 55
END_FRAME = 68


def add_segment(segments, strategy, task, start, duration, frame_id):
    """Add one CPU execution segment to the timeline."""
    if duration <= 0:
        return

    segments.append({
        "strategy": strategy,
        "task": task,
        "start_ms": start,
        "duration_ms": duration,
        "end_ms": start + duration,
        "frame_id": frame_id,
    })


def simulate_normal_only(df):
    """
    NormalOnly:
    Image task runs until it finishes.
    If it overruns, it blocks other tasks.
    """
    segments = []

    for _, row in df.iterrows():
        frame_id = int(row["frame_id"])
        image_time = float(row["execution_time_ms"])
        period_start = frame_id * FAST_PERIOD_MS

        current_time = period_start

        # Image task runs first and is allowed to continue until finished.
        add_segment(segments, "NormalOnly", "Image", current_time, image_time, frame_id)
        current_time += image_time

        # Other tasks run only after image task finishes.
        add_segment(segments, "NormalOnly", "Control", current_time, CONTROL_TASK_TIME_MS, frame_id)
        current_time += CONTROL_TASK_TIME_MS

        add_segment(segments, "NormalOnly", "Comm", current_time, COMM_TASK_TIME_MS, frame_id)

    return segments


def simulate_wcet(df, wcet_estimate_ms):
    """
    WCET:
    Each frame gets a long enough reserved slot.
    This is safe but reduces effective frequency.
    """
    segments = []
    period_ms = wcet_estimate_ms * WCET_SAFETY_FACTOR

    for _, row in df.iterrows():
        frame_id = int(row["frame_id"])
        image_time = float(row["execution_time_ms"])
        period_start = frame_id * period_ms

        current_time = period_start

        # Image executes inside a WCET-sized reservation.
        add_segment(segments, "WCET", "Image", current_time, image_time, frame_id)
        current_time += image_time

        # Other tasks still fit because the period is much longer.
        add_segment(segments, "WCET", "Control", current_time, CONTROL_TASK_TIME_MS, frame_id)
        current_time += CONTROL_TASK_TIME_MS

        add_segment(segments, "WCET", "Comm", current_time, COMM_TASK_TIME_MS, frame_id)

        # Unused reserved time is intentionally not plotted as a task.
        # It represents idle/spare capacity caused by WCET reservation.

    return segments


def simulate_cbs_inspired(df):
    """
    CBS-inspired:
    Image task can only use its normal budget first.
    If it still has remaining work, Control and Comm get CPU before
    the image task continues later.
    """
    segments = []

    for _, row in df.iterrows():
        frame_id = int(row["frame_id"])
        image_time = float(row["execution_time_ms"])
        period_start = frame_id * FAST_PERIOD_MS

        current_time = period_start

        first_chunk = min(image_time, NORMAL_BUDGET_MS)
        remaining = max(0.0, image_time - first_chunk)

        # Image gets only its reserved budget first.
        add_segment(segments, "CBSInspired", "Image budget", current_time, first_chunk, frame_id)
        current_time += first_chunk

        # Other tasks are protected and can execute.
        add_segment(segments, "CBSInspired", "Control", current_time, CONTROL_TASK_TIME_MS, frame_id)
        current_time += CONTROL_TASK_TIME_MS

        add_segment(segments, "CBSInspired", "Comm", current_time, COMM_TASK_TIME_MS, frame_id)
        current_time += COMM_TASK_TIME_MS

        # Overrunning image task continues later.
        if remaining > 0:
            add_segment(segments, "CBSInspired", "Image overrun", current_time, remaining, frame_id)

    return segments


def simulate_cbshd_inspired(df):
    """
    CBShd-inspired:
    Similar to CBS, but the remaining overrun is treated more precisely.
    If only a small amount remains, only that needed amount is executed.
    This version also splits large remaining work into budget-sized chunks,
    showing more controlled execution.
    """
    segments = []

    for _, row in df.iterrows():
        frame_id = int(row["frame_id"])
        image_time = float(row["execution_time_ms"])
        period_start = frame_id * FAST_PERIOD_MS

        current_time = period_start
        remaining = image_time

        # First normal budget
        first_chunk = min(remaining, NORMAL_BUDGET_MS)
        add_segment(segments, "CBShdInspired", "Image budget", current_time, first_chunk, frame_id)
        current_time += first_chunk
        remaining -= first_chunk

        # Protected tasks execute after normal budget is consumed.
        add_segment(segments, "CBShdInspired", "Control", current_time, CONTROL_TASK_TIME_MS, frame_id)
        current_time += CONTROL_TASK_TIME_MS

        add_segment(segments, "CBShdInspired", "Comm", current_time, COMM_TASK_TIME_MS, frame_id)
        current_time += COMM_TASK_TIME_MS

        # Remaining image work is executed in controlled chunks.
        # If remaining is small, only the small remaining part is executed.
        while remaining > 0:
            chunk = min(remaining, NORMAL_BUDGET_MS)
            add_segment(segments, "CBShdInspired", "Image remaining", current_time, chunk, frame_id)
            current_time += chunk
            remaining -= chunk

            # Small separation to visually show budget chunks.
            # This represents possible scheduling points.
            if remaining > 0:
                current_time += 0.05

    return segments


def filter_segments_for_frames(segments, start_frame, end_frame):
    return [
        s for s in segments
        if start_frame <= s["frame_id"] <= end_frame
    ]


def plot_timeline(segments, strategy_name, filename):
    """
    Plot a Gantt-style CPU timeline.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filtered = filter_segments_for_frames(segments, START_FRAME, END_FRAME)

    if not filtered:
        print(f"No segments to plot for {strategy_name}")
        return

    task_order = [
        "Image",
        "Image budget",
        "Image overrun",
        "Image remaining",
        "Control",
        "Comm",
    ]

    y_positions = {task: i for i, task in enumerate(task_order)}

    plt.figure(figsize=(12, 5))

    for seg in filtered:
        task = seg["task"]
        y = y_positions.get(task, 0)

        plt.barh(
            y=y,
            width=seg["duration_ms"],
            left=seg["start_ms"],
            height=0.6,
            label=task
        )

    # Avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    plt.yticks(
        list(y_positions.values()),
        list(y_positions.keys())
    )

    plt.xlabel("Time (ms)")
    plt.ylabel("Task / execution part")
    plt.title(f"CPU Timeline: {strategy_name}")
    plt.grid(axis="x")
    plt.legend(unique.values(), unique.keys(), loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    wcet_estimate_ms = df["execution_time_ms"].max()

    print(f"Measured WCET estimate: {wcet_estimate_ms:.4f} ms")
    print(f"Normal budget: {NORMAL_BUDGET_MS:.4f} ms")
    print(f"Fast period: {FAST_PERIOD_MS:.4f} ms")
    print()

    normal_segments = simulate_normal_only(df)
    wcet_segments = simulate_wcet(df, wcet_estimate_ms)
    cbs_segments = simulate_cbs_inspired(df)
    cbshd_segments = simulate_cbshd_inspired(df)

    all_segments = (
        normal_segments
        + wcet_segments
        + cbs_segments
        + cbshd_segments
    )

    timeline_df = pd.DataFrame(all_segments)
    timeline_df.to_csv("results/cpu_timeline_log.csv", index=False)

    plot_timeline(normal_segments, "NormalOnly", "timeline_NormalOnly.png")
    plot_timeline(wcet_segments, "WCET", "timeline_WCET.png")
    plot_timeline(cbs_segments, "CBSInspired", "timeline_CBSInspired.png")
    plot_timeline(cbshd_segments, "CBShdInspired", "timeline_CBShdInspired.png")

    print("Saved:")
    print("results/cpu_timeline_log.csv")
    print("results/timeline_NormalOnly.png")
    print("results/timeline_WCET.png")
    print("results/timeline_CBSInspired.png")
    print("results/timeline_CBShdInspired.png")


if __name__ == "__main__":
    main()