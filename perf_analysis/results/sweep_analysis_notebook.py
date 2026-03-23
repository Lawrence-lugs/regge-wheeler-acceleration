#%%

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


#%%
RESULTS_DIR = Path(__file__).resolve().parent
SWEEP_CSV = RESULTS_DIR / "sweep_latency_summary.csv"

df = pd.read_csv(SWEEP_CSV)
print(f"Loaded {len(df)} rows from {SWEEP_CSV.name}")
df.head()


#%%
def save_bandwidth_slice_heatmaps(frame: pd.DataFrame) -> None:
	"""Save one speedup heatmap per L1 bandwidth slice."""

	bandwidth_values = sorted(frame["l1_bandwidth_bits"].unique())

	for bandwidth in bandwidth_values:
		slice_df = frame.loc[frame["l1_bandwidth_bits"] == bandwidth]
		pivot = (
			slice_df.pivot_table(
				index="max_lanes",
				columns="tpu_size",
				values="speedup_all_primitives_vs_scalar",
				aggfunc="mean",
			)
			.sort_index()
			.sort_index(axis=1)
		)

		fig, ax = plt.subplots(figsize=(6, 4))
		im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", origin="lower")

		ax.set_title(f"Speedup Heatmap @ {bandwidth}b L1 Bandwidth")
		ax.set_xlabel("TPU Tile Size")
		ax.set_ylabel("Max Vector Lanes")
		ax.set_xticks(range(len(pivot.columns)))
		ax.set_xticklabels(pivot.columns)
		ax.set_yticks(range(len(pivot.index)))
		ax.set_yticklabels(pivot.index)

		for row_idx in range(len(pivot.index)):
			for col_idx in range(len(pivot.columns)):
				value = float(pivot.iloc[row_idx, col_idx])
				ax.text(
					col_idx,
					row_idx,
					f"{value:.1f}x",
					ha="center",
					va="center",
					color="white" if value > pivot.values.mean() else "black",
					fontsize=8,
				)

		cbar = fig.colorbar(im, ax=ax)
		cbar.set_label("Speedup (All Primitives vs Scalar Only)")

		fig.tight_layout()
		png_path = RESULTS_DIR / f"sweep_heatmap_speedup_bw_{bandwidth}.png"
		pdf_path = RESULTS_DIR / f"sweep_heatmap_speedup_bw_{bandwidth}.pdf"
		fig.savefig(png_path, dpi=300)
		fig.savefig(pdf_path)
		plt.close(fig)

		print(f"Saved {png_path.name} and {pdf_path.name}")


#%%
save_bandwidth_slice_heatmaps(df)


#%%
def save_concatenated_heatmap(frame: pd.DataFrame) -> None:
    """Old-paper-style single concatenated-axis heatmap.

    Y-axis: TPU tile sizes repeated for each bandwidth slice, with bold
    bandwidth group labels to the left.  X-axis: max vector lanes.
    Thick white rules separate each bandwidth group.
    Saves two variants: linear and log colormap scale.
    """
    import numpy as np
    from matplotlib.colors import LogNorm
    from matplotlib.transforms import blended_transform_factory

    bandwidth_values = sorted(frame["l1_bandwidth_bits"].unique())
    tpu_sizes = sorted(frame["tpu_size"].unique())
    lane_values = sorted(frame["max_lanes"].unique())
    n_per_group = len(tpu_sizes)

    # Build (bw, tpu) × lane matrix in group order
    data_rows: list[list[float]] = []
    row_labels: list[str] = []
    group_starts: list[int] = []

    for bw in bandwidth_values:
        group_starts.append(len(data_rows))
        bw_df = frame[frame["l1_bandwidth_bits"] == bw]
        for tpu in tpu_sizes:
            data_rows.append([
                float(
                    bw_df.loc[
                        (bw_df["tpu_size"] == tpu) & (bw_df["max_lanes"] == lane),
                        "speedup_all_primitives_vs_scalar",
                    ].iloc[0]
                )
                for lane in lane_values
            ])
            row_labels.append(str(tpu))

    data = np.array(data_rows)
    n_rows, n_cols = data.shape
    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))

    variants = [
        ("linear", None,        (vmin + vmax) / 2.0),
        ("log",    LogNorm(),   10 ** ((np.log10(max(vmin, 1e-3)) + np.log10(vmax)) / 2.0)),
    ]

    for scale, norm, vmid in variants:
        fig, ax = plt.subplots(figsize=(4, 5))

        im = ax.imshow(
            data, cmap="viridis", aspect="auto", origin="upper",
            norm=norm, vmin=vmin if norm is None else None, vmax=vmax if norm is None else None,
        )

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(lane_values, fontsize=8)
        ax.set_xlabel("Max Vector Lanes", labelpad=4)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=7)
        # "TPU Tile Size" label sits just outside the fine TPU tick labels
        ax.text(
            -0.19, 0.5, "TPU Tile Size",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8, rotation=90, clip_on=False,
        )

        # Cell annotations
        for r in range(n_rows):
            for c in range(n_cols):
                v = data[r, c]
                ax.text(
                    c, r, f"{v:.0f}x",
                    ha="center", va="center", fontsize=6,
                    color="white" if v < vmid else "black",
                )

        # Bandwidth group separators, coarse BW tick labels, and outer "Bandwidth" label
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        for bw, start in zip(bandwidth_values, group_starts):
            mid = start + (n_per_group - 1) / 2.0
            if start > 0:
                ax.axhline(start - 0.5, color="white", linewidth=2.5, zorder=3)
            ax.text(
                -0.27, mid, f"{bw}b",
                transform=trans, ha="right", va="center",
                fontsize=9, fontweight="bold", clip_on=False,
            )
        ax.text(
            -0.5, 0.5, "Bandwidth",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, fontweight="bold", rotation=90, clip_on=False,
        )

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Speedup vs Scalar Only", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(
                RESULTS_DIR / f"sweep_heatmap_concatenated_{scale}.{suffix}",
                dpi=300, bbox_inches="tight",
            )
        plt.close(fig)
        print(f"Saved sweep_heatmap_concatenated_{scale}.png and .pdf")


save_concatenated_heatmap(df)

