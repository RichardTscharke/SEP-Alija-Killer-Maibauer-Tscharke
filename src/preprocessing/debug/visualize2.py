import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

save_path = "/Users/richardachtnull/Desktop/final_report/preprocess_pipeline.pdf"

def visualize(stages, show_box=True, show_landmarks=True):

    with mpl.rc_context({
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.linewidth": 1.0,
    }):

        n = len(stages)

        # Single column width if <=2, else full width
        width = 3.4 if n <= 2 else 6.8
        fig, axes = plt.subplots(1, n, figsize=(width, 3.0))

        if n == 1:
            axes = [axes]

        for ax, (stage, sample) in zip(axes, stages):

            image = sample["image"]

            # BGR → RGB
            if image.ndim == 3 and image.shape[2] == 3:
                image = image[..., ::-1]

            bx, by, bw, bh = sample["box"]

            ax.imshow(image)
            ax.set_title("")

            ax.axis("off")

            # Bounding box (subtle red, thinner line)
            if show_box:
                ax.add_patch(
                    patches.Rectangle(
                        (bx, by),
                        bw,
                        bh,
                        fill=False,
                        linewidth=2.0,
                        edgecolor=(0, 0.8, 0)
                    )
                )

            # Landmarks (smaller markers)
            if show_landmarks:
                for (lx, ly) in sample["eyes"].values():
                    ax.plot(
                        lx,
                        ly,
                        marker="o",
                        color=(1, 0, 0),
                        markersize=2.5
                    )

        plt.tight_layout()

        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
