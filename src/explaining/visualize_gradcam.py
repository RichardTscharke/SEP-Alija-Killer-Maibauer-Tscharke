import cv2
import numpy as np
import matplotlib.pyplot as plt

emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]

BG_COLOR     = "#909392"
BAR_BG_COLOR = "#373737"
BAR_COLOR    = "lightgray"

def visualize(original_img,
              aligned_img,
              cam,
              probs,
              threshold=0.4):

    # Resize CAM to match aligned face resolution
    cam = cv2.resize(cam, (aligned_img.shape[1], aligned_img.shape[0]))

    # Create the heatmap
    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Only overlay regions with sufficiently activation based on explain_image configurations
    mask = cam >= threshold
    overlay = cv2.cvtColor(aligned_img.copy(), cv2.COLOR_BGR2RGB)
    overlay[mask] = (
        0.6 * overlay[mask] + 0.4 * heatmap_color[mask]
    ).astype(np.uint8)

    # Figure & Grid
    fig = plt.figure(figsize=(16, 8), facecolor=BG_COLOR)
    fig.canvas.manager.set_window_title("Explainable AI: Grad-CAM")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2])

    ax_orig   = fig.add_subplot(gs[0, 0])
    ax_align  = fig.add_subplot(gs[0, 1])
    ax_cam    = fig.add_subplot(gs[1, 0])
    ax_overlay= fig.add_subplot(gs[1, 1])
    ax_bar    = fig.add_subplot(gs[:, 2])

    # Image handling
    ax_orig.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    ax_orig.set_title("Original", color="white")

    ax_align.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
    ax_align.set_title("Aligned", color="white")

    ax_cam.imshow(cam, cmap="jet")
    ax_cam.set_title("Grad-CAM", color="white")

    ax_overlay.imshow(overlay)
    ax_overlay.set_title("Overlay", color="white")

    # Visual sugarcoating
    for ax in [ax_orig, ax_align, ax_cam, ax_overlay]:
        ax.axis("off")
        ax.set_facecolor(BG_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(2)

    # Bar chart
    probs_pct = probs * 100
    bars = ax_bar.bar(emotions, probs_pct, color=BAR_COLOR)

    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("Probability (%)", color="white", fontsize=12)
    ax_bar.set_title("Emotion Probabilities", color="white", fontsize=20, pad=15)

    ax_bar.set_facecolor(BAR_BG_COLOR)
    ax_bar.tick_params(axis="x", colors="white", labelsize=11)
    ax_bar.tick_params(axis="y", colors="white", labelsize=11)

    for spine in ax_bar.spines.values():
        spine.set_color("black")
        spine.set_linewidth(2)

    ax_bar.grid(axis="y", color="white", alpha=0.2)

    # Percentages per emotion(bar)
    for bar, p in zip(bars, probs_pct):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{p:.1f}%",
            ha="center",
            va="bottom",
            color="white",
            fontsize=11,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.show()