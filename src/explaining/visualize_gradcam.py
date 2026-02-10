import cv2
import numpy as np
import matplotlib.pyplot as plt

emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]

BG_COLOR     = "#909392"
BAR_BG_COLOR = "#373737"
BAR_COLOR    = "lightgray"

def normalize_for_viz(cam):
    cam = cam.copy()
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    return cam

def visualize(original_img,
              aligned_img,
              cam_aligned,
              cam_original,
              probs,
              threshold=0.4):
    
    # Create Grad-CAM for the aligned image
    cam_aligned = cv2.resize(cam_aligned, (aligned_img.shape[1], aligned_img.shape[0]))
    cam_aligned = np.clip(cam_aligned, 0, 1)

    # Create the heatmap for the aligned version
    heat_aligned = np.uint8(255 * cam_aligned)
    heat_aligned = cv2.applyColorMap(heat_aligned, cv2.COLORMAP_JET)
    heat_aligned = cv2.cvtColor(heat_aligned, cv2.COLOR_BGR2RGB)

    # Create the overlay for the aligned version
    aligned_rgb = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
    mask_a = cam_aligned >= threshold
    overlay_aligned = aligned_rgb.copy()
    overlay_aligned[mask_a] = (0.6 * overlay_aligned[mask_a] + 0.4 * heat_aligned[mask_a]).astype(np.uint8)

    # Create Grad-CAM for the original image
    cam_original = np.clip(cam_original, 0, 1)

    # Create the heatmap for the original version
    heat_original = np.uint8(255 * cam_original)
    heat_original = cv2.applyColorMap(heat_original, cv2.COLORMAP_JET)
    heat_original = cv2.cvtColor(heat_original, cv2.COLOR_BGR2RGB)

    # Create the overlay for the original version
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    mask_o = cam_original >= threshold
    overlay_original = original_rgb.copy()
    overlay_original[mask_o] = (0.6 * overlay_original[mask_o] + 0.4 * heat_original[mask_o]).astype(np.uint8)

    # Figure & Grid layout
    fig = plt.figure(figsize=(22, 10), facecolor=BG_COLOR)
    fig.canvas.manager.set_window_title("Explainable AI: Grad-CAM")

    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.2], height_ratios=[1, 1])

    ax_orig      = fig.add_subplot(gs[0, 0])
    ax_align     = fig.add_subplot(gs[0, 1])
    ax_cam_orig  = fig.add_subplot(gs[1, 0])
    ax_overlay   = fig.add_subplot(gs[1, 1])
    ax_cam_only  = fig.add_subplot(gs[:, 2])
    ax_bar       = fig.add_subplot(gs[:, 3])

    # Image plots
    ax_orig.imshow(original_rgb)
    ax_orig.set_title("Original Image", color="white")

    ax_align.imshow(overlay_aligned)
    ax_align.set_title("Aligned + CAM", color="white")

    ax_cam_orig.imshow(normalize_for_viz(cam_original), cmap="jet", vmin=0, vmax=1)
    ax_cam_orig.set_title("Cam (original space)", color="white")

    ax_overlay.imshow(overlay_original)
    ax_overlay.set_title("Final Overlay", color="white")

    ax_cam_only.imshow(cam_aligned, cmap="jet")
    ax_cam_only.set_title("Cam (aligned)", color="white")

    # Visual sugarcoating
    for ax in [ax_orig, ax_align, ax_cam_orig, ax_overlay, ax_cam_only]:
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

    print(
    "aligned:", cam_aligned.min(), cam_aligned.max(),
    "original:", cam_original.min(), cam_original.max()
)