import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize(stages, show_box = True, show_landmarks = True):
    '''
    Visualizes preprocessing stages for debugging.
    Expects samples with BGR images and box format (x, y, w, h).
    Converts BGR to RGB for display only.
    '''
    n = len(stages)

    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

    if n == 1:
        axes = [axes]

    for ax, (stage, sample) in zip(axes, stages):

        image = sample["image"]
        # BGR → RGB for plotting
        if image.ndim == 3 and image.shape[2] == 3:
            image = image[..., ::-1] 

        bx, by, bw, bh = sample["box"]

        ax.imshow(image)
        ax.set_title(stage)

        # Draw the bounding box
        if show_box:
            ax.add_patch(patches.Rectangle((bx, by), bw, bh, fill=False, color="red"))

        # Draw the landmarks
        if show_landmarks:
            for (lx, ly) in sample["eyes"].values():
                ax.plot(lx, ly, "ro", markersize=3)

    plt.tight_layout()
    plt.show()