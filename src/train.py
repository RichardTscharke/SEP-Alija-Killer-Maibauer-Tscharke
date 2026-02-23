from src.training.train_utils import get_device
from src.training.train_loop import trainings_loop
'''
This is the training interface of our project.
For custom training configurations adjust these:
'''
TRAIN_CONFIGURATIONS = {

    # Choose one of "RafCustom", "ResNetLight1" and "ResNetLight2"
    "model": "ResNetLight2",

    # Choose one of "val_acc" and "val_loss"
    "train_on": "val_loss",

    # By default: Class inverse frequency weights. If you flag to False please adjust custom_weights
    "use_inv_freq_w": True,
    #[w_ang, w_dis, w_fear, w_happy, w_sad, w_surprise]
    "custom_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],

    # Training metrics
    "learning_rate": 0.001,
    "epochs": 110,
    "early_stop_patience": 40,

    # Performance metrics
    "batch_size": 64,
    "num_workers": 4,
}

def main():
   
   DEVICE = get_device()

   trainings_loop(TRAIN_CONFIGURATIONS, DEVICE)

if __name__ == "__main__":
    main()
