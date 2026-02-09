import torch

emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]

def load_model(model_class, weight_path, device, num_classes = len(emotions)):

    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def np_to_tensor(np_img, device):
    np_img = np_img[:, :, ::-1].astype("float32") / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).to(device)
