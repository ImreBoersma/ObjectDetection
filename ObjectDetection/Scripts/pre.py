import sys
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(sys.argv[1])
# Create a mini-batch as expected by the model.
input_batch = input_tensor.unsqueeze(0)
