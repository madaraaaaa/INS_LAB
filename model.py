import torch
import torch.nn as nn

class EMG_CNN(nn.Module):
    def __init__(self):
        super(EMG_CNN, self).__init__()

        # --- Architecture Configuration ---
        # Input Shape: (Batch, 1, 128, 512)
        
        # Layer 1
        # Conv: 1 channel -> 16 channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        # Output after Pool1: (16, 64, 256)

        # Layer 2
        # Conv: 16 channels -> 32 channels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 
        # Output after Pool2: (32, 32, 128)

        # Layer 3
        # Conv: 32 channels -> 64 channels
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 
        # Output after Pool3: (64, 16, 64)

        # Fully Connected Layers
        # Flatten Calculation: 64 channels * 16 height * 64 width
        self.flatten_size = 64 * 16 * 64

        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(128, 6) # Output: 6 Classes

    def forward(self, x):
        # Feature Extraction
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        # Flattening (Batch, Channels, Height, Width) -> (Batch, Features)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# Optional: Test code to verify shape
if __name__ == "__main__":
    # Create a fake input (Batch=1, Channel=1, Sensors=128, Time=512)
    fake_input = torch.randn(1, 1, 128, 512)
    model = EMG_CNN()
    output = model(fake_input)
    print("✅ Architecture Check Passed!")
    print(f"Input Shape: {fake_input.shape}")
    print(f"Output Shape: {output.shape} (Should be [1, 6])")