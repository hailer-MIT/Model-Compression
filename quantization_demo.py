
# quantization_demo.py
import torch
import torch.nn as nn

class CreditRiskModel(nn.Module):
    def __init__(self, input_size=20):
        super(CreditRiskModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Load and quantize
model = CreditRiskModel()
model.load_state_dict(torch.load('baseline_model.pth', weights_only=True))
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)
print("Quantized model created successfully!")
print(f"Model size: {sum(p.numel() for p in quantized_model.parameters()):,} parameters")
