import torch
import torch.nn as nn
import torch.nn.functional as F

# Định nghĩa lại lớp CNN_WRAP
class CNN_WRAP(nn.Module):
    def __init__(self, model_ft):
        super(CNN_WRAP, self).__init__()
        # Kiểm tra nếu model_ft không phải là một instance của nn.Module
        if not isinstance(model_ft, nn.Module):
            raise ValueError("model_ft must be an instance of nn.Module")
        self.cnn_out = nn.Sequential(model_ft)

        self.mr1 = nn.Linear(3, 8)
        self.mr1_bn = nn.BatchNorm1d(8)
        self.mr1_drop = nn.Dropout(0.5)

        self.fc1 = nn.Linear(24, 7)

    def forward(self, x, mr):
        y_cnn = self.cnn_out(x)

        y_mr = F.relu(self.mr1_drop(self.mr1_bn(self.mr1(mr))))
        x = self.fc1(torch.cat((y_cnn, y_mr), 1))
        return F.log_softmax(x, dim=1)

# Load the model from .pt file
model_dict = torch.load('D:/Hoc tap/AI project/Skin_Cancer_Detection/CNN_Skin_Cancer_Detection_Project2.pt')

# Lấy state_dict của mô hình chính
if 'model_state_dict' in model_dict:
    model_state_dict = model_dict['model_state_dict']
else:
    model_state_dict = model_dict

# Khởi tạo một phiên bản của lớp CNN_WRAP
model = CNN_WRAP(model_state_dict)

# Nạp state_dict vào lớp mô hình
model.load_state_dict(model_state_dict, strict=False)

# Set mô hình vào chế độ đánh giá
model.eval()

# Chuyển đổi mô hình từ PyTorch sang TensorFlow Lite
traced_script_module = torch.jit.trace(model, (torch.randn(1, 3, 224, 224), torch.randn(1, 3)))
buffer = traced_script_module._save_to_buffer()

# Lưu mô hình dưới dạng tệp .h5
with open('model.h5', 'wb') as f:
    f.write(buffer)
