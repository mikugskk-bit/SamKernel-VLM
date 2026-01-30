import torch
from samOsem import SamAgentVL

# # 加载 .pth 文件
# state_dict = torch.load("RN50/SAM-decoder-Unf100_ckt.pth")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net = SamAgentVL().to(device)
# # state_dict = torch.load('./RN50/'+model_name+'_ckt.pth')

# net.load_state_dict(state_dict['network'])
# # 提取参数字典
# params_dict = model_dict[['network']]

# # 统计总参数量
# total_params = sum(param_tensor.numel() for param_tensor in params_dict.values())

# print(f"模型的参数量约为：{total_params / 1_000_000:.2f}M（百万个参数）")



import torch
 
model_dict = torch.load("RN50/SAM-decoder-Unf100_ckt.pth")
params_dict=model_dict['network']
 
total_params = 0
for param_tensor in params_dict.values():
    # 将当前参数的元素数（即参数大小）加到总和中
    total_params += param_tensor.numel()
 
print(f"参数量约为：{total_params/1000000:.2f}M（百万个参数）。")