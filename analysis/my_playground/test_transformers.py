# %%
import torch
import torch.nn as nn
import torch.optim as optim
# %%
class TransformerRegressionModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=3, dim_feedforward=2048, dropout=0.1):
        super(TransformerRegressionModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)  # 回帰タスクのための出力層

    def forward(self, src, src_mask=None):
        memory = self.encoder(src, src_mask)
        # シーケンス長とバッチサイズに関して平均を取る
        memory = memory.mean(dim=0)  # (バッチサイズ, 埋め込み次元)
        output = self.fc(memory)
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# モデルのインスタンス化
model = TransformerRegressionModel()

# ダミーデータの生成
src = torch.rand((10, 32, 512))  # (シーケンス長, バッチサイズ, 埋め込み次元)

# マスクの生成（必要に応じて）
src_mask = model.generate_square_subsequent_mask(src.size(0))

# フォワードパスの実行
output = model(src, src_mask)

print(output.shape)  # (バッチサイズ, 1)

criterion = nn.MSELoss()  # 平均二乗誤差損失
optimizer = optim.Adam(model.parameters(), lr=0.001)
# %%
output.shape
# %%
