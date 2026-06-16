import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class MoEFFNGating(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts):
        super(MoEFFNGating, self).__init__()
        self.gating_network = nn.Linear(dim, dim)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)) for _ in range(num_experts)])

    def forward(self, x):
        weights = self.gating_network(x)
        weights = torch.nn.functional.softmax(weights, dim=-1)
        outputs = [expert(x) for expert in self.experts]
        outputs = torch.stack(outputs, dim=0)
        outputs = (weights.unsqueeze(0) * outputs).sum(dim=0)
        return outputs


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    把整张特征图切成不重叠的窗口。

    原理：
      原始: (B, H, W, C)  例如 (B, 56, 56, 96)
      第1步: reshape → (B, 56/7, 7, 56/7, 7, 96) = (B, 8, 7, 8, 7, 96)
             将 H 和 W 分别拆为 [段数, 窗口大小]
      第2步: permute → (B, 8, 8, 7, 7, 96)
             让 H段数和W段数相邻，7×7变为相邻的空间维
      第3步: view → (B×64, 7, 7, 96)
             合并 B 和窗口数，64 个窗口互相独立

    Args:
        x: (B, H, W, C)
        window_size (int): 窗口边长，默认 7

    Returns:
        windows: (num_windows×B, window_size, window_size, C)
                 例如 (B×64, 7, 7, 96)
    """
    B, H, W, C = x.shape
    # 将 H 和 W 维度各拆为两个: [段数, 窗口大小]
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 调整维度顺序: 把窗口的空间维(7,7)放到最后，段数维相邻
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    把窗口拼回整张特征图 —— window_partition 的逆操作。

    原理：
      原始: (B×64, 7, 7, 96)
      第1步: view → (B, 8, 8, 7, 7, 96)
             恢复 B 和段数维度
      第2步: permute → (B, 8, 7, 8, 7, 96)
             把 7×7 小块放回 H 和 W 的对应位置
      第3步: view → (B, 56, 56, 96)
             合并段数和窗口大小，恢复完整 H×W

    Args:
        windows: (num_windows×B, window_size, window_size, C)
        window_size (int): 窗口大小
        H (int): 原始特征图高度
        W (int): 原始特征图宽度

    Returns:
        x: (B, H, W, C)
    """
    # 从总窗口数反推 B: 总窗口 = B × (H/7) × (W/7)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # 恢复 (B, H段数, W段数, 7, 7, C) 结构
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # 把 7×7 小块交错排列回 H×W 的对应位置
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" 窗口内多头自注意力 + 相对位置偏置。
    支持 W-MSA(正常窗口) 和 SW-MSA(移位窗口, 通过 mask 屏蔽跨区域 token)。

    Args:
        dim (int): 输入通道数。
        window_size (tuple[int]): 窗口的 (高, 宽)，如 (7, 7)。
        num_heads (int): 多头注意力头数。
        qkv_bias (bool): QKV 投影是否带偏置。
        qk_scale (float | None): 手动指定 QK 缩放。None 时自动 = head_dim^(-0.5)。
        attn_drop (float): 注意力权重的 Dropout 概率。
        proj_drop (float): 输出投影的 Dropout 概率。
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim                                    # 特征维度
        self.window_size = window_size                    # (7, 7)
        self.num_heads = num_heads                        # 注意力头数
        head_dim = dim // num_heads                       # 每个头的维度 = dim / num_heads
        # QK 缩放因子: 防止内积太大导致 Softmax 梯度消失
        # 默认值 head_dim^(-0.5) 来自原始 Transformer 论文
        self.scale = qk_scale or head_dim ** -0.5

        # ── 相对位置偏置表 (Swin 的关键创新之一) ──
        # 窗口内每个像素对之间有一个可学习的偏置值
        # 大小: (2*7-1) × (2*7-1) = 13×13 = 169 种相对位置 × num_heads
        # 与其让模型自己猜像素间的空间关系，不如直接告诉它"你在我左边3格"
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # ── 预计算相对位置索引 (固定值, 不需要梯度) ──
        # 对 7×7 窗口内的每对像素 (i,j)，计算它们的二维相对坐标
        coords_h = torch.arange(self.window_size[0])      # [0,1,2,3,4,5,6]
        coords_w = torch.arange(self.window_size[1])      # [0,1,2,3,4,5,6]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, 7, 7)
        coords_flatten = torch.flatten(coords, 1)         # (2, 49) — 展开为序列
        # 广播减法: 每对像素的坐标差 → (2, 49, 49)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (49, 49, 2)
        # 将坐标从 [-6,6] 平移到 [0,12]，避免负数索引
        relative_coords[:, :, 0] += self.window_size[0] - 1  # H方向 +6
        relative_coords[:, :, 1] += self.window_size[1] - 1  # W方向 +6
        # 将二维坐标压缩为一维索引: row × (2*W-1) + col
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)   # (49, 49)
        # 存为 buffer (不参与梯度, 但随模型保存)
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV 合并投影: dim → 3×dim (一次算出 Q、K、V)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)              # 注意力权重 dropout
        self.proj = nn.Linear(dim, dim)                     # 输出投影
        self.proj_drop = nn.Dropout(proj_drop)              # 输出 dropout

        trunc_normal_(self.relative_position_bias_table, std=.02)  # 初始化位置偏置
        self.softmax = nn.Softmax(dim=-1)                  # 沿最后一维做 Softmax

    def forward(self, x, mask=None):
        """
        Args:
            x: (B_ × num_windows, 49, C) — 所有窗口展平后的 token 序列
            mask: SW-MSA 的跨区域掩码，W-MSA 时为 None

        Returns:
            (B_ × num_windows, 49, C) — 注意力加权后的特征
        """
        B_, N, C = x.shape  # N = 7×7 = 49, C = dim

        # ── 第 1 步: QKV 一次生成并拆分为多头 ──
        # Linear → (B_, 49, 3C) → reshape → (3, B_, num_heads, 49, head_dim)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]     # 各 (B_, num_heads, 49, head_dim)

        # ── 第 2 步: 计算注意力分数 ──
        # Q @ K^T: 每个像素问(Q) 和所有像素答(K) 的匹配度
        q = q * self.scale                     # 缩放防止内积过大
        attn = (q @ k.transpose(-2, -1))       # (B_, num_heads, 49, 49)

        # ── 第 3 步: 加上相对位置偏置 ──
        # 索引查表得到每个像素对的位置偏置 → 加到注意力分数上
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)  # (49, 49, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, 49, 49)
        attn = attn + relative_position_bias.unsqueeze(0)   # 广播加到所有样本

        # ── 第 4 步: SW-MSA 掩码 + Softmax ──
        if mask is not None:
            # 把掩码加到注意力分数上: 不同区域 → -100 → Softmax ≈ 0
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)              # (B_, num_heads, 49, 49) 每行和为1
        attn = self.attn_drop(attn)            # Dropout 正则化

        # ── 第 5 步: 加权求和 + 投影输出 ──
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # 按权重汇总 V → (B_, 49, C)
        x = self.proj(x)                        # 输出投影
        x = self.proj_drop(x)                   # Dropout
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # 估算一个窗口(含 N 个 token)的计算量
        flops = 0
        flops += N * self.dim * 3 * self.dim                     # QKV 投影
        flops += self.num_heads * N * (self.dim // self.num_heads) * N  # Q @ K^T
        flops += self.num_heads * N * N * (self.dim // self.num_heads)   # attn @ V
        flops += N * self.dim * self.dim                         # 输出投影
        return flops


class SwinTransformerBlock(nn.Module):
    r"""
    Swin Transformer 的核心模块 —— 整个架构的"心脏"。

    每个 Block = 1个窗口注意力(W-MSA 或 SW-MSA) + 1个 MLP。
    偶数层用 W-MSA (正常窗口)，奇数层用 SW-MSA (移位窗口，让跨窗口信息流动)。
    每层内部有残差连接(抄近道)，防止深层网络退化。

    信息流: 输入 → [LayerNorm → 切窗口 → 注意力 → 拼回 → 加残差] → [LayerNorm → MLP → 加残差] → 输出

    参数速查:
      dim=96/192/384/768  不同阶段的特征维度
      input_resolution=(56,56) (28,28) (14,14) (7,7)
      window_size=7   窗口边长(像素)
      shift_size=0(W-MSA) 或 3(SW-MSA)
      num_heads=3/6/12/24  多头数
      mlp_ratio=4   MLP 扩维倍数

    Args:
        dim (int): 输入特征向量的长度(通道数)。
        input_resolution (tuple[int]): 当前特征图的高宽。
        num_heads (int): 多头注意力的头数。dim 必须能被 num_heads 整除。
        window_size (int): 窗口边长(像素)，默认 7。7×7=49 个 token 为一组互相"看"。
        shift_size (int): SW-MSA 的移位距离。0=W-MSA(不移)，3=SW-MSA(移3像素)。
        mlp_ratio (float): MLP 扩维倍数，默认 4。
        qkv_bias (bool): QKV 的 Linear 是否带偏置，默认 True。
        qk_scale (float | None): QK 缩放因子。None 时自动 = head_dim^(-0.5)。
        drop (float): 注意力和 MLP 输出的 Dropout 概率，默认 0。
        attn_drop (float): 注意力权重的 Dropout 概率，默认 0。
        drop_path (float): Stochastic Depth 概率(随机跳过当前 Block)，默认 0。
        act_layer (nn.Module): MLP 激活函数，默认 GELU。
        norm_layer (nn.Module): 归一化层，默认 LayerNorm。
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()  # 调用 nn.Module.__init__()，使 PyTorch 跟踪所有子模块

        # ── 保存超参数，forward 中会用 ──
        self.dim = dim                              # 特征通道数，阶段1=96, 阶段2=192...
        self.input_resolution = input_resolution    # 特征图 (H, W)，如 (56, 56)
        self.num_heads = num_heads                 # 多头注意力头数，如 3
        self.window_size = window_size              # 窗口大小，默认 7
        self.shift_size = shift_size                # 移位距离: 0=W-MSA, 3=SW-MSA
        self.mlp_ratio = mlp_ratio                # MLP 扩维倍数，默认 4

        # ── 防御性检查：特征图比窗口还小？退化为全局注意力 ──
        # 瓶颈层分辨率 7×7，窗口也是 7×7 → 整张图=一个窗口 → 不需要移位
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0                              # 关掉移位
            self.window_size = min(self.input_resolution)    # 窗口=全图大小
        # 确保移位距离合法: 必须在 [0, window_size) 内
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # ═══════════════════════════════════════════════════════
        # 子模块 ①: LayerNorm + 窗口注意力
        # ═══════════════════════════════════════════════════════
        # LayerNorm: 对每个 token 的所有通道做标准化(均值=0, 方差=1)
        # 放在注意力前面 → 稳定训练，防止数值爆炸
        self.norm1 = norm_layer(dim)

        # WindowAttention: 窗口内的自注意力 + 相对位置偏置
        # 这是 Swin 最底层的计算 —— 每个窗口内 49 个 token 互相"看"
        self.attn = WindowAttention(
            dim,                                          # 特征维度
            window_size=to_2tuple(self.window_size),      # (7, 7) 或退化的全图大小
            num_heads=num_heads,                          # 多头数
            qkv_bias=qkv_bias,                            # QKV 偏置
            qk_scale=qk_scale,                            # QK 缩放
            attn_drop=attn_drop,                          # 注意力权重 dropout
            proj_drop=drop)                               # 输出 dropout

        # ═══════════════════════════════════════════════════════
        # 子模块 ②: 随机深度 (Stochastic Depth)
        # ═══════════════════════════════════════════════════════
        # 训练时以概率 drop_path 直接跳过注意力输出(短路为输入)
        # 正则化手段: 让模型不依赖某几个特定层，强迫每层都学到有用特征
        # drop_path=0 时退化为 nn.Identity() → 原样通过
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # ═══════════════════════════════════════════════════════
        # 子模块 ③: LayerNorm + MLP (前馈网络)
        # ═══════════════════════════════════════════════════════
        self.norm2 = norm_layer(dim)
        # MLP: dim → 4×dim → GELU → Dropout → dim
        # 注意力负责"看"(信息汇聚)，MLP 负责"想"(加工变换)
        # 为什么 4 倍扩维？给足够容量学复杂的非线性变换
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,                     # 输入维度
                       hidden_features=mlp_hidden_dim,       # 隐藏层 = dim×4
                       act_layer=act_layer,                  # GELU 激活函数(比 ReLU 光滑)
                       drop=drop)                            # Dropout

        # ═══════════════════════════════════════════════════════
        # 子模块 ④: SW-MSA 的注意力掩码 (仅奇数层 / shift_size>0 时生成)
        # ═══════════════════════════════════════════════════════
        if self.shift_size > 0:
            H, W = self.input_resolution

            # --- 第 1 步: 将特征图划分为 3×3=9 个区域 ---
            # 移位后，窗口可能跨越原来不同窗口的像素 → 需要知道每个像素的"原始归属"
            # h_slices 和 w_slices 沿移位产生的边界线切出 3×3=9 块矩形区域
            # 以 H=56, window=7, shift=3 为例:
            #   h_slices[0]=[0,49)  上段(49行)
            #   h_slices[1]=[49,53) 中段(4行=7-3)
            #   h_slices[2]=[53,56) 下段(3行=shift)
            h_slices = (slice(0, -self.window_size),            # 从开头到倒数第7行
                        slice(-self.window_size, -self.shift_size),  # 从倒数第7行到倒数第3行
                        slice(-self.shift_size, None))          # 从倒数第3行到末尾
            w_slices = (slice(0, -self.window_size),            # 同上, 宽度方向
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))

            # --- 第 2 步: 给每个区域编号 0~8 ---
            # img_mask 是一个 56×56 的标签图，每个像素值 ∈ {0,1,...,8}
            img_mask = torch.zeros((1, H, W, 1))                # (1, 56, 56, 1) 空白标签图
            cnt = 0
            for h in h_slices:                                  # 遍历 3 个高度段
                for w in w_slices:                              # 遍历 3 个宽度段
                    img_mask[:, h, w, :] = cnt                  # 该区域所有像素标为 cnt
                    cnt += 1                                    # 编号 +1

            # --- 第 3 步: 把标签图也切成 7×7 窗口 ---
            mask_windows = window_partition(img_mask, self.window_size)
            # → (num_windows, 7, 7, 1) — 每个窗口内 49 个像素各有一个区域编号

            # 展平窗口 → (num_windows, 49)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

            # --- 第 4 步: 广播减法生成配对矩阵 ---
            # mask[:, None] - mask[:, :, None]
            #   列向量(49,1) - 行向量(1,49) = 矩阵(49,49)
            #   矩阵元素(i,j) = 像素i的区域编号 - 像素j的区域编号
            #     = 0  → i和j来自同一区域 → 允许互相关注 ✅
            #     ≠ 0  → i和j来自不同区域 → 需要屏蔽   ❌
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

            # --- 第 5 步: 转为 Softmax 掩码 ---
            # 不同区域 → 填 -100 → exp(-100) ≈ 3.7×10^(-44) ≈ 0 → 注意力被压死
            # 相同区域 → 填 0    → exp(0) = 1 → 正常参与 Softmax
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)) \
                                 .masked_fill(attn_mask == 0, float(0.0))
        else:
            # W-MSA (偶数层): 每个窗口都是纯的，不需要掩码
            attn_mask = None

        # register_buffer: 掩码只算一次，存到 checkpoint，但不参与梯度更新
        # (因为掩码只取决于 input_resolution/window_size/shift_size —— 这些是固定的)
        self.register_buffer("attn_mask", attn_mask)

    # ═══════════════════════════════════════════════════════════
    # forward(): 前向传播 —— 特征图经过一个 SwinBlock 的完整流水线
    #
    # 总流程 (8 步):
    #   (1) LayerNorm → (2) 展开为二维 → (3) [循环移位] → (4) 切窗口
    #   → (5) 窗口注意力 → (6) 拼回二维 → (7) [反向移位] → 残差连接
    #   → (8) LayerNorm → MLP → 残差连接
    # ═══════════════════════════════════════════════════════════
    def forward(self, x):
        H, W = self.input_resolution                      # 特征图尺寸，如 (56, 56)
        B, L, C = x.shape                                 # (批大小, token数=3136, 通道数=96)
        assert L == H * W, "input feature has wrong size"  # 确保序列长度 = H×W

        # ══ 阶段 1: LayerNorm → 恢复二维空间形状 ══
        shortcut = x              # 保存原始输入 → 用于最后的残差连接 (抄近道)
        x = self.norm1(x)         # LayerNorm: 每个 token 的 C 个通道 → 均值0, 方差1
        x = x.view(B, H, W, C)    # (B, H×W, C) → (B, H, W, C)  恢复二维空间

        # ══ 阶段 2: 循环移位 (仅 SW-MSA / shift_size>0) ══
        # torch.roll(-3,-3): 整张图沿 H 和 W 各向左上滚动 3 像素
        # 滚出边界外的像素从对面"循环"回来
        # 目的: 让原本在不同窗口的像素移位后进入同一窗口 → 可以跨窗口交流
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x              # W-MSA: 不移位，直接切

        # ══ 阶段 3: 切窗口 + 展平 ══
        # window_partition: (B,56,56,C) → (B×64,7,7,C)  切成64个独立窗口
        x_windows = window_partition(shifted_x, self.window_size)
        # 每个窗口展平: 7×7=49 个 token 排成一维序列 → (B×64, 49, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # ══ 阶段 4: 窗口内自注意力 (W-MSA 或 SW-MSA) ══
        # 这是整个 Block 最核心的计算
        # 每个窗口内 49 个 token:
        #   W-MSA: 正常做注意力，所有 token 自由互看
        #   SW-MSA: 跨区域 token 被 mask 屏蔽 → 只和同区域 token 交流
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        # → (B×64, 49, C)  形状不变，每个 token 的内容被注意力更新

        # ══ 阶段 5: 恢复空间形状 + 拼窗口回整图 ══
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # → (B×64, 7, 7, C)
        # window_reverse: 把 64 个 7×7 窗口按原始位置拼回 56×56 大图
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        # → (B, 56, 56, C)

        # ══ 阶段 6: 反向循环移位 + 恢复序列格式 ══
        # torch.roll(+3,+3): 把之前滚过来的像素滚回原位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x              # W-MSA: 没移就不滚
        x = x.view(B, H * W, C)        # (B,56,56,C) → (B,3136,C)  恢复为一维序列

        # ══ 阶段 7: 第一个残差连接 (注意力分支) ══
        # shortcut = 进入 Block 时的原始输入
        # 加上注意力处理后的结果 → 梯度可以直接流过 "+" 回传
        # drop_path: 随机深度——训练时可能丢弃整个注意力分支
        x = shortcut + self.drop_path(x)

        # ══ 阶段 8: 第二个残差连接 (MLP 分支) ══
        # norm2 → MLP(dim→4dim→dim) → drop_path → 加到当前值
        # 注意力汇集的全局信息，MLP 进一步加工变换
        # 残差连接再次确保梯度畅通 → 堆很多层也不会退化
        shortcut = x       # 保存 MLP 前的值 (等下加上去)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    # ── print(block) 时的信息显示 ──
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    # ── 估算浮点运算量 (用于模型分析，不需要有 GPU) ──
    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # LayerNorm1: H×W 个 token × dim 次运算
        flops += self.dim * H * W
        # 窗口注意力: 窗口数 × 每窗口注意力计算量
        nW = H * W / self.window_size / self.window_size           # 窗口总数
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # MLP: 每个 token 的两层全连接
        #   dim → 4dim: H×W × dim × 4dim = H×W × 4dim²
        #   4dim → dim: H×W × 4dim × dim = H×W × 4dim²
        #   合计 = H×W × 8dim² = 2 × H×W × dim × dim × mlp_ratio
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # LayerNorm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerSys(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                         patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                         patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        return x, x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)

        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
