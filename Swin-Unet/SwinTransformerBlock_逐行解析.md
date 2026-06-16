# SwinTransformerBlock 逐行解析

> 文件位置: `networks/swin_transformer_unet_skip_expand_decoder_sys.py`
> 核心类，Swin-Unet 整个架构就靠它堆叠而成。

---

## 目录

- [0. 前置知识：它依赖的 3 个组件](#0-前置知识它依赖的-3-个组件)
- [1. `__init__` — 初始化：搭好骨架](#1-__init__--初始化搭好骨架)
- [2. 注意力掩码生成 — SW-MSA 最关键的部分](#2-注意力掩码生成--sw-msa-最关键的部分)
- [3. `forward` — 前向传播：一次完整的数据流动](#3-forward--前向传播一次完整的数据流动)
- [4. 辅助方法](#4-辅助方法)

---

## 0. 前置知识：它依赖的 3 个组件

在深入 `SwinTransformerBlock` 之前，先理解它调用的 3 个外部函数/类。

### 0.1 `window_partition` — 把整张图切成窗口

```python
def window_partition(x, window_size):
    """
    x: (B, H, W, C)  例如 (B, 56, 56, 96)
    返回: (B × num_windows, window_size, window_size, C)
          例如 (B × 64, 7, 7, 96)
    """
    B, H, W, C = x.shape
    # 第 1 步: 把 H 和 W 维度拆成 [段数, 窗口大小]
    # (B, 56, 56, 96) → (B, 8, 7, 8, 7, 96)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

    # 第 2 步: 调整维度顺序，让窗口成为独立维度
    # (B, 8, 7, 8, 7, 96) → (B, 8, 8, 7, 7, 96)
    # 含义: (B, H段数, W段数, 窗口内H, 窗口内W, 通道)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    # 第 3 步: 合并 B 和窗口数
    # → (B × 64, 7, 7, 96)
    windows = windows.view(-1, window_size, window_size, C)
    return windows
```

**直观理解（把 reshape 当成乐高拆装）：**

```
整张图 56×56:
┌──────────────────────┐
│ A B C D E F G H │ 8列
│ ...              │
│ ...          8行 │
└──────────────────────┘

reshape → 切成 7×7 的小块 → 64 个窗口:
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│...│...│...│...│...│...│...│...│  = 64 个 7×7 窗口
├───┼───┼───┼───┼───┼───┼───┼───┤
│57 │58 │...│...│...│...│...│ 64│
└───┴───┴───┴───┴───┴───┴───┴───┘

view(-1, 7, 7, 96) → 展平为 (64 × B, 7, 7, 96)
→ 每个窗口独立存在，可以分别做注意力
```

---

### 0.2 `window_reverse` — 把窗口拼回整张图

```python
def window_reverse(windows, window_size, H, W):
    """
    windows: (B × num_windows, 7, 7, C)
    返回: (B, H, W, C)  例如 (B, 56, 56, 96)
    """
    # 第 1 步: 算出 B
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # 例如: windows有 1024×B 个 → B = 1024×B / (56×56/7/7) = B

    # 第 2 步: 恢复结构 (B, H段数, W段数, 7, 7, C)
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)

    # 第 3 步: 把 7×7 块拼回完整行列 → (B, 56, 56, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
```

---

### 0.3 `WindowAttention` — 窗口内的自注意力

这是整个架构最深层的计算核心。它的 `forward` 只做 4 件事：

```python
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, ...):
        # Q、K、V 合并为一个 Linear 层，一次算出三个矩阵
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 相对位置偏置表 —— 这是 Swin 的关键创新
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*Wh-1) * (2*Ww-1), num_heads))
        self.proj = nn.Linear(dim, dim)    # 输出投影
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # x: (num_windows×B, 49, C)  — 每个窗口 7×7=49 个 token

        # 步骤 1: QKV 一次生成
        qkv = self.qkv(x)          # → (..., 49, 3C)
        q, k, v = qkv.chunk(3, dim=-1)  # 拆成 3 份

        # 步骤 2: 计算注意力分数 — Q 和 K 的内积
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn: (num_windows×B, num_heads, 49, 49)
        # attn[i,j] = 窗口内第 i 个像素对第 j 个像素的"关注度"

        # 步骤 3: 加上相对位置偏置 + 掩码（如 SW-MSA） + Softmax
        attn = attn + self.relative_position_bias_table[...]
        if mask is not None:
            attn = attn + mask.unsqueeze(0)  # 屏蔽跨窗口 token
        attn = self.softmax(attn)

        # 步骤 4: 加权求和 + 投影
        x = attn @ v               # 按注意力权重汇总
        x = self.proj(x)           # 线性投影
        return x
```

**QKV 是什么（不看公式版）：**
- **Q (Query)** = "我在找什么？" → 每个像素根据自身内容生成一个问题
- **K (Key)** = "我是什么？" → 每个像素生成一个"标签"
- **V (Value)** = "我实际有什么信息？" → 每个像素的实际特征
- **Q @ K^T** = 匹配度 → 高匹配 = 多关注，低匹配 = 少关注
- **attn @ V** = 按匹配度加权汇总 → 每个像素获得窗口内所有像素的综合信息

---

## 1. `__init__` — 初始化：搭好骨架

```python
def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
             mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
             drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
```

### 1.1 参数表

| 参数 | 默认值 | 含义 | 在本项目中的值 |
|:--|:--|:--|:--|
| `dim` | — | 输入通道数（特征维度） | 96 / 192 / 384 / 768 |
| `input_resolution` | — | 当前特征图的高宽 `(H, W)` | (56,56) / (28,28) / (14,14) / (7,7) |
| `num_heads` | — | 多头注意力的头数 | 3 / 6 / 12 / 24 |
| `window_size` | 7 | 窗口边长（像素） | 7 |
| `shift_size` | 0 | 移位距离。0 = W-MSA（正常），>0 = SW-MSA（移位） | 0 或 3 |
| `mlp_ratio` | 4.0 | MLP 隐藏层维度 = dim × mlp_ratio | 4 |
| `qkv_bias` | True | QKV 线性层是否带偏置 | True |
| `qk_scale` | None | 缩放因子，None 时自动 = head_dim^(-0.5) | None |
| `drop` | 0.0 | 注意力输出和 MLP 输出的 Dropout 概率 | 0.0 |
| `attn_drop` | 0.0 | 注意力权重的 Dropout 概率 | 0.0 |
| `drop_path` | 0.0 | 随机深度（Stochastic Depth）概率 | 0.0 ~ 0.2 |
| `act_layer` | nn.GELU | MLP 中的激活函数 | GELU |
| `norm_layer` | nn.LayerNorm | 归一化层类型 | LayerNorm |

### 1.2 逐行解读

```python
super().__init__()
```
调用 `nn.Module` 的初始化，注册所有子模块。

```python
self.dim = dim
self.input_resolution = input_resolution
self.num_heads = num_heads
self.window_size = window_size
self.shift_size = shift_size
self.mlp_ratio = mlp_ratio
```
把参数存为实例属性，后续 `forward` 里要用。

```python
if min(self.input_resolution) <= self.window_size:
    self.shift_size = 0
    self.window_size = min(self.input_resolution)
```
**防御性代码：** 如果特征图尺寸已经小于等于窗口大小（例如瓶颈层 7×7、窗口 7），移位就没有意义了——直接关掉移位，窗口改为全图大小。此时 W-MSA 退化为全局自注意力。

```python
assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
```
移位距离必须在 [0, window_size) 范围内——如果移位 ≥ 窗口大小，等于没移。

### 1.3 构建 3 个子模块

```python
self.norm1 = norm_layer(dim)
```
**第 1 个 LayerNorm：** 注意力之前的归一化。LayerNorm 对每个 token 的 C 个通道做标准化——让每个像素的特征均值=0、方差=1，防止数值爆炸。

```python
self.attn = WindowAttention(
    dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
```
**窗口注意力模块：** 这是整个 Block 最核心的计算。`to_2tuple(7)` → `(7, 7)`。

```python
self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
```
**随机深度（Stochastic Depth）：** 训练时以概率 `drop_path` **随机跳过整个 Block 的输出**。这是一种正则化手段，防止过拟合。drop_path=0 时退化为恒等映射（什么都不做）。

```python
self.norm2 = norm_layer(dim)
```
**第 2 个 LayerNorm：** MLP 之前的归一化。

```python
mlp_hidden_dim = int(dim * mlp_ratio)
self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
               act_layer=act_layer, drop=drop)
```
**MLP（多层感知机）：** 两层的全连接网络：
```
dim → (dim × 4) → GELU → Dropout → dim
```
为什么需要 MLP？注意力负责"看"（信息汇聚），MLP 负责"想"（信息加工）。4 倍扩维让网络有足够容量学习复杂模式。

---

## 2. 注意力掩码生成 — SW-MSA 最关键的部分

这是 `__init__` 中最后也是最复杂的部分（第 220-243 行）。

### 2.1 为什么需要掩码？

当 `shift_size > 0`（奇数层），窗口发生了移位。此时直接切窗口会导致**原来不属于同一窗口的像素被分到一起**——如果不加掩码，它们会错误地互相注意到。

**目标：** 生成一个掩码矩阵，让不同"原始窗口"的 token 之间注意力 = 0。

### 2.2 掩码生成 — 逐行分解

```python
if self.shift_size > 0:
    H, W = self.input_resolution          # 例如 (56, 56)
    img_mask = torch.zeros((1, H, W, 1))  # 创建一张全 0 的"标签图"
```

**创建标签模板：** 形状 `(1, 56, 56, 1)`，每个像素初始标签 = 0。

```python
    h_slices = (slice(0, -self.window_size),           # 切片 A: [0, 49)
                slice(-self.window_size, -self.shift_size),  # 切片 B: [49, 53)
                slice(-self.shift_size, None))          # 切片 C: [53, 56)
    w_slices = (slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None))
```

**把图分成 9 块（3×3 区域）：**

以 H=56, W=56, window=7, shift=3 为例：

```
h_slices[0] = [0, 49)     # 从上边到倒数第7行
h_slices[1] = [49, 53)    # 中间的 4 行（7-3=4）
h_slices[2] = [53, 56)    # 最后 3 行

w_slices[0] = [0, 49)     # 从左到倒数第7列
w_slices[1] = [49, 53)    # 中间 4 列
w_slices[2] = [53, 56)    # 最右 3 列
```

```
┌──────────────────────────────────┐
│                                  │   h[0]
│          区域 (0,0)              │   w[0]
│                                  │
├──────────────────────┬───────────┤
│                      │           │   h[1]
│      区域 (1,0)      │  区域(1,2)│   w[0→1,2]
│                      │           │
├──────────────────────┼───────────┤
│                      │           │   h[2]
│      区域 (2,0)      │  区域(2,2)│   w[0→1,2]
│                      │           │
└──────────────────────┴───────────┘
      w[0]                 w[1→2]
```

共 3×3=9 个区域。移位窗口产生的"碎片窗口"就来自这 9 块。

```python
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
```

**给 9 个区域分别编号 0-8：**

```
┌─────┬──────┬────┐
│  0  │  1   │ 2  │
├─────┼──────┼────┤
│  3  │  4   │ 5  │
├─────┼──────┼────┤
│  6  │  7   │ 8  │
└─────┴──────┴────┘
```

现在 `img_mask` 的 56×56 个像素各有一个区域编号。

```python
    mask_windows = window_partition(img_mask, self.window_size)
    # → (num_windows, 7, 7, 1)
    # 每个窗口内的 49 个像素各有一个[区域编号]

    mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
    # → (num_windows, 49)
    # 展平为 49 维向量
```

```python
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    # → (num_windows, 49, 49)
    # 广播减法: 窗口内每对像素的区域编号之差
    # 同一区域 = 0，不同区域 ≠ 0
```

**举例：** 如果一个窗口跨越了两个原始区域，窗口内有 20 个像素来自区域 A、29 个来自区域 B：

```
attn_mask[i,j] = 0  → 像素 i 和 j 来自同一区域 → 允许关注
attn_mask[i,j] ≠ 0  → 像素 i 和 j 来自不同区域 → 禁止关注
```

```python
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
```

**转换为实际掩码值：**
- 同一区域 → 加 0（不影响注意力）
- 不同区域 → 加 -100（Softmax 后 ≈ 0，相当于忽略）

`-100` 不是随便选的——Softmax 中 $e^{-100} \approx 3.7 \times 10^{-44}$，在实际计算中就是 0。

```python
else:
    attn_mask = None       # W-MSA（偶数层）不需要掩码

self.register_buffer("attn_mask", attn_mask)
```

`register_buffer` 把掩码注册为持久化的张量（保存到 checkpoint，但不参与梯度）。因为掩码只取决于 `input_resolution`、`window_size`、`shift_size`——这些是固定的，所以掩码只需算一次。

---

## 3. `forward` — 前向传播：一次完整的数据流动

这是每次前向传播时执行的代码。数据形如一条流水线经过。

```python
def forward(self, x):
    H, W = self.input_resolution          # 例如 (56, 56)
    B, L, C = x.shape                     # B=批大小, L=3136, C=96
    assert L == H * W, "input feature has wrong size"
```

确认 token 数 = H×W。输入格式是 `(B, H×W, C)`——所有像素展平为一维序列，这是 Transformer 标准输入。

### 3.1 阶段 1：归一化 + 恢复空间形状

```python
    shortcut = x                          # 保存输入的副本（残差连接用）
    x = self.norm1(x)                     # LayerNorm 归一化
    x = x.view(B, H, W, C)               # (B, 3136, C) → (B, 56, 56, C)
```

**为什么 `shortcut = x`？** 这是残差连接（Residual Connection）的关键——最后要把注意力输出和原始输入**加回去**。这个"抄近道"让梯度可以直接回传，训练深网络不退化。

### 3.2 阶段 2：循环移位（仅 SW-MSA）

```python
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                               dims=(1, 2))
    else:
        shifted_x = x                    # W-MSA 不移位
```

**`torch.roll`** 把整个特征图沿 H 和 W 方向各**向左上滚动 3 个像素**。

```
原始:              roll(-3, -3):
┌──────────┐      ┌──────────┬───┐
│ A B C D  │      │ E F G H  │ A │
│ E F G H  │  →   │ I J K L  │ B │
│ I J K L  │      ├──────────┼───┤
│ M N O P  │      │ M N O P  │ D │
└──────────┘      └──────────┴───┘

滚出图外的像素从对面进来（循环移位）
```

**为什么移位？** 移位+切窗后，原来在窗口 1 和窗口 2 的像素现在在同一个新窗口里 → 它们可以交流了 → 这是"Shifted Window"名字的由来。

### 3.3 阶段 3：切窗口 → 注意力 → 拼回

```python
    x_windows = window_partition(shifted_x, self.window_size)
    # → (B × num_windows, 7, 7, C)
    # 例如 B=2: (2×64, 7, 7, 96) = (128, 7, 7, 96)

    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
    # → (128, 49, 96)
    # 每个窗口展平: 7×7=49 个 token
```

**格式转换：** `WindowAttention` 需要输入为 `(num_windows, 49, C)`。

```python
    attn_windows = self.attn(x_windows, mask=self.attn_mask)
    # → (128, 49, 96)
    # 每个窗口内部做了自注意力
```

**这一行的效果：**
- 128 个窗口，每个窗口内 49 个像素互相"看"
- 如果是 SW-MSA（奇数层），跨区域像素被掩码屏蔽
- 输出：每个像素的特征更新为窗口内其他相关像素的加权组合

```python
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    # → (128, 7, 7, 96)
    # 恢复空间形状

    shifted_x = window_reverse(attn_windows, self.window_size, H, W)
    # → (B, 56, 56, 96)
    # 把 64 个窗口拼回完整特征图
```

### 3.4 阶段 4：反向移位 + 残差连接

```python
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
                       dims=(1, 2))
    else:
        x = shifted_x

    x = x.view(B, H * W, C)              # (B, 56, 56, 96) → (B, 3136, 96)

    x = shortcut + self.drop_path(x)      # 残差连接
```

**反向移位：** `roll(+3, +3)` 把之前滚动过去的像素滚回来，恢复原始空间对应关系。

**`shortcut + drop_path(x)`:** 把原始输入和注意力输出相加。`drop_path` 可能随机丢弃整个注意力输出（训练时的正则化）。

### 3.5 阶段 5：MLP + 第二次残差连接

```python
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    # self.norm2(x)  → 归一化
    # self.mlp(...)  → 全连接网络加工（dim→4×dim→dim）
    # self.drop_path → 随机深度
    # x + ...        → 残差连接
```

**完整的信息流：**

```
输入 x
  │
  ├→ [保存 shortcut]
  │
  ├→ LayerNorm1 → 恢复空间形状 → [循环移位] → 切窗口
  │                                              ↓
  │                                         注意力 (W-MSA/SW-MSA)
  │                                              ↓
  │                                   拼窗口 → [反向移位] → 展平
  │                                              ↓
  ├→ + shortcut ──────────────────→ x (残差连接 1)
  │
  ├→ [保存 shortcut]
  │
  ├→ LayerNorm2 → MLP (dim → 4×dim → dim)
  │                    ↓
  ├→ + shortcut ──→ x (残差连接 2)
  │
  ↓
输出
```

---

## 4. 辅助方法

### 4.1 `extra_repr` — 打印信息

```python
def extra_repr(self) -> str:
    return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
           f"num_heads={self.num_heads}, window_size={self.window_size}, " \
           f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
```

当你在 Python 里 `print(block)` 时会显示：`dim=96, input_resolution=(56, 56), num_heads=3, window_size=7, shift_size=0, mlp_ratio=4.0`

### 4.2 `flops` — 计算量估算

```python
def flops(self):
    flops = 0
    H, W = self.input_resolution
    flops += self.dim * H * W          # norm1: 每个像素 C 次乘法
    nW = H * W / self.window_size / self.window_size
    flops += nW * self.attn.flops(self.window_size * self.window_size)  # W-MSA
    flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio          # MLP
    flops += self.dim * H * W          # norm2
    return flops
```

简单估算**不用 GPU 跑也能知道计算量**。例如窗口注意力：
- 64 个窗口 × 每个窗口 49 个 token × 注意力复杂度

---

## 附录：关键数字速查

| 符号 | 值 | 含义 |
|:--|:--|:--|
| window_size | 7 | 窗口边长，W-MSA/SW-MSA 的基本单元 |
| shift_size | 0 或 3 | 0 = W-MSA（偶数层），3 = SW-MSA（奇数层） |
| mlp_ratio | 4 | MLP 扩维倍数 |
| dim | 96/192/384/768 | 不同阶段的特征维度 |
| num_heads | 3/6/12/24 | 对应阶段的多头注意力头数 |
| shortcut | 残差连接 | 保证梯度能回传，训练不退化 |
| -100 | Softmax 掩码值 | 将跨区域注意力的权重压为 0 |
