
# SwinTransformer开始


# SwinTransformer 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]]], device='cuda:0')
```


x.shape: 

```python
torch.Size([1, 3, 224, 224])
```


# PatchEmbed 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]]], device='cuda:0')
```


x.shape: 

```python
torch.Size([1, 3, 224, 224])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>padding操作</div>


x.shape: 

```python
torch.Size([1, 3, 224, 224])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>下采样patch_size倍</div>


: 

```python
x = self.proj(x)
```


self.proj: 

```python
Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
```


x.shape: 

```python
torch.Size([1, 96, 56, 56])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>变换操作</div>


: 

```python
x = x.flatten(2).transpose(1, 2)
```


: 

```python
flatten: [B, C, H, W] -> [B, C, HW]
```


: 

```python
transpose: [B, C, HW] -> [B, HW, C]
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>norm操作</div>


: 

```python
x = self.norm(x)
```


self.norm: 

```python
LayerNorm((96,), eps=1e-05, elementwise_affine=True)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[0.4074, 1.3865, 1.5364],
         [0.4074, 1.3865, 1.5364],
         [0.4074, 1.3865, 1.5364]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


# PatchEmbed 结束


# for layer in self.layers


# BasicLayer 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>A basic Swin Transformer layer for one stage</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.4074, 1.3865, 1.5364],
         [0.4074, 1.3865, 1.5364],
         [0.4074, 1.3865, 1.5364]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


H: 

```python
56
```


W: 

```python
56
```


## 创建mask：self.create_mask(x, H, W)


### window_partition 开始


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>将feature map按照window_size划分成一个个没有重叠的window</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]]]], device='cuda:0')
```


x.shape: 

```python
torch.Size([1, 56, 56, 1])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
```


x.shape: 

```python
torch.Size([1, 8, 7, 8, 7, 1])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>permute操作</div>


: 

```python
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
```


windows.shape: 

```python
torch.Size([64, 7, 7, 1])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]]],


        [[[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]]],


        [[[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]]]], device='cuda:0')
```


windows.shape: 

```python
torch.Size([64, 7, 7, 1])
```


### window_partition 结束


<div style='color:#fe618e;font-weight:800;font-size:23px;'>创建mask</div>


: 

```python
attn_mask = self.create_mask(x, H, W)
```


attn_mask.shape: 

```python
torch.Size([64, 49, 49])
```


## for blk in self.blocks:


<div style='color:#fe618e;font-weight:800;font-size:23px;'>for blk in self.blocks:</div>


核心代码: 

```python
x = blk(x, attn_mask)
```


## SwinTransformerBlock 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.4074, 1.3865, 1.5364],
         [0.4074, 1.3865, 1.5364],
         [0.4074, 1.3865, 1.5364]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


attn_mask[:3,:3,:3]: 

```python
tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]], device='cuda:0')
```


attn_mask.shape: 

```python
torch.Size([64, 49, 49])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>保存输入便于做残差连接</div>


: 

```python
shortcut = x
```


shortcut.shape: 

```python
torch.Size([1, 3136, 96])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>norm操作</div>


: 

```python
x = self.norm1(x)
```


self.norm1: 

```python
LayerNorm((96,), eps=1e-05, elementwise_affine=True)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H, W, C)
```


x.shape: 

```python
torch.Size([1, 56, 56, 96])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>把feature map给pad到window size的整数倍</div>


<div style='color:#fd7949;font-weight:800;font-size:23px;'>cyclic shift</div>


: 

```python
shifted_x = x
```


: 

```python
attn_mask = None
```


shifted_x.shape: 

```python
torch.Size([1, 56, 56, 96])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>partition windows</div>


### window_partition 开始


<div style='color:#3296ee;font-weight:800;font-size:23px;'>将feature map按照window_size划分成一个个没有重叠的window</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375]],

         [[0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375]],

         [[0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 56, 56, 96])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
```


x.shape: 

```python
torch.Size([1, 8, 7, 8, 7, 96])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>permute操作</div>


: 

```python
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
```


windows.shape: 

```python
torch.Size([64, 7, 7, 96])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375]],

         [[0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375]],

         [[0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375]]],


        [[[0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375]],

         [[0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375]],

         [[0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375]]],


        [[[0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375]],

         [[0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375]],

         [[0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375],
          [0.4077, 1.3875, 1.5375]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([64, 7, 7, 96])
```


### window_partition 结束


: 

```python
x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
```


x_windows.shape: 

```python
torch.Size([64, 49, 96])
```


### WindowAttention 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.4077, 1.3875, 1.5375],
         [0.4077, 1.3875, 1.5375],
         [0.4077, 1.3875, 1.5375]],

        [[0.4077, 1.3875, 1.5375],
         [0.4077, 1.3875, 1.5375],
         [0.4077, 1.3875, 1.5375]],

        [[0.4077, 1.3875, 1.5375],
         [0.4077, 1.3875, 1.5375],
         [0.4077, 1.3875, 1.5375]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([64, 49, 96])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
```


self.qkv: 

```python
Linear(in_features=96, out_features=288, bias=True)
```


qkv.shape: 

```python
torch.Size([3, 64, 3, 49, 32])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
q, k, v = qkv.unbind(0)
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1127, -0.1350, -0.2559],
          [ 0.1127, -0.1350, -0.2559],
          [ 0.1127, -0.1350, -0.2559]],

         [[-0.1723, -0.3099,  0.3068],
          [-0.1723, -0.3099,  0.3068],
          [-0.1723, -0.3099,  0.3068]],

         [[ 0.4080, -0.1704,  0.0453],
          [ 0.4080, -0.1704,  0.0453],
          [ 0.4080, -0.1704,  0.0453]]],


        [[[ 0.1127, -0.1350, -0.2559],
          [ 0.1127, -0.1350, -0.2559],
          [ 0.1127, -0.1350, -0.2559]],

         [[-0.1723, -0.3099,  0.3068],
          [-0.1723, -0.3099,  0.3068],
          [-0.1723, -0.3099,  0.3068]],

         [[ 0.4080, -0.1704,  0.0453],
          [ 0.4080, -0.1704,  0.0453],
          [ 0.4080, -0.1704,  0.0453]]],


        [[[ 0.1127, -0.1350, -0.2559],
          [ 0.1127, -0.1350, -0.2559],
          [ 0.1127, -0.1350, -0.2559]],

         [[-0.1723, -0.3099,  0.3068],
          [-0.1723, -0.3099,  0.3068],
          [-0.1723, -0.3099,  0.3068]],

         [[ 0.4080, -0.1704,  0.0453],
          [ 0.4080, -0.1704,  0.0453],
          [ 0.4080, -0.1704,  0.0453]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


q.shape: 

```python
torch.Size([64, 3, 49, 32])
```


k[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0884, -0.2740, -0.1342],
          [ 0.0884, -0.2740, -0.1342],
          [ 0.0884, -0.2740, -0.1342]],

         [[ 0.1688, -0.2264,  0.0091],
          [ 0.1688, -0.2264,  0.0091],
          [ 0.1688, -0.2264,  0.0091]],

         [[ 0.0582,  0.2651, -0.0319],
          [ 0.0582,  0.2651, -0.0319],
          [ 0.0582,  0.2651, -0.0319]]],


        [[[ 0.0884, -0.2740, -0.1342],
          [ 0.0884, -0.2740, -0.1342],
          [ 0.0884, -0.2740, -0.1342]],

         [[ 0.1688, -0.2264,  0.0091],
          [ 0.1688, -0.2264,  0.0091],
          [ 0.1688, -0.2264,  0.0091]],

         [[ 0.0582,  0.2651, -0.0319],
          [ 0.0582,  0.2651, -0.0319],
          [ 0.0582,  0.2651, -0.0319]]],


        [[[ 0.0884, -0.2740, -0.1342],
          [ 0.0884, -0.2740, -0.1342],
          [ 0.0884, -0.2740, -0.1342]],

         [[ 0.1688, -0.2264,  0.0091],
          [ 0.1688, -0.2264,  0.0091],
          [ 0.1688, -0.2264,  0.0091]],

         [[ 0.0582,  0.2651, -0.0319],
          [ 0.0582,  0.2651, -0.0319],
          [ 0.0582,  0.2651, -0.0319]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


k.shape: 

```python
torch.Size([64, 3, 49, 32])
```


v[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0747, -0.3110, -0.0579],
          [ 0.0747, -0.3110, -0.0579],
          [ 0.0747, -0.3110, -0.0579]],

         [[-0.0714, -0.1530, -0.1182],
          [-0.0714, -0.1530, -0.1182],
          [-0.0714, -0.1530, -0.1182]],

         [[-0.1459,  0.2129, -0.1816],
          [-0.1459,  0.2129, -0.1816],
          [-0.1459,  0.2129, -0.1816]]],


        [[[ 0.0747, -0.3110, -0.0579],
          [ 0.0747, -0.3110, -0.0579],
          [ 0.0747, -0.3110, -0.0579]],

         [[-0.0714, -0.1530, -0.1182],
          [-0.0714, -0.1530, -0.1182],
          [-0.0714, -0.1530, -0.1182]],

         [[-0.1459,  0.2129, -0.1816],
          [-0.1459,  0.2129, -0.1816],
          [-0.1459,  0.2129, -0.1816]]],


        [[[ 0.0747, -0.3110, -0.0579],
          [ 0.0747, -0.3110, -0.0579],
          [ 0.0747, -0.3110, -0.0579]],

         [[-0.0714, -0.1530, -0.1182],
          [-0.0714, -0.1530, -0.1182],
          [-0.0714, -0.1530, -0.1182]],

         [[-0.1459,  0.2129, -0.1816],
          [-0.1459,  0.2129, -0.1816],
          [-0.1459,  0.2129, -0.1816]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


v.shape: 

```python
torch.Size([64, 3, 49, 32])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>q = q * self.scale</div>


: 

```python
q = q * self.scale
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0199, -0.0239, -0.0452],
          [ 0.0199, -0.0239, -0.0452],
          [ 0.0199, -0.0239, -0.0452]],

         [[-0.0305, -0.0548,  0.0542],
          [-0.0305, -0.0548,  0.0542],
          [-0.0305, -0.0548,  0.0542]],

         [[ 0.0721, -0.0301,  0.0080],
          [ 0.0721, -0.0301,  0.0080],
          [ 0.0721, -0.0301,  0.0080]]],


        [[[ 0.0199, -0.0239, -0.0452],
          [ 0.0199, -0.0239, -0.0452],
          [ 0.0199, -0.0239, -0.0452]],

         [[-0.0305, -0.0548,  0.0542],
          [-0.0305, -0.0548,  0.0542],
          [-0.0305, -0.0548,  0.0542]],

         [[ 0.0721, -0.0301,  0.0080],
          [ 0.0721, -0.0301,  0.0080],
          [ 0.0721, -0.0301,  0.0080]]],


        [[[ 0.0199, -0.0239, -0.0452],
          [ 0.0199, -0.0239, -0.0452],
          [ 0.0199, -0.0239, -0.0452]],

         [[-0.0305, -0.0548,  0.0542],
          [-0.0305, -0.0548,  0.0542],
          [-0.0305, -0.0548,  0.0542]],

         [[ 0.0721, -0.0301,  0.0080],
          [ 0.0721, -0.0301,  0.0080],
          [ 0.0721, -0.0301,  0.0080]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.scale: 

```python
0.1767766952966369
```


q.shape: 

```python
torch.Size([64, 3, 49, 32])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>计算q和k相似性的得分</div>


: 

```python
attn = (q @ k.transpose(-2, -1))
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0414, -0.0414, -0.0414],
          [-0.0414, -0.0414, -0.0414],
          [-0.0414, -0.0414, -0.0414]],

         [[ 0.1029,  0.1029,  0.1029],
          [ 0.1029,  0.1029,  0.1029],
          [ 0.1029,  0.1029,  0.1029]],

         [[-0.0483, -0.0483, -0.0483],
          [-0.0483, -0.0483, -0.0483],
          [-0.0483, -0.0483, -0.0483]]],


        [[[-0.0414, -0.0414, -0.0414],
          [-0.0414, -0.0414, -0.0414],
          [-0.0414, -0.0414, -0.0414]],

         [[ 0.1029,  0.1029,  0.1029],
          [ 0.1029,  0.1029,  0.1029],
          [ 0.1029,  0.1029,  0.1029]],

         [[-0.0483, -0.0483, -0.0483],
          [-0.0483, -0.0483, -0.0483],
          [-0.0483, -0.0483, -0.0483]]],


        [[[-0.0414, -0.0414, -0.0414],
          [-0.0414, -0.0414, -0.0414],
          [-0.0414, -0.0414, -0.0414]],

         [[ 0.1029,  0.1029,  0.1029],
          [ 0.1029,  0.1029,  0.1029],
          [ 0.1029,  0.1029,  0.1029]],

         [[-0.0483, -0.0483, -0.0483],
          [-0.0483, -0.0483, -0.0483],
          [-0.0483, -0.0483, -0.0483]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([64, 3, 49, 49])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>relative_position_bias</div>


: 

```python
relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
```


relative_position_bias[:3,:3,:3]: 

```python
tensor([[[ 0.0063,  0.0119, -0.0126],
         [-0.0385,  0.0063,  0.0119],
         [ 0.0099, -0.0385,  0.0063]],

        [[ 0.0222,  0.0048, -0.0090],
         [ 0.0032,  0.0222,  0.0048],
         [ 0.0232,  0.0032,  0.0222]],

        [[-0.0228,  0.0180,  0.0115],
         [-0.0374, -0.0228,  0.0180],
         [ 0.0193, -0.0374, -0.0228]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


relative_position_bias.shape: 

```python
torch.Size([3, 49, 49])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>attn加上bias操作</div>


: 

```python
attn = attn + relative_position_bias.unsqueeze(0)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0351, -0.0295, -0.0539],
          [-0.0799, -0.0351, -0.0295],
          [-0.0315, -0.0799, -0.0351]],

         [[ 0.1251,  0.1077,  0.0939],
          [ 0.1060,  0.1251,  0.1077],
          [ 0.1261,  0.1060,  0.1251]],

         [[-0.0712, -0.0304, -0.0368],
          [-0.0858, -0.0712, -0.0304],
          [-0.0291, -0.0858, -0.0712]]],


        [[[-0.0351, -0.0295, -0.0539],
          [-0.0799, -0.0351, -0.0295],
          [-0.0315, -0.0799, -0.0351]],

         [[ 0.1251,  0.1077,  0.0939],
          [ 0.1060,  0.1251,  0.1077],
          [ 0.1261,  0.1060,  0.1251]],

         [[-0.0712, -0.0304, -0.0368],
          [-0.0858, -0.0712, -0.0304],
          [-0.0291, -0.0858, -0.0712]]],


        [[[-0.0351, -0.0295, -0.0539],
          [-0.0799, -0.0351, -0.0295],
          [-0.0315, -0.0799, -0.0351]],

         [[ 0.1251,  0.1077,  0.0939],
          [ 0.1060,  0.1251,  0.1077],
          [ 0.1261,  0.1060,  0.1251]],

         [[-0.0712, -0.0304, -0.0368],
          [-0.0858, -0.0712, -0.0304],
          [-0.0291, -0.0858, -0.0712]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([64, 3, 49, 49])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>mask操作</div>


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>计算相似性权重</div>


: 

```python
attn = self.softmax(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0207, 0.0208, 0.0203],
          [0.0198, 0.0207, 0.0208],
          [0.0208, 0.0198, 0.0207]],

         [[0.0209, 0.0205, 0.0202],
          [0.0205, 0.0208, 0.0205],
          [0.0209, 0.0204, 0.0208]],

         [[0.0200, 0.0208, 0.0207],
          [0.0197, 0.0200, 0.0208],
          [0.0208, 0.0197, 0.0199]]],


        [[[0.0207, 0.0208, 0.0203],
          [0.0198, 0.0207, 0.0208],
          [0.0208, 0.0198, 0.0207]],

         [[0.0209, 0.0205, 0.0202],
          [0.0205, 0.0208, 0.0205],
          [0.0209, 0.0204, 0.0208]],

         [[0.0200, 0.0208, 0.0207],
          [0.0197, 0.0200, 0.0208],
          [0.0208, 0.0197, 0.0199]]],


        [[[0.0207, 0.0208, 0.0203],
          [0.0198, 0.0207, 0.0208],
          [0.0208, 0.0198, 0.0207]],

         [[0.0209, 0.0205, 0.0202],
          [0.0205, 0.0208, 0.0205],
          [0.0209, 0.0204, 0.0208]],

         [[0.0200, 0.0208, 0.0207],
          [0.0197, 0.0200, 0.0208],
          [0.0208, 0.0197, 0.0199]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([64, 3, 49, 49])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>drop操作</div>


: 

```python
attn = self.attn_drop(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0207, 0.0208, 0.0203],
          [0.0198, 0.0207, 0.0208],
          [0.0208, 0.0198, 0.0207]],

         [[0.0209, 0.0205, 0.0202],
          [0.0205, 0.0208, 0.0205],
          [0.0209, 0.0204, 0.0208]],

         [[0.0200, 0.0208, 0.0207],
          [0.0197, 0.0200, 0.0208],
          [0.0208, 0.0197, 0.0199]]],


        [[[0.0207, 0.0208, 0.0203],
          [0.0198, 0.0207, 0.0208],
          [0.0208, 0.0198, 0.0207]],

         [[0.0209, 0.0205, 0.0202],
          [0.0205, 0.0208, 0.0205],
          [0.0209, 0.0204, 0.0208]],

         [[0.0200, 0.0208, 0.0207],
          [0.0197, 0.0200, 0.0208],
          [0.0208, 0.0197, 0.0199]]],


        [[[0.0207, 0.0208, 0.0203],
          [0.0198, 0.0207, 0.0208],
          [0.0208, 0.0198, 0.0207]],

         [[0.0209, 0.0205, 0.0202],
          [0.0205, 0.0208, 0.0205],
          [0.0209, 0.0204, 0.0208]],

         [[0.0200, 0.0208, 0.0207],
          [0.0197, 0.0200, 0.0208],
          [0.0208, 0.0197, 0.0199]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([64, 3, 49, 49])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>相似性得分乘v加堆叠多头操作</div>


: 

```python
x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
```


x[:3,:3,:3]: 

```python
tensor([[[ 0.0747, -0.3110, -0.0579],
         [ 0.0747, -0.3110, -0.0579],
         [ 0.0747, -0.3110, -0.0579]],

        [[ 0.0747, -0.3110, -0.0579],
         [ 0.0747, -0.3110, -0.0579],
         [ 0.0747, -0.3110, -0.0579]],

        [[ 0.0747, -0.3110, -0.0579],
         [ 0.0747, -0.3110, -0.0579],
         [ 0.0747, -0.3110, -0.0579]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([64, 49, 96])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>投影操作</div>


: 

```python
x = self.proj(x)
```


x[:3,:3,:3]: 

```python
tensor([[[-0.0022,  0.0233, -0.0017],
         [-0.0022,  0.0233, -0.0017],
         [-0.0022,  0.0233, -0.0017]],

        [[-0.0022,  0.0233, -0.0017],
         [-0.0022,  0.0233, -0.0017],
         [-0.0022,  0.0233, -0.0017]],

        [[-0.0022,  0.0233, -0.0017],
         [-0.0022,  0.0233, -0.0017],
         [-0.0022,  0.0233, -0.0017]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.proj: 

```python
Linear(in_features=96, out_features=96, bias=True)
```


x.shape: 

```python
torch.Size([64, 49, 96])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>投影dropout操作</div>


: 

```python
x = self.proj(x)
```


self.proj_drop: 

```python
Dropout(p=0.0, inplace=False)
```


x.shape: 

```python
torch.Size([64, 49, 96])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0022,  0.0233, -0.0017],
         [-0.0022,  0.0233, -0.0017],
         [-0.0022,  0.0233, -0.0017]],

        [[-0.0022,  0.0233, -0.0017],
         [-0.0022,  0.0233, -0.0017],
         [-0.0022,  0.0233, -0.0017]],

        [[-0.0022,  0.0233, -0.0017],
         [-0.0022,  0.0233, -0.0017],
         [-0.0022,  0.0233, -0.0017]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([64, 49, 96])
```


### WindowAttention 结束


<div style='color:#ff9702;font-weight:800;font-size:23px;'>W-MSA/SW-MSA</div>


: 

```python
attn_windows = self.attn(x_windows, mask=attn_mask)
```


attn_windows.shape: 

```python
torch.Size([64, 49, 96])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>merge windows（window_reverse）</div>


: 

```python
attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
```


attn_windows.shape: 

```python
torch.Size([64, 7, 7, 96])
```


### window_reverse 开始


<div style='color:#ff9702;font-weight:800;font-size:23px;'>将一个个window还原成一个feature map</div>


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>输入</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017]],

         [[-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017]],

         [[-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017]]],


        [[[-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017]],

         [[-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017]],

         [[-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017]]],


        [[[-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017]],

         [[-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017]],

         [[-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([64, 7, 7, 96])
```


: 

```python
B = int(windows.shape[0] / (H * W / window_size / window_size))
```


B: 

```python
1
```


: 

```python
x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
```


x.shape: 

```python
torch.Size([1, 8, 8, 7, 7, 96])
```


: 

```python
x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
```


x.shape: 

```python
torch.Size([1, 56, 56, 96])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017]],

         [[-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017]],

         [[-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017],
          [-0.0022,  0.0233, -0.0017]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 56, 56, 96])
```


### window_reverse 结束


: 

```python
shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
```


shifted_x.shape: 

```python
torch.Size([1, 56, 56, 96])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>reverse cyclic shift</div>


: 

```python
x = shifted_x
```


x.shape: 

```python
torch.Size([1, 56, 56, 96])
```


: 

```python
x = x.view(B, H * W, C)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>残差连接</div>


: 

```python
x = shortcut + self.drop_path(x)
```


self.drop_path: 

```python
Identity()
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


### Mlp 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.3987, 1.3976, 1.5218],
         [0.3987, 1.3976, 1.5218],
         [0.3987, 1.3976, 1.5218]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


self.fc1: 

```python
Linear(in_features=96, out_features=384, bias=True)
```


self.act: 

```python
GELU()
```


self.drop1: 

```python
Dropout(p=0.0, inplace=False)
```


self.fc2: 

```python
Linear(in_features=384, out_features=96, bias=True)
```


self.drop2: 

```python
Dropout(p=0.0, inplace=False)
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0115, -0.1102, -0.0807],
         [-0.0115, -0.1102, -0.0807],
         [-0.0115, -0.1102, -0.0807]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


### Mlp 结束


: 

```python
x = x + self.drop_path(self.mlp(self.norm2(x)))
```


self.norm2: 

```python
LayerNorm((96,), eps=1e-05, elementwise_affine=True)
```


self.mlp: 

```python
Mlp(
  (fc1): Linear(in_features=96, out_features=384, bias=True)
  (act): GELU()
  (drop1): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=384, out_features=96, bias=True)
  (drop2): Dropout(p=0.0, inplace=False)
)
```


self.drop_path: 

```python
Identity()
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[0.3937, 1.2995, 1.4539],
         [0.3937, 1.2995, 1.4539],
         [0.3937, 1.2995, 1.4539]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


## SwinTransformerBlock 结束


## SwinTransformerBlock 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.3937, 1.2995, 1.4539],
         [0.3937, 1.2995, 1.4539],
         [0.3937, 1.2995, 1.4539]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


attn_mask[:3,:3,:3]: 

```python
tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]], device='cuda:0')
```


attn_mask.shape: 

```python
torch.Size([64, 49, 49])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>保存输入便于做残差连接</div>


: 

```python
shortcut = x
```


shortcut.shape: 

```python
torch.Size([1, 3136, 96])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>norm操作</div>


: 

```python
x = self.norm1(x)
```


self.norm1: 

```python
LayerNorm((96,), eps=1e-05, elementwise_affine=True)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H, W, C)
```


x.shape: 

```python
torch.Size([1, 56, 56, 96])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>把feature map给pad到window size的整数倍</div>


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>cyclic shift</div>


: 

```python
shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
```


self.shift_size: 

```python
3
```


shifted_x.shape: 

```python
torch.Size([1, 56, 56, 96])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>partition windows</div>


### window_partition 开始


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>将feature map按照window_size划分成一个个没有重叠的window</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437]],

         [[0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437]],

         [[0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 56, 56, 96])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
```


x.shape: 

```python
torch.Size([1, 8, 7, 8, 7, 96])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>permute操作</div>


: 

```python
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
```


windows.shape: 

```python
torch.Size([64, 7, 7, 96])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437]],

         [[0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437]],

         [[0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437]]],


        [[[0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437]],

         [[0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437]],

         [[0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437]]],


        [[[0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437]],

         [[0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437]],

         [[0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437],
          [0.3858, 1.2896, 1.4437]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([64, 7, 7, 96])
```


### window_partition 结束


: 

```python
x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
```


x_windows.shape: 

```python
torch.Size([64, 49, 96])
```


### WindowAttention 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.3858, 1.2896, 1.4437],
         [0.3858, 1.2896, 1.4437],
         [0.3858, 1.2896, 1.4437]],

        [[0.3858, 1.2896, 1.4437],
         [0.3858, 1.2896, 1.4437],
         [0.3858, 1.2896, 1.4437]],

        [[0.3858, 1.2896, 1.4437],
         [0.3858, 1.2896, 1.4437],
         [0.3858, 1.2896, 1.4437]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([64, 49, 96])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
```


self.qkv: 

```python
Linear(in_features=96, out_features=288, bias=True)
```


qkv.shape: 

```python
torch.Size([3, 64, 3, 49, 32])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
q, k, v = qkv.unbind(0)
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1294, -0.0326,  0.1055],
          [ 0.1294, -0.0326,  0.1055],
          [ 0.1294, -0.0326,  0.1055]],

         [[-0.0291, -0.1457,  0.2213],
          [-0.0291, -0.1457,  0.2213],
          [-0.0291, -0.1457,  0.2213]],

         [[-0.0774, -0.0259,  0.0038],
          [-0.0774, -0.0259,  0.0038],
          [-0.0774, -0.0259,  0.0038]]],


        [[[ 0.1294, -0.0326,  0.1055],
          [ 0.1294, -0.0326,  0.1055],
          [ 0.1294, -0.0326,  0.1055]],

         [[-0.0291, -0.1457,  0.2213],
          [-0.0291, -0.1457,  0.2213],
          [-0.0291, -0.1457,  0.2213]],

         [[-0.0774, -0.0259,  0.0038],
          [-0.0774, -0.0259,  0.0038],
          [-0.0774, -0.0259,  0.0038]]],


        [[[ 0.1294, -0.0326,  0.1055],
          [ 0.1294, -0.0326,  0.1055],
          [ 0.1294, -0.0326,  0.1055]],

         [[-0.0291, -0.1457,  0.2213],
          [-0.0291, -0.1457,  0.2213],
          [-0.0291, -0.1457,  0.2213]],

         [[-0.0774, -0.0259,  0.0038],
          [-0.0774, -0.0259,  0.0038],
          [-0.0774, -0.0259,  0.0038]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


q.shape: 

```python
torch.Size([64, 3, 49, 32])
```


k[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0908,  0.0743,  0.1606],
          [-0.0908,  0.0743,  0.1606],
          [-0.0908,  0.0743,  0.1606]],

         [[ 0.1237,  0.4767, -0.1842],
          [ 0.1237,  0.4767, -0.1842],
          [ 0.1237,  0.4767, -0.1842]],

         [[-0.1756,  0.1742, -0.4258],
          [-0.1756,  0.1742, -0.4258],
          [-0.1756,  0.1742, -0.4258]]],


        [[[-0.0908,  0.0743,  0.1606],
          [-0.0908,  0.0743,  0.1606],
          [-0.0908,  0.0743,  0.1606]],

         [[ 0.1237,  0.4767, -0.1842],
          [ 0.1237,  0.4767, -0.1842],
          [ 0.1237,  0.4767, -0.1842]],

         [[-0.1756,  0.1742, -0.4258],
          [-0.1756,  0.1742, -0.4258],
          [-0.1756,  0.1742, -0.4258]]],


        [[[-0.0908,  0.0743,  0.1606],
          [-0.0908,  0.0743,  0.1606],
          [-0.0908,  0.0743,  0.1606]],

         [[ 0.1237,  0.4767, -0.1842],
          [ 0.1237,  0.4767, -0.1842],
          [ 0.1237,  0.4767, -0.1842]],

         [[-0.1756,  0.1742, -0.4258],
          [-0.1756,  0.1742, -0.4258],
          [-0.1756,  0.1742, -0.4258]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


k.shape: 

```python
torch.Size([64, 3, 49, 32])
```


v[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0493, -0.1024,  0.0388],
          [ 0.0493, -0.1024,  0.0388],
          [ 0.0493, -0.1024,  0.0388]],

         [[-0.3532, -0.1188, -0.0119],
          [-0.3532, -0.1188, -0.0119],
          [-0.3532, -0.1188, -0.0119]],

         [[ 0.1786, -0.1465, -0.2304],
          [ 0.1786, -0.1465, -0.2304],
          [ 0.1786, -0.1465, -0.2304]]],


        [[[ 0.0493, -0.1024,  0.0388],
          [ 0.0493, -0.1024,  0.0388],
          [ 0.0493, -0.1024,  0.0388]],

         [[-0.3532, -0.1188, -0.0119],
          [-0.3532, -0.1188, -0.0119],
          [-0.3532, -0.1188, -0.0119]],

         [[ 0.1786, -0.1465, -0.2304],
          [ 0.1786, -0.1465, -0.2304],
          [ 0.1786, -0.1465, -0.2304]]],


        [[[ 0.0493, -0.1024,  0.0388],
          [ 0.0493, -0.1024,  0.0388],
          [ 0.0493, -0.1024,  0.0388]],

         [[-0.3532, -0.1188, -0.0119],
          [-0.3532, -0.1188, -0.0119],
          [-0.3532, -0.1188, -0.0119]],

         [[ 0.1786, -0.1465, -0.2304],
          [ 0.1786, -0.1465, -0.2304],
          [ 0.1786, -0.1465, -0.2304]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


v.shape: 

```python
torch.Size([64, 3, 49, 32])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>q = q * self.scale</div>


: 

```python
q = q * self.scale
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0229, -0.0058,  0.0186],
          [ 0.0229, -0.0058,  0.0186],
          [ 0.0229, -0.0058,  0.0186]],

         [[-0.0051, -0.0258,  0.0391],
          [-0.0051, -0.0258,  0.0391],
          [-0.0051, -0.0258,  0.0391]],

         [[-0.0137, -0.0046,  0.0007],
          [-0.0137, -0.0046,  0.0007],
          [-0.0137, -0.0046,  0.0007]]],


        [[[ 0.0229, -0.0058,  0.0186],
          [ 0.0229, -0.0058,  0.0186],
          [ 0.0229, -0.0058,  0.0186]],

         [[-0.0051, -0.0258,  0.0391],
          [-0.0051, -0.0258,  0.0391],
          [-0.0051, -0.0258,  0.0391]],

         [[-0.0137, -0.0046,  0.0007],
          [-0.0137, -0.0046,  0.0007],
          [-0.0137, -0.0046,  0.0007]]],


        [[[ 0.0229, -0.0058,  0.0186],
          [ 0.0229, -0.0058,  0.0186],
          [ 0.0229, -0.0058,  0.0186]],

         [[-0.0051, -0.0258,  0.0391],
          [-0.0051, -0.0258,  0.0391],
          [-0.0051, -0.0258,  0.0391]],

         [[-0.0137, -0.0046,  0.0007],
          [-0.0137, -0.0046,  0.0007],
          [-0.0137, -0.0046,  0.0007]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.scale: 

```python
0.1767766952966369
```


q.shape: 

```python
torch.Size([64, 3, 49, 32])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>计算q和k相似性的得分</div>


: 

```python
attn = (q @ k.transpose(-2, -1))
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0064, -0.0064, -0.0064],
          [-0.0064, -0.0064, -0.0064],
          [-0.0064, -0.0064, -0.0064]],

         [[-0.0577, -0.0577, -0.0577],
          [-0.0577, -0.0577, -0.0577],
          [-0.0577, -0.0577, -0.0577]],

         [[ 0.0326,  0.0326,  0.0326],
          [ 0.0326,  0.0326,  0.0326],
          [ 0.0326,  0.0326,  0.0326]]],


        [[[-0.0064, -0.0064, -0.0064],
          [-0.0064, -0.0064, -0.0064],
          [-0.0064, -0.0064, -0.0064]],

         [[-0.0577, -0.0577, -0.0577],
          [-0.0577, -0.0577, -0.0577],
          [-0.0577, -0.0577, -0.0577]],

         [[ 0.0326,  0.0326,  0.0326],
          [ 0.0326,  0.0326,  0.0326],
          [ 0.0326,  0.0326,  0.0326]]],


        [[[-0.0064, -0.0064, -0.0064],
          [-0.0064, -0.0064, -0.0064],
          [-0.0064, -0.0064, -0.0064]],

         [[-0.0577, -0.0577, -0.0577],
          [-0.0577, -0.0577, -0.0577],
          [-0.0577, -0.0577, -0.0577]],

         [[ 0.0326,  0.0326,  0.0326],
          [ 0.0326,  0.0326,  0.0326],
          [ 0.0326,  0.0326,  0.0326]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([64, 3, 49, 49])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>relative_position_bias</div>


: 

```python
relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
```


relative_position_bias[:3,:3,:3]: 

```python
tensor([[[ 0.0193, -0.0024,  0.0359],
         [-0.0183,  0.0193, -0.0024],
         [-0.0419, -0.0183,  0.0193]],

        [[-0.0247,  0.0240,  0.0181],
         [-0.0356, -0.0247,  0.0240],
         [ 0.0253, -0.0356, -0.0247]],

        [[ 0.0046,  0.0250, -0.0001],
         [ 0.0143,  0.0046,  0.0250],
         [ 0.0289,  0.0143,  0.0046]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


relative_position_bias.shape: 

```python
torch.Size([3, 49, 49])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>attn加上bias操作</div>


: 

```python
attn = attn + relative_position_bias.unsqueeze(0)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0129, -0.0088,  0.0295],
          [-0.0247,  0.0129, -0.0088],
          [-0.0482, -0.0247,  0.0129]],

         [[-0.0824, -0.0337, -0.0396],
          [-0.0933, -0.0824, -0.0337],
          [-0.0324, -0.0933, -0.0824]],

         [[ 0.0373,  0.0576,  0.0325],
          [ 0.0469,  0.0373,  0.0576],
          [ 0.0615,  0.0469,  0.0373]]],


        [[[ 0.0129, -0.0088,  0.0295],
          [-0.0247,  0.0129, -0.0088],
          [-0.0482, -0.0247,  0.0129]],

         [[-0.0824, -0.0337, -0.0396],
          [-0.0933, -0.0824, -0.0337],
          [-0.0324, -0.0933, -0.0824]],

         [[ 0.0373,  0.0576,  0.0325],
          [ 0.0469,  0.0373,  0.0576],
          [ 0.0615,  0.0469,  0.0373]]],


        [[[ 0.0129, -0.0088,  0.0295],
          [-0.0247,  0.0129, -0.0088],
          [-0.0482, -0.0247,  0.0129]],

         [[-0.0824, -0.0337, -0.0396],
          [-0.0933, -0.0824, -0.0337],
          [-0.0324, -0.0933, -0.0824]],

         [[ 0.0373,  0.0576,  0.0325],
          [ 0.0469,  0.0373,  0.0576],
          [ 0.0615,  0.0469,  0.0373]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([64, 3, 49, 49])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>mask操作</div>


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>mask矩阵</div>


mask: 

```python
tensor([[[   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         ...,
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         ...,
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         ...,
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.]],

        ...,

        [[   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         ...,
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         ...,
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         ...,
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.]]], device='cuda:0')
```


mask.shape: 

```python
torch.Size([64, 49, 49])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>mask操作</div>


: 

```python
attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
```


attn[:3,:3,:3,:3,:3]: 

```python
tensor([[[[[ 0.0129, -0.0088,  0.0295],
           [-0.0247,  0.0129, -0.0088],
           [-0.0482, -0.0247,  0.0129]],

          [[-0.0824, -0.0337, -0.0396],
           [-0.0933, -0.0824, -0.0337],
           [-0.0324, -0.0933, -0.0824]],

          [[ 0.0373,  0.0576,  0.0325],
           [ 0.0469,  0.0373,  0.0576],
           [ 0.0615,  0.0469,  0.0373]]],


         [[[ 0.0129, -0.0088,  0.0295],
           [-0.0247,  0.0129, -0.0088],
           [-0.0482, -0.0247,  0.0129]],

          [[-0.0824, -0.0337, -0.0396],
           [-0.0933, -0.0824, -0.0337],
           [-0.0324, -0.0933, -0.0824]],

          [[ 0.0373,  0.0576,  0.0325],
           [ 0.0469,  0.0373,  0.0576],
           [ 0.0615,  0.0469,  0.0373]]],


         [[[ 0.0129, -0.0088,  0.0295],
           [-0.0247,  0.0129, -0.0088],
           [-0.0482, -0.0247,  0.0129]],

          [[-0.0824, -0.0337, -0.0396],
           [-0.0933, -0.0824, -0.0337],
           [-0.0324, -0.0933, -0.0824]],

          [[ 0.0373,  0.0576,  0.0325],
           [ 0.0469,  0.0373,  0.0576],
           [ 0.0615,  0.0469,  0.0373]]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([1, 64, 3, 49, 49])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>attn.view</div>


: 

```python
attn = attn.view(-1, self.num_heads, N, N)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0129, -0.0088,  0.0295],
          [-0.0247,  0.0129, -0.0088],
          [-0.0482, -0.0247,  0.0129]],

         [[-0.0824, -0.0337, -0.0396],
          [-0.0933, -0.0824, -0.0337],
          [-0.0324, -0.0933, -0.0824]],

         [[ 0.0373,  0.0576,  0.0325],
          [ 0.0469,  0.0373,  0.0576],
          [ 0.0615,  0.0469,  0.0373]]],


        [[[ 0.0129, -0.0088,  0.0295],
          [-0.0247,  0.0129, -0.0088],
          [-0.0482, -0.0247,  0.0129]],

         [[-0.0824, -0.0337, -0.0396],
          [-0.0933, -0.0824, -0.0337],
          [-0.0324, -0.0933, -0.0824]],

         [[ 0.0373,  0.0576,  0.0325],
          [ 0.0469,  0.0373,  0.0576],
          [ 0.0615,  0.0469,  0.0373]]],


        [[[ 0.0129, -0.0088,  0.0295],
          [-0.0247,  0.0129, -0.0088],
          [-0.0482, -0.0247,  0.0129]],

         [[-0.0824, -0.0337, -0.0396],
          [-0.0933, -0.0824, -0.0337],
          [-0.0324, -0.0933, -0.0824]],

         [[ 0.0373,  0.0576,  0.0325],
          [ 0.0469,  0.0373,  0.0576],
          [ 0.0615,  0.0469,  0.0373]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([64, 3, 49, 49])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>计算相似性权重</div>


: 

```python
attn = self.softmax(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0208, 0.0204, 0.0212],
          [0.0200, 0.0208, 0.0204],
          [0.0196, 0.0201, 0.0208]],

         [[0.0199, 0.0209, 0.0208],
          [0.0197, 0.0199, 0.0209],
          [0.0209, 0.0196, 0.0199]],

         [[0.0205, 0.0209, 0.0204],
          [0.0207, 0.0205, 0.0209],
          [0.0210, 0.0207, 0.0205]]],


        [[[0.0208, 0.0204, 0.0212],
          [0.0200, 0.0208, 0.0204],
          [0.0196, 0.0201, 0.0208]],

         [[0.0199, 0.0209, 0.0208],
          [0.0197, 0.0199, 0.0209],
          [0.0209, 0.0196, 0.0199]],

         [[0.0205, 0.0209, 0.0204],
          [0.0207, 0.0205, 0.0209],
          [0.0210, 0.0207, 0.0205]]],


        [[[0.0208, 0.0204, 0.0212],
          [0.0200, 0.0208, 0.0204],
          [0.0196, 0.0201, 0.0208]],

         [[0.0199, 0.0209, 0.0208],
          [0.0197, 0.0199, 0.0209],
          [0.0209, 0.0196, 0.0199]],

         [[0.0205, 0.0209, 0.0204],
          [0.0207, 0.0205, 0.0209],
          [0.0210, 0.0207, 0.0205]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([64, 3, 49, 49])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>drop操作</div>


: 

```python
attn = self.attn_drop(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0208, 0.0204, 0.0212],
          [0.0200, 0.0208, 0.0204],
          [0.0196, 0.0201, 0.0208]],

         [[0.0199, 0.0209, 0.0208],
          [0.0197, 0.0199, 0.0209],
          [0.0209, 0.0196, 0.0199]],

         [[0.0205, 0.0209, 0.0204],
          [0.0207, 0.0205, 0.0209],
          [0.0210, 0.0207, 0.0205]]],


        [[[0.0208, 0.0204, 0.0212],
          [0.0200, 0.0208, 0.0204],
          [0.0196, 0.0201, 0.0208]],

         [[0.0199, 0.0209, 0.0208],
          [0.0197, 0.0199, 0.0209],
          [0.0209, 0.0196, 0.0199]],

         [[0.0205, 0.0209, 0.0204],
          [0.0207, 0.0205, 0.0209],
          [0.0210, 0.0207, 0.0205]]],


        [[[0.0208, 0.0204, 0.0212],
          [0.0200, 0.0208, 0.0204],
          [0.0196, 0.0201, 0.0208]],

         [[0.0199, 0.0209, 0.0208],
          [0.0197, 0.0199, 0.0209],
          [0.0209, 0.0196, 0.0199]],

         [[0.0205, 0.0209, 0.0204],
          [0.0207, 0.0205, 0.0209],
          [0.0210, 0.0207, 0.0205]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([64, 3, 49, 49])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>相似性得分乘v加堆叠多头操作</div>


: 

```python
x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
```


x[:3,:3,:3]: 

```python
tensor([[[ 0.0493, -0.1024,  0.0388],
         [ 0.0493, -0.1024,  0.0388],
         [ 0.0493, -0.1024,  0.0388]],

        [[ 0.0493, -0.1024,  0.0388],
         [ 0.0493, -0.1024,  0.0388],
         [ 0.0493, -0.1024,  0.0388]],

        [[ 0.0493, -0.1024,  0.0388],
         [ 0.0493, -0.1024,  0.0388],
         [ 0.0493, -0.1024,  0.0388]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([64, 49, 96])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>投影操作</div>


: 

```python
x = self.proj(x)
```


x[:3,:3,:3]: 

```python
tensor([[[-0.0016, -0.0281,  0.0249],
         [-0.0016, -0.0281,  0.0249],
         [-0.0016, -0.0281,  0.0249]],

        [[-0.0016, -0.0281,  0.0249],
         [-0.0016, -0.0281,  0.0249],
         [-0.0016, -0.0281,  0.0249]],

        [[-0.0016, -0.0281,  0.0249],
         [-0.0016, -0.0281,  0.0249],
         [-0.0016, -0.0281,  0.0249]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.proj: 

```python
Linear(in_features=96, out_features=96, bias=True)
```


x.shape: 

```python
torch.Size([64, 49, 96])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>投影dropout操作</div>


: 

```python
x = self.proj(x)
```


self.proj_drop: 

```python
Dropout(p=0.0, inplace=False)
```


x.shape: 

```python
torch.Size([64, 49, 96])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0016, -0.0281,  0.0249],
         [-0.0016, -0.0281,  0.0249],
         [-0.0016, -0.0281,  0.0249]],

        [[-0.0016, -0.0281,  0.0249],
         [-0.0016, -0.0281,  0.0249],
         [-0.0016, -0.0281,  0.0249]],

        [[-0.0016, -0.0281,  0.0249],
         [-0.0016, -0.0281,  0.0249],
         [-0.0016, -0.0281,  0.0249]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([64, 49, 96])
```


### WindowAttention 结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>W-MSA/SW-MSA</div>


: 

```python
attn_windows = self.attn(x_windows, mask=attn_mask)
```


attn_windows.shape: 

```python
torch.Size([64, 49, 96])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>merge windows（window_reverse）</div>


: 

```python
attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
```


attn_windows.shape: 

```python
torch.Size([64, 7, 7, 96])
```


### window_reverse 开始


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>将一个个window还原成一个feature map</div>


<div style='color:#fe1111;font-weight:800;font-size:23px;'>输入</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249]],

         [[-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249]],

         [[-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249]]],


        [[[-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249]],

         [[-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249]],

         [[-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249]]],


        [[[-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249]],

         [[-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249]],

         [[-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([64, 7, 7, 96])
```


: 

```python
B = int(windows.shape[0] / (H * W / window_size / window_size))
```


B: 

```python
1
```


: 

```python
x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
```


x.shape: 

```python
torch.Size([1, 8, 8, 7, 7, 96])
```


: 

```python
x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
```


x.shape: 

```python
torch.Size([1, 56, 56, 96])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249]],

         [[-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249]],

         [[-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249],
          [-0.0016, -0.0281,  0.0249]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 56, 56, 96])
```


### window_reverse 结束


: 

```python
shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
```


shifted_x.shape: 

```python
torch.Size([1, 56, 56, 96])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>reverse cyclic shift</div>


: 

```python
x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
```


x.shape: 

```python
torch.Size([1, 56, 56, 96])
```


: 

```python
x = x.view(B, H * W, C)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>残差连接</div>


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0016, -0.0281,  0.0249],
         [-0.0016, -0.0281,  0.0249],
         [-0.0016, -0.0281,  0.0249]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0016, -0.0281,  0.0249],
         [-0.0016, -0.0281,  0.0249],
         [-0.0016, -0.0281,  0.0249]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[-0.0016, -0.0284,  0.0251,  ..., -0.0160,  0.0248, -0.0045],
         [-0.0016, -0.0284,  0.0251,  ..., -0.0160,  0.0248, -0.0045],
         [-0.0016, -0.0284,  0.0251,  ..., -0.0160,  0.0248, -0.0045],
         ...,
         [-0.0016, -0.0284,  0.0251,  ..., -0.0160,  0.0248, -0.0045],
         [-0.0016, -0.0284,  0.0251,  ..., -0.0160,  0.0248, -0.0045],
         [-0.0016, -0.0284,  0.0251,  ..., -0.0160,  0.0248, -0.0045]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 3136, 96])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[-0.0016, -0.0284,  0.0251,  ..., -0.0160,  0.0248, -0.0045],
         [-0.0016, -0.0284,  0.0251,  ..., -0.0160,  0.0248, -0.0045],
         [-0.0016, -0.0284,  0.0251,  ..., -0.0160,  0.0248, -0.0045],
         ...,
         [-0.0016, -0.0284,  0.0251,  ..., -0.0160,  0.0248, -0.0045],
         [-0.0016, -0.0284,  0.0251,  ..., -0.0160,  0.0248, -0.0045],
         [-0.0016, -0.0284,  0.0251,  ..., -0.0160,  0.0248, -0.0045]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 3136, 96])
```


### DropPath 结束


: 

```python
x = shortcut + self.drop_path(x)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


### Mlp 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.3869, 1.2679, 1.4762],
         [0.3869, 1.2679, 1.4762],
         [0.3869, 1.2679, 1.4762]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


self.fc1: 

```python
Linear(in_features=96, out_features=384, bias=True)
```


self.act: 

```python
GELU()
```


self.drop1: 

```python
Dropout(p=0.0, inplace=False)
```


self.fc2: 

```python
Linear(in_features=384, out_features=96, bias=True)
```


self.drop2: 

```python
Dropout(p=0.0, inplace=False)
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0734, -0.0014, -0.0010],
         [-0.0734, -0.0014, -0.0010],
         [-0.0734, -0.0014, -0.0010]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


### Mlp 结束


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0734, -0.0014, -0.0010],
         [-0.0734, -0.0014, -0.0010],
         [-0.0734, -0.0014, -0.0010]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0734, -0.0014, -0.0010],
         [-0.0734, -0.0014, -0.0010],
         [-0.0734, -0.0014, -0.0010]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[-0.0740, -0.0014, -0.0010,  ...,  0.0489,  0.0131, -0.0146],
         [-0.0740, -0.0014, -0.0010,  ...,  0.0489,  0.0131, -0.0146],
         [-0.0740, -0.0014, -0.0010,  ...,  0.0489,  0.0131, -0.0146],
         ...,
         [-0.0740, -0.0014, -0.0010,  ...,  0.0489,  0.0131, -0.0146],
         [-0.0740, -0.0014, -0.0010,  ...,  0.0489,  0.0131, -0.0146],
         [-0.0740, -0.0014, -0.0010,  ...,  0.0489,  0.0131, -0.0146]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 3136, 96])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[-0.0740, -0.0014, -0.0010,  ...,  0.0489,  0.0131, -0.0146],
         [-0.0740, -0.0014, -0.0010,  ...,  0.0489,  0.0131, -0.0146],
         [-0.0740, -0.0014, -0.0010,  ...,  0.0489,  0.0131, -0.0146],
         ...,
         [-0.0740, -0.0014, -0.0010,  ...,  0.0489,  0.0131, -0.0146],
         [-0.0740, -0.0014, -0.0010,  ...,  0.0489,  0.0131, -0.0146],
         [-0.0740, -0.0014, -0.0010,  ...,  0.0489,  0.0131, -0.0146]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 3136, 96])
```


### DropPath 结束


: 

```python
x = x + self.drop_path(self.mlp(self.norm2(x)))
```


self.norm2: 

```python
LayerNorm((96,), eps=1e-05, elementwise_affine=True)
```


self.mlp: 

```python
Mlp(
  (fc1): Linear(in_features=96, out_features=384, bias=True)
  (act): GELU()
  (drop1): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=384, out_features=96, bias=True)
  (drop2): Dropout(p=0.0, inplace=False)
)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[0.3180, 1.2698, 1.4780],
         [0.3180, 1.2698, 1.4780],
         [0.3180, 1.2698, 1.4780]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


## SwinTransformerBlock 结束


<div style='color:#fe618e;font-weight:800;font-size:23px;'>END: for blk in self.blocks:</div>


## downsample操作


## PatchMerging 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.3180, 1.2698, 1.4780],
         [0.3180, 1.2698, 1.4780],
         [0.3180, 1.2698, 1.4780]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 3136, 96])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H, W, C)
```


x.shape: 

```python
torch.Size([1, 56, 56, 96])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>pad操作</div>


: 

```python
x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
```


x.shape: 

```python
torch.Size([1, 56, 56, 96])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>PatchMerging取元素操作</div>


: 

```python
x0 = x[:, 0::2, 0::2, :]
```


: 

```python
x1 = x[:, 1::2, 0::2, :]
```


: 

```python
x2 = x[:, 0::2, 1::2, :]
```


: 

```python
x3 = x[:, 1::2, 1::2, :]
```


x0.shape: 

```python
torch.Size([1, 28, 28, 96])
```


x1.shape: 

```python
torch.Size([1, 28, 28, 96])
```


x2.shape: 

```python
torch.Size([1, 28, 28, 96])
```


x3.shape: 

```python
torch.Size([1, 28, 28, 96])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>cat操作</div>


: 

```python
x = torch.cat([x0, x1, x2, x3], -1)
```


x.shape: 

```python
torch.Size([1, 28, 28, 384])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, -1, 4 * C) 
```


x.shape: 

```python
torch.Size([1, 784, 384])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>norm操作</div>


: 

```python
x = self.norm(x)
```


self.norm: 

```python
LayerNorm((384,), eps=1e-05, elementwise_affine=True)
```


x.shape: 

```python
torch.Size([1, 784, 384])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>reduction操作</div>


: 

```python
x = self.reduction(x)
```


self.reduction: 

```python
Linear(in_features=384, out_features=192, bias=False)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0563, -0.7451, -0.7046],
         [ 0.0563, -0.7451, -0.7046],
         [ 0.0563, -0.7451, -0.7046]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


## PatchMerging 结束


<div style='color:#3296ee;font-weight:800;font-size:23px;'>downsample操作</div>


: 

```python
x = self.downsample(x, H, W)
```


self.downsample: 

```python
PatchMerging(
  (reduction): Linear(in_features=384, out_features=192, bias=False)
  (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0563, -0.7451, -0.7046],
         [ 0.0563, -0.7451, -0.7046],
         [ 0.0563, -0.7451, -0.7046]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


H: 

```python
28
```


W: 

```python
28
```


# BasicLayer 结束


# BasicLayer 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>A basic Swin Transformer layer for one stage</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0563, -0.7451, -0.7046],
         [ 0.0563, -0.7451, -0.7046],
         [ 0.0563, -0.7451, -0.7046]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


H: 

```python
28
```


W: 

```python
28
```


## 创建mask：self.create_mask(x, H, W)


### window_partition 开始


<div style='color:#fe1111;font-weight:800;font-size:23px;'>将feature map按照window_size划分成一个个没有重叠的window</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]]]], device='cuda:0')
```


x.shape: 

```python
torch.Size([1, 28, 28, 1])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
```


x.shape: 

```python
torch.Size([1, 4, 7, 4, 7, 1])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>permute操作</div>


: 

```python
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
```


windows.shape: 

```python
torch.Size([16, 7, 7, 1])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]]],


        [[[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]]],


        [[[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]]]], device='cuda:0')
```


windows.shape: 

```python
torch.Size([16, 7, 7, 1])
```


### window_partition 结束


<div style='color:#fe618e;font-weight:800;font-size:23px;'>创建mask</div>


: 

```python
attn_mask = self.create_mask(x, H, W)
```


attn_mask.shape: 

```python
torch.Size([16, 49, 49])
```


## for blk in self.blocks:


<div style='color:#fe618e;font-weight:800;font-size:23px;'>for blk in self.blocks:</div>


核心代码: 

```python
x = blk(x, attn_mask)
```


## SwinTransformerBlock 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0563, -0.7451, -0.7046],
         [ 0.0563, -0.7451, -0.7046],
         [ 0.0563, -0.7451, -0.7046]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


attn_mask[:3,:3,:3]: 

```python
tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]], device='cuda:0')
```


attn_mask.shape: 

```python
torch.Size([16, 49, 49])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>保存输入便于做残差连接</div>


: 

```python
shortcut = x
```


shortcut.shape: 

```python
torch.Size([1, 784, 192])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>norm操作</div>


: 

```python
x = self.norm1(x)
```


self.norm1: 

```python
LayerNorm((192,), eps=1e-05, elementwise_affine=True)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H, W, C)
```


x.shape: 

```python
torch.Size([1, 28, 28, 192])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>把feature map给pad到window size的整数倍</div>


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>cyclic shift</div>


: 

```python
shifted_x = x
```


: 

```python
attn_mask = None
```


shifted_x.shape: 

```python
torch.Size([1, 28, 28, 192])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>partition windows</div>


### window_partition 开始


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>将feature map按照window_size划分成一个个没有重叠的window</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387]],

         [[ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387]],

         [[ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 28, 28, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
```


x.shape: 

```python
torch.Size([1, 4, 7, 4, 7, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>permute操作</div>


: 

```python
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
```


windows.shape: 

```python
torch.Size([16, 7, 7, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387]],

         [[ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387]],

         [[ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387]]],


        [[[ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387]],

         [[ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387]],

         [[ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387]]],


        [[[ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387]],

         [[ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387]],

         [[ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387],
          [ 0.1578, -1.8396, -1.7387]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([16, 7, 7, 192])
```


### window_partition 结束


: 

```python
x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
```


x_windows.shape: 

```python
torch.Size([16, 49, 192])
```


### WindowAttention 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.1578, -1.8396, -1.7387],
         [ 0.1578, -1.8396, -1.7387],
         [ 0.1578, -1.8396, -1.7387]],

        [[ 0.1578, -1.8396, -1.7387],
         [ 0.1578, -1.8396, -1.7387],
         [ 0.1578, -1.8396, -1.7387]],

        [[ 0.1578, -1.8396, -1.7387],
         [ 0.1578, -1.8396, -1.7387],
         [ 0.1578, -1.8396, -1.7387]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([16, 49, 192])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
```


self.qkv: 

```python
Linear(in_features=192, out_features=576, bias=True)
```


qkv.shape: 

```python
torch.Size([3, 16, 6, 49, 32])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
q, k, v = qkv.unbind(0)
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[-0.1054, -0.2612, -0.1439],
          [-0.1054, -0.2612, -0.1439],
          [-0.1054, -0.2612, -0.1439]],

         [[-0.3428, -0.1586,  0.0208],
          [-0.3428, -0.1586,  0.0208],
          [-0.3428, -0.1586,  0.0208]],

         [[-0.2344,  0.2305,  0.0235],
          [-0.2344,  0.2305,  0.0235],
          [-0.2344,  0.2305,  0.0235]]],


        [[[-0.1054, -0.2612, -0.1439],
          [-0.1054, -0.2612, -0.1439],
          [-0.1054, -0.2612, -0.1439]],

         [[-0.3428, -0.1586,  0.0208],
          [-0.3428, -0.1586,  0.0208],
          [-0.3428, -0.1586,  0.0208]],

         [[-0.2344,  0.2305,  0.0235],
          [-0.2344,  0.2305,  0.0235],
          [-0.2344,  0.2305,  0.0235]]],


        [[[-0.1054, -0.2612, -0.1439],
          [-0.1054, -0.2612, -0.1439],
          [-0.1054, -0.2612, -0.1439]],

         [[-0.3428, -0.1586,  0.0208],
          [-0.3428, -0.1586,  0.0208],
          [-0.3428, -0.1586,  0.0208]],

         [[-0.2344,  0.2305,  0.0235],
          [-0.2344,  0.2305,  0.0235],
          [-0.2344,  0.2305,  0.0235]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


q.shape: 

```python
torch.Size([16, 6, 49, 32])
```


k[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.3779, -0.1148,  0.0311],
          [ 0.3779, -0.1148,  0.0311],
          [ 0.3779, -0.1148,  0.0311]],

         [[ 0.3651,  0.5256,  0.0396],
          [ 0.3651,  0.5256,  0.0396],
          [ 0.3651,  0.5256,  0.0396]],

         [[-0.5506, -0.0210,  0.1654],
          [-0.5506, -0.0210,  0.1654],
          [-0.5506, -0.0210,  0.1654]]],


        [[[ 0.3779, -0.1148,  0.0311],
          [ 0.3779, -0.1148,  0.0311],
          [ 0.3779, -0.1148,  0.0311]],

         [[ 0.3651,  0.5256,  0.0396],
          [ 0.3651,  0.5256,  0.0396],
          [ 0.3651,  0.5256,  0.0396]],

         [[-0.5506, -0.0210,  0.1654],
          [-0.5506, -0.0210,  0.1654],
          [-0.5506, -0.0210,  0.1654]]],


        [[[ 0.3779, -0.1148,  0.0311],
          [ 0.3779, -0.1148,  0.0311],
          [ 0.3779, -0.1148,  0.0311]],

         [[ 0.3651,  0.5256,  0.0396],
          [ 0.3651,  0.5256,  0.0396],
          [ 0.3651,  0.5256,  0.0396]],

         [[-0.5506, -0.0210,  0.1654],
          [-0.5506, -0.0210,  0.1654],
          [-0.5506, -0.0210,  0.1654]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


k.shape: 

```python
torch.Size([16, 6, 49, 32])
```


v[:3,:3,:3,:3]: 

```python
tensor([[[[-0.2251, -0.1824, -0.0117],
          [-0.2251, -0.1824, -0.0117],
          [-0.2251, -0.1824, -0.0117]],

         [[ 0.0600, -0.1284, -0.1123],
          [ 0.0600, -0.1284, -0.1123],
          [ 0.0600, -0.1284, -0.1123]],

         [[ 0.1203, -0.3469,  0.0448],
          [ 0.1203, -0.3469,  0.0448],
          [ 0.1203, -0.3469,  0.0448]]],


        [[[-0.2251, -0.1824, -0.0117],
          [-0.2251, -0.1824, -0.0117],
          [-0.2251, -0.1824, -0.0117]],

         [[ 0.0600, -0.1284, -0.1123],
          [ 0.0600, -0.1284, -0.1123],
          [ 0.0600, -0.1284, -0.1123]],

         [[ 0.1203, -0.3469,  0.0448],
          [ 0.1203, -0.3469,  0.0448],
          [ 0.1203, -0.3469,  0.0448]]],


        [[[-0.2251, -0.1824, -0.0117],
          [-0.2251, -0.1824, -0.0117],
          [-0.2251, -0.1824, -0.0117]],

         [[ 0.0600, -0.1284, -0.1123],
          [ 0.0600, -0.1284, -0.1123],
          [ 0.0600, -0.1284, -0.1123]],

         [[ 0.1203, -0.3469,  0.0448],
          [ 0.1203, -0.3469,  0.0448],
          [ 0.1203, -0.3469,  0.0448]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


v.shape: 

```python
torch.Size([16, 6, 49, 32])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>q = q * self.scale</div>


: 

```python
q = q * self.scale
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0186, -0.0462, -0.0254],
          [-0.0186, -0.0462, -0.0254],
          [-0.0186, -0.0462, -0.0254]],

         [[-0.0606, -0.0280,  0.0037],
          [-0.0606, -0.0280,  0.0037],
          [-0.0606, -0.0280,  0.0037]],

         [[-0.0414,  0.0407,  0.0041],
          [-0.0414,  0.0407,  0.0041],
          [-0.0414,  0.0407,  0.0041]]],


        [[[-0.0186, -0.0462, -0.0254],
          [-0.0186, -0.0462, -0.0254],
          [-0.0186, -0.0462, -0.0254]],

         [[-0.0606, -0.0280,  0.0037],
          [-0.0606, -0.0280,  0.0037],
          [-0.0606, -0.0280,  0.0037]],

         [[-0.0414,  0.0407,  0.0041],
          [-0.0414,  0.0407,  0.0041],
          [-0.0414,  0.0407,  0.0041]]],


        [[[-0.0186, -0.0462, -0.0254],
          [-0.0186, -0.0462, -0.0254],
          [-0.0186, -0.0462, -0.0254]],

         [[-0.0606, -0.0280,  0.0037],
          [-0.0606, -0.0280,  0.0037],
          [-0.0606, -0.0280,  0.0037]],

         [[-0.0414,  0.0407,  0.0041],
          [-0.0414,  0.0407,  0.0041],
          [-0.0414,  0.0407,  0.0041]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.scale: 

```python
0.1767766952966369
```


q.shape: 

```python
torch.Size([16, 6, 49, 32])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>计算q和k相似性的得分</div>


: 

```python
attn = (q @ k.transpose(-2, -1))
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0253, -0.0253, -0.0253],
          [-0.0253, -0.0253, -0.0253],
          [-0.0253, -0.0253, -0.0253]],

         [[-0.0847, -0.0847, -0.0847],
          [-0.0847, -0.0847, -0.0847],
          [-0.0847, -0.0847, -0.0847]],

         [[ 0.0250,  0.0250,  0.0250],
          [ 0.0250,  0.0250,  0.0250],
          [ 0.0250,  0.0250,  0.0250]]],


        [[[-0.0253, -0.0253, -0.0253],
          [-0.0253, -0.0253, -0.0253],
          [-0.0253, -0.0253, -0.0253]],

         [[-0.0847, -0.0847, -0.0847],
          [-0.0847, -0.0847, -0.0847],
          [-0.0847, -0.0847, -0.0847]],

         [[ 0.0250,  0.0250,  0.0250],
          [ 0.0250,  0.0250,  0.0250],
          [ 0.0250,  0.0250,  0.0250]]],


        [[[-0.0253, -0.0253, -0.0253],
          [-0.0253, -0.0253, -0.0253],
          [-0.0253, -0.0253, -0.0253]],

         [[-0.0847, -0.0847, -0.0847],
          [-0.0847, -0.0847, -0.0847],
          [-0.0847, -0.0847, -0.0847]],

         [[ 0.0250,  0.0250,  0.0250],
          [ 0.0250,  0.0250,  0.0250],
          [ 0.0250,  0.0250,  0.0250]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([16, 6, 49, 49])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>relative_position_bias</div>


: 

```python
relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
```


relative_position_bias[:3,:3,:3]: 

```python
tensor([[[-0.0136,  0.0021, -0.0001],
         [-0.0231, -0.0136,  0.0021],
         [-0.0142, -0.0231, -0.0136]],

        [[ 0.0008,  0.0049,  0.0136],
         [ 0.0186,  0.0008,  0.0049],
         [ 0.0070,  0.0186,  0.0008]],

        [[-0.0514, -0.0325,  0.0111],
         [-0.0105, -0.0514, -0.0325],
         [ 0.0176, -0.0105, -0.0514]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


relative_position_bias.shape: 

```python
torch.Size([6, 49, 49])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>attn加上bias操作</div>


: 

```python
attn = attn + relative_position_bias.unsqueeze(0)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0388, -0.0232, -0.0254],
          [-0.0484, -0.0388, -0.0232],
          [-0.0395, -0.0484, -0.0388]],

         [[-0.0840, -0.0799, -0.0711],
          [-0.0662, -0.0840, -0.0799],
          [-0.0777, -0.0662, -0.0840]],

         [[-0.0264, -0.0075,  0.0361],
          [ 0.0145, -0.0264, -0.0075],
          [ 0.0426,  0.0145, -0.0264]]],


        [[[-0.0388, -0.0232, -0.0254],
          [-0.0484, -0.0388, -0.0232],
          [-0.0395, -0.0484, -0.0388]],

         [[-0.0840, -0.0799, -0.0711],
          [-0.0662, -0.0840, -0.0799],
          [-0.0777, -0.0662, -0.0840]],

         [[-0.0264, -0.0075,  0.0361],
          [ 0.0145, -0.0264, -0.0075],
          [ 0.0426,  0.0145, -0.0264]]],


        [[[-0.0388, -0.0232, -0.0254],
          [-0.0484, -0.0388, -0.0232],
          [-0.0395, -0.0484, -0.0388]],

         [[-0.0840, -0.0799, -0.0711],
          [-0.0662, -0.0840, -0.0799],
          [-0.0777, -0.0662, -0.0840]],

         [[-0.0264, -0.0075,  0.0361],
          [ 0.0145, -0.0264, -0.0075],
          [ 0.0426,  0.0145, -0.0264]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([16, 6, 49, 49])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>mask操作</div>


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>计算相似性权重</div>


: 

```python
attn = self.softmax(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0201, 0.0204, 0.0204],
          [0.0199, 0.0201, 0.0204],
          [0.0201, 0.0199, 0.0201]],

         [[0.0205, 0.0206, 0.0207],
          [0.0208, 0.0204, 0.0205],
          [0.0205, 0.0208, 0.0204]],

         [[0.0194, 0.0198, 0.0207],
          [0.0202, 0.0194, 0.0198],
          [0.0208, 0.0202, 0.0194]]],


        [[[0.0201, 0.0204, 0.0204],
          [0.0199, 0.0201, 0.0204],
          [0.0201, 0.0199, 0.0201]],

         [[0.0205, 0.0206, 0.0207],
          [0.0208, 0.0204, 0.0205],
          [0.0205, 0.0208, 0.0204]],

         [[0.0194, 0.0198, 0.0207],
          [0.0202, 0.0194, 0.0198],
          [0.0208, 0.0202, 0.0194]]],


        [[[0.0201, 0.0204, 0.0204],
          [0.0199, 0.0201, 0.0204],
          [0.0201, 0.0199, 0.0201]],

         [[0.0205, 0.0206, 0.0207],
          [0.0208, 0.0204, 0.0205],
          [0.0205, 0.0208, 0.0204]],

         [[0.0194, 0.0198, 0.0207],
          [0.0202, 0.0194, 0.0198],
          [0.0208, 0.0202, 0.0194]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([16, 6, 49, 49])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>drop操作</div>


: 

```python
attn = self.attn_drop(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0201, 0.0204, 0.0204],
          [0.0199, 0.0201, 0.0204],
          [0.0201, 0.0199, 0.0201]],

         [[0.0205, 0.0206, 0.0207],
          [0.0208, 0.0204, 0.0205],
          [0.0205, 0.0208, 0.0204]],

         [[0.0194, 0.0198, 0.0207],
          [0.0202, 0.0194, 0.0198],
          [0.0208, 0.0202, 0.0194]]],


        [[[0.0201, 0.0204, 0.0204],
          [0.0199, 0.0201, 0.0204],
          [0.0201, 0.0199, 0.0201]],

         [[0.0205, 0.0206, 0.0207],
          [0.0208, 0.0204, 0.0205],
          [0.0205, 0.0208, 0.0204]],

         [[0.0194, 0.0198, 0.0207],
          [0.0202, 0.0194, 0.0198],
          [0.0208, 0.0202, 0.0194]]],


        [[[0.0201, 0.0204, 0.0204],
          [0.0199, 0.0201, 0.0204],
          [0.0201, 0.0199, 0.0201]],

         [[0.0205, 0.0206, 0.0207],
          [0.0208, 0.0204, 0.0205],
          [0.0205, 0.0208, 0.0204]],

         [[0.0194, 0.0198, 0.0207],
          [0.0202, 0.0194, 0.0198],
          [0.0208, 0.0202, 0.0194]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([16, 6, 49, 49])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>相似性得分乘v加堆叠多头操作</div>


: 

```python
x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
```


x[:3,:3,:3]: 

```python
tensor([[[-0.2251, -0.1824, -0.0117],
         [-0.2251, -0.1824, -0.0117],
         [-0.2251, -0.1824, -0.0117]],

        [[-0.2251, -0.1824, -0.0117],
         [-0.2251, -0.1824, -0.0117],
         [-0.2251, -0.1824, -0.0117]],

        [[-0.2251, -0.1824, -0.0117],
         [-0.2251, -0.1824, -0.0117],
         [-0.2251, -0.1824, -0.0117]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([16, 49, 192])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>投影操作</div>


: 

```python
x = self.proj(x)
```


x[:3,:3,:3]: 

```python
tensor([[[-0.1368,  0.1139, -0.0460],
         [-0.1368,  0.1139, -0.0460],
         [-0.1368,  0.1139, -0.0460]],

        [[-0.1368,  0.1139, -0.0460],
         [-0.1368,  0.1139, -0.0460],
         [-0.1368,  0.1139, -0.0460]],

        [[-0.1368,  0.1139, -0.0460],
         [-0.1368,  0.1139, -0.0460],
         [-0.1368,  0.1139, -0.0460]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.proj: 

```python
Linear(in_features=192, out_features=192, bias=True)
```


x.shape: 

```python
torch.Size([16, 49, 192])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>投影dropout操作</div>


: 

```python
x = self.proj(x)
```


self.proj_drop: 

```python
Dropout(p=0.0, inplace=False)
```


x.shape: 

```python
torch.Size([16, 49, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.1368,  0.1139, -0.0460],
         [-0.1368,  0.1139, -0.0460],
         [-0.1368,  0.1139, -0.0460]],

        [[-0.1368,  0.1139, -0.0460],
         [-0.1368,  0.1139, -0.0460],
         [-0.1368,  0.1139, -0.0460]],

        [[-0.1368,  0.1139, -0.0460],
         [-0.1368,  0.1139, -0.0460],
         [-0.1368,  0.1139, -0.0460]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([16, 49, 192])
```


### WindowAttention 结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>W-MSA/SW-MSA</div>


: 

```python
attn_windows = self.attn(x_windows, mask=attn_mask)
```


attn_windows.shape: 

```python
torch.Size([16, 49, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>merge windows（window_reverse）</div>


: 

```python
attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
```


attn_windows.shape: 

```python
torch.Size([16, 7, 7, 192])
```


### window_reverse 开始


<div style='color:#fe1111;font-weight:800;font-size:23px;'>将一个个window还原成一个feature map</div>


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>输入</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460]],

         [[-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460]],

         [[-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460]]],


        [[[-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460]],

         [[-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460]],

         [[-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460]]],


        [[[-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460]],

         [[-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460]],

         [[-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([16, 7, 7, 192])
```


: 

```python
B = int(windows.shape[0] / (H * W / window_size / window_size))
```


B: 

```python
1
```


: 

```python
x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
```


x.shape: 

```python
torch.Size([1, 4, 4, 7, 7, 192])
```


: 

```python
x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
```


x.shape: 

```python
torch.Size([1, 28, 28, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460]],

         [[-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460]],

         [[-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460],
          [-0.1368,  0.1139, -0.0460]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 28, 28, 192])
```


### window_reverse 结束


: 

```python
shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
```


shifted_x.shape: 

```python
torch.Size([1, 28, 28, 192])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>reverse cyclic shift</div>


: 

```python
x = shifted_x
```


x.shape: 

```python
torch.Size([1, 28, 28, 192])
```


: 

```python
x = x.view(B, H * W, C)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>残差连接</div>


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.1368,  0.1139, -0.0460],
         [-0.1368,  0.1139, -0.0460],
         [-0.1368,  0.1139, -0.0460]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.1368,  0.1139, -0.0460],
         [-0.1368,  0.1139, -0.0460],
         [-0.1368,  0.1139, -0.0460]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[-0.1393,  0.1160, -0.0469,  ..., -0.1009,  0.0090,  0.0886],
         [-0.1393,  0.1160, -0.0469,  ..., -0.1009,  0.0090,  0.0886],
         [-0.1393,  0.1160, -0.0469,  ..., -0.1009,  0.0090,  0.0886],
         ...,
         [-0.1393,  0.1160, -0.0469,  ..., -0.1009,  0.0090,  0.0886],
         [-0.1393,  0.1160, -0.0469,  ..., -0.1009,  0.0090,  0.0886],
         [-0.1393,  0.1160, -0.0469,  ..., -0.1009,  0.0090,  0.0886]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 784, 192])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[-0.1393,  0.1160, -0.0469,  ..., -0.1009,  0.0090,  0.0886],
         [-0.1393,  0.1160, -0.0469,  ..., -0.1009,  0.0090,  0.0886],
         [-0.1393,  0.1160, -0.0469,  ..., -0.1009,  0.0090,  0.0886],
         ...,
         [-0.1393,  0.1160, -0.0469,  ..., -0.1009,  0.0090,  0.0886],
         [-0.1393,  0.1160, -0.0469,  ..., -0.1009,  0.0090,  0.0886],
         [-0.1393,  0.1160, -0.0469,  ..., -0.1009,  0.0090,  0.0886]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 784, 192])
```


### DropPath 结束


: 

```python
x = shortcut + self.drop_path(x)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


### Mlp 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.1520, -1.5018, -1.8045],
         [-0.1520, -1.5018, -1.8045],
         [-0.1520, -1.5018, -1.8045]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


self.fc1: 

```python
Linear(in_features=192, out_features=768, bias=True)
```


self.act: 

```python
GELU()
```


self.drop1: 

```python
Dropout(p=0.0, inplace=False)
```


self.fc2: 

```python
Linear(in_features=768, out_features=192, bias=True)
```


self.drop2: 

```python
Dropout(p=0.0, inplace=False)
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0894,  0.0353, -0.0769],
         [-0.0894,  0.0353, -0.0769],
         [-0.0894,  0.0353, -0.0769]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


### Mlp 结束


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0894,  0.0353, -0.0769],
         [-0.0894,  0.0353, -0.0769],
         [-0.0894,  0.0353, -0.0769]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0894,  0.0353, -0.0769],
         [-0.0894,  0.0353, -0.0769],
         [-0.0894,  0.0353, -0.0769]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[-0.0910,  0.0359, -0.0783,  ..., -0.0243, -0.0593, -0.0248],
         [-0.0910,  0.0359, -0.0783,  ..., -0.0243, -0.0593, -0.0248],
         [-0.0910,  0.0359, -0.0783,  ..., -0.0243, -0.0593, -0.0248],
         ...,
         [-0.0910,  0.0359, -0.0783,  ..., -0.0243, -0.0593, -0.0248],
         [-0.0910,  0.0359, -0.0783,  ..., -0.0243, -0.0593, -0.0248],
         [-0.0910,  0.0359, -0.0783,  ..., -0.0243, -0.0593, -0.0248]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 784, 192])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[-0.0910,  0.0359, -0.0783,  ..., -0.0243, -0.0593, -0.0248],
         [-0.0910,  0.0359, -0.0783,  ..., -0.0243, -0.0593, -0.0248],
         [-0.0910,  0.0359, -0.0783,  ..., -0.0243, -0.0593, -0.0248],
         ...,
         [-0.0910,  0.0359, -0.0783,  ..., -0.0243, -0.0593, -0.0248],
         [-0.0910,  0.0359, -0.0783,  ..., -0.0243, -0.0593, -0.0248],
         [-0.0910,  0.0359, -0.0783,  ..., -0.0243, -0.0593, -0.0248]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 784, 192])
```


### DropPath 结束


: 

```python
x = x + self.drop_path(self.mlp(self.norm2(x)))
```


self.norm2: 

```python
LayerNorm((192,), eps=1e-05, elementwise_affine=True)
```


self.mlp: 

```python
Mlp(
  (fc1): Linear(in_features=192, out_features=768, bias=True)
  (act): GELU()
  (drop1): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=768, out_features=192, bias=True)
  (drop2): Dropout(p=0.0, inplace=False)
)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.1741, -0.5931, -0.8298],
         [-0.1741, -0.5931, -0.8298],
         [-0.1741, -0.5931, -0.8298]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


## SwinTransformerBlock 结束


## SwinTransformerBlock 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.1741, -0.5931, -0.8298],
         [-0.1741, -0.5931, -0.8298],
         [-0.1741, -0.5931, -0.8298]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


attn_mask[:3,:3,:3]: 

```python
tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]], device='cuda:0')
```


attn_mask.shape: 

```python
torch.Size([16, 49, 49])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>保存输入便于做残差连接</div>


: 

```python
shortcut = x
```


shortcut.shape: 

```python
torch.Size([1, 784, 192])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>norm操作</div>


: 

```python
x = self.norm1(x)
```


self.norm1: 

```python
LayerNorm((192,), eps=1e-05, elementwise_affine=True)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H, W, C)
```


x.shape: 

```python
torch.Size([1, 28, 28, 192])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>把feature map给pad到window size的整数倍</div>


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>cyclic shift</div>


: 

```python
shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
```


self.shift_size: 

```python
3
```


shifted_x.shape: 

```python
torch.Size([1, 28, 28, 192])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>partition windows</div>


### window_partition 开始


<div style='color:#fd7949;font-weight:800;font-size:23px;'>将feature map按照window_size划分成一个个没有重叠的window</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742]],

         [[-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742]],

         [[-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 28, 28, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
```


x.shape: 

```python
torch.Size([1, 4, 7, 4, 7, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>permute操作</div>


: 

```python
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
```


windows.shape: 

```python
torch.Size([16, 7, 7, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742]],

         [[-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742]],

         [[-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742]]],


        [[[-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742]],

         [[-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742]],

         [[-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742]]],


        [[[-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742]],

         [[-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742]],

         [[-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742],
          [-0.3998, -1.4059, -1.9742]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([16, 7, 7, 192])
```


### window_partition 结束


: 

```python
x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
```


x_windows.shape: 

```python
torch.Size([16, 49, 192])
```


### WindowAttention 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.3998, -1.4059, -1.9742],
         [-0.3998, -1.4059, -1.9742],
         [-0.3998, -1.4059, -1.9742]],

        [[-0.3998, -1.4059, -1.9742],
         [-0.3998, -1.4059, -1.9742],
         [-0.3998, -1.4059, -1.9742]],

        [[-0.3998, -1.4059, -1.9742],
         [-0.3998, -1.4059, -1.9742],
         [-0.3998, -1.4059, -1.9742]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([16, 49, 192])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
```


self.qkv: 

```python
Linear(in_features=192, out_features=576, bias=True)
```


qkv.shape: 

```python
torch.Size([3, 16, 6, 49, 32])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
q, k, v = qkv.unbind(0)
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.3761,  0.5027, -0.2528],
          [ 0.3761,  0.5027, -0.2528],
          [ 0.3761,  0.5027, -0.2528]],

         [[-0.3774, -0.3749, -0.1730],
          [-0.3774, -0.3749, -0.1730],
          [-0.3774, -0.3749, -0.1730]],

         [[ 0.1609,  0.5037, -0.3075],
          [ 0.1609,  0.5037, -0.3075],
          [ 0.1609,  0.5037, -0.3075]]],


        [[[ 0.3761,  0.5027, -0.2528],
          [ 0.3761,  0.5027, -0.2528],
          [ 0.3761,  0.5027, -0.2528]],

         [[-0.3774, -0.3749, -0.1730],
          [-0.3774, -0.3749, -0.1730],
          [-0.3774, -0.3749, -0.1730]],

         [[ 0.1609,  0.5037, -0.3075],
          [ 0.1609,  0.5037, -0.3075],
          [ 0.1609,  0.5037, -0.3075]]],


        [[[ 0.3761,  0.5027, -0.2528],
          [ 0.3761,  0.5027, -0.2528],
          [ 0.3761,  0.5027, -0.2528]],

         [[-0.3774, -0.3749, -0.1730],
          [-0.3774, -0.3749, -0.1730],
          [-0.3774, -0.3749, -0.1730]],

         [[ 0.1609,  0.5037, -0.3075],
          [ 0.1609,  0.5037, -0.3075],
          [ 0.1609,  0.5037, -0.3075]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


q.shape: 

```python
torch.Size([16, 6, 49, 32])
```


k[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.2849, -0.2789,  0.1036],
          [ 0.2849, -0.2789,  0.1036],
          [ 0.2849, -0.2789,  0.1036]],

         [[-0.4890,  0.0651,  0.4683],
          [-0.4890,  0.0651,  0.4683],
          [-0.4890,  0.0651,  0.4683]],

         [[-0.2875, -0.0900, -0.1241],
          [-0.2875, -0.0900, -0.1241],
          [-0.2875, -0.0900, -0.1241]]],


        [[[ 0.2849, -0.2789,  0.1036],
          [ 0.2849, -0.2789,  0.1036],
          [ 0.2849, -0.2789,  0.1036]],

         [[-0.4890,  0.0651,  0.4683],
          [-0.4890,  0.0651,  0.4683],
          [-0.4890,  0.0651,  0.4683]],

         [[-0.2875, -0.0900, -0.1241],
          [-0.2875, -0.0900, -0.1241],
          [-0.2875, -0.0900, -0.1241]]],


        [[[ 0.2849, -0.2789,  0.1036],
          [ 0.2849, -0.2789,  0.1036],
          [ 0.2849, -0.2789,  0.1036]],

         [[-0.4890,  0.0651,  0.4683],
          [-0.4890,  0.0651,  0.4683],
          [-0.4890,  0.0651,  0.4683]],

         [[-0.2875, -0.0900, -0.1241],
          [-0.2875, -0.0900, -0.1241],
          [-0.2875, -0.0900, -0.1241]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


k.shape: 

```python
torch.Size([16, 6, 49, 32])
```


v[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.4360, -0.2199, -0.2909],
          [ 0.4360, -0.2199, -0.2909],
          [ 0.4360, -0.2199, -0.2909]],

         [[-0.6888, -0.3103,  0.1408],
          [-0.6888, -0.3103,  0.1408],
          [-0.6888, -0.3103,  0.1408]],

         [[ 0.0746,  0.5033, -0.2764],
          [ 0.0746,  0.5033, -0.2764],
          [ 0.0746,  0.5033, -0.2764]]],


        [[[ 0.4360, -0.2199, -0.2909],
          [ 0.4360, -0.2199, -0.2909],
          [ 0.4360, -0.2199, -0.2909]],

         [[-0.6888, -0.3103,  0.1408],
          [-0.6888, -0.3103,  0.1408],
          [-0.6888, -0.3103,  0.1408]],

         [[ 0.0746,  0.5033, -0.2764],
          [ 0.0746,  0.5033, -0.2764],
          [ 0.0746,  0.5033, -0.2764]]],


        [[[ 0.4360, -0.2199, -0.2909],
          [ 0.4360, -0.2199, -0.2909],
          [ 0.4360, -0.2199, -0.2909]],

         [[-0.6888, -0.3103,  0.1408],
          [-0.6888, -0.3103,  0.1408],
          [-0.6888, -0.3103,  0.1408]],

         [[ 0.0746,  0.5033, -0.2764],
          [ 0.0746,  0.5033, -0.2764],
          [ 0.0746,  0.5033, -0.2764]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


v.shape: 

```python
torch.Size([16, 6, 49, 32])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>q = q * self.scale</div>


: 

```python
q = q * self.scale
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0665,  0.0889, -0.0447],
          [ 0.0665,  0.0889, -0.0447],
          [ 0.0665,  0.0889, -0.0447]],

         [[-0.0667, -0.0663, -0.0306],
          [-0.0667, -0.0663, -0.0306],
          [-0.0667, -0.0663, -0.0306]],

         [[ 0.0284,  0.0890, -0.0544],
          [ 0.0284,  0.0890, -0.0544],
          [ 0.0284,  0.0890, -0.0544]]],


        [[[ 0.0665,  0.0889, -0.0447],
          [ 0.0665,  0.0889, -0.0447],
          [ 0.0665,  0.0889, -0.0447]],

         [[-0.0667, -0.0663, -0.0306],
          [-0.0667, -0.0663, -0.0306],
          [-0.0667, -0.0663, -0.0306]],

         [[ 0.0284,  0.0890, -0.0544],
          [ 0.0284,  0.0890, -0.0544],
          [ 0.0284,  0.0890, -0.0544]]],


        [[[ 0.0665,  0.0889, -0.0447],
          [ 0.0665,  0.0889, -0.0447],
          [ 0.0665,  0.0889, -0.0447]],

         [[-0.0667, -0.0663, -0.0306],
          [-0.0667, -0.0663, -0.0306],
          [-0.0667, -0.0663, -0.0306]],

         [[ 0.0284,  0.0890, -0.0544],
          [ 0.0284,  0.0890, -0.0544],
          [ 0.0284,  0.0890, -0.0544]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.scale: 

```python
0.1767766952966369
```


q.shape: 

```python
torch.Size([16, 6, 49, 32])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>计算q和k相似性的得分</div>


: 

```python
attn = (q @ k.transpose(-2, -1))
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0142, -0.0142, -0.0142],
          [-0.0142, -0.0142, -0.0142],
          [-0.0142, -0.0142, -0.0142]],

         [[-0.0700, -0.0700, -0.0700],
          [-0.0700, -0.0700, -0.0700],
          [-0.0700, -0.0700, -0.0700]],

         [[ 0.0225,  0.0225,  0.0225],
          [ 0.0225,  0.0225,  0.0225],
          [ 0.0225,  0.0225,  0.0225]]],


        [[[-0.0142, -0.0142, -0.0142],
          [-0.0142, -0.0142, -0.0142],
          [-0.0142, -0.0142, -0.0142]],

         [[-0.0700, -0.0700, -0.0700],
          [-0.0700, -0.0700, -0.0700],
          [-0.0700, -0.0700, -0.0700]],

         [[ 0.0225,  0.0225,  0.0225],
          [ 0.0225,  0.0225,  0.0225],
          [ 0.0225,  0.0225,  0.0225]]],


        [[[-0.0142, -0.0142, -0.0142],
          [-0.0142, -0.0142, -0.0142],
          [-0.0142, -0.0142, -0.0142]],

         [[-0.0700, -0.0700, -0.0700],
          [-0.0700, -0.0700, -0.0700],
          [-0.0700, -0.0700, -0.0700]],

         [[ 0.0225,  0.0225,  0.0225],
          [ 0.0225,  0.0225,  0.0225],
          [ 0.0225,  0.0225,  0.0225]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([16, 6, 49, 49])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>relative_position_bias</div>


: 

```python
relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
```


relative_position_bias[:3,:3,:3]: 

```python
tensor([[[ 0.0265,  0.0029,  0.0123],
         [ 0.0065,  0.0265,  0.0029],
         [ 0.0046,  0.0065,  0.0265]],

        [[ 0.0220, -0.0063, -0.0022],
         [-0.0224,  0.0220, -0.0063],
         [-0.0047, -0.0224,  0.0220]],

        [[-0.0277,  0.0082, -0.0219],
         [ 0.0124, -0.0277,  0.0082],
         [-0.0017,  0.0124, -0.0277]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


relative_position_bias.shape: 

```python
torch.Size([6, 49, 49])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>attn加上bias操作</div>


: 

```python
attn = attn + relative_position_bias.unsqueeze(0)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0123, -0.0113, -0.0020],
          [-0.0078,  0.0123, -0.0113],
          [-0.0096, -0.0078,  0.0123]],

         [[-0.0479, -0.0763, -0.0722],
          [-0.0923, -0.0479, -0.0763],
          [-0.0747, -0.0923, -0.0479]],

         [[-0.0051,  0.0308,  0.0006],
          [ 0.0349, -0.0051,  0.0308],
          [ 0.0209,  0.0349, -0.0051]]],


        [[[ 0.0123, -0.0113, -0.0020],
          [-0.0078,  0.0123, -0.0113],
          [-0.0096, -0.0078,  0.0123]],

         [[-0.0479, -0.0763, -0.0722],
          [-0.0923, -0.0479, -0.0763],
          [-0.0747, -0.0923, -0.0479]],

         [[-0.0051,  0.0308,  0.0006],
          [ 0.0349, -0.0051,  0.0308],
          [ 0.0209,  0.0349, -0.0051]]],


        [[[ 0.0123, -0.0113, -0.0020],
          [-0.0078,  0.0123, -0.0113],
          [-0.0096, -0.0078,  0.0123]],

         [[-0.0479, -0.0763, -0.0722],
          [-0.0923, -0.0479, -0.0763],
          [-0.0747, -0.0923, -0.0479]],

         [[-0.0051,  0.0308,  0.0006],
          [ 0.0349, -0.0051,  0.0308],
          [ 0.0209,  0.0349, -0.0051]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([16, 6, 49, 49])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>mask操作</div>


<div style='color:#3296ee;font-weight:800;font-size:23px;'>mask矩阵</div>


mask: 

```python
tensor([[[   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         ...,
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         ...,
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         ...,
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.]],

        ...,

        [[   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         ...,
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         ...,
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         ...,
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.]]], device='cuda:0')
```


mask.shape: 

```python
torch.Size([16, 49, 49])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>mask操作</div>


: 

```python
attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
```


attn[:3,:3,:3,:3,:3]: 

```python
tensor([[[[[ 0.0123, -0.0113, -0.0020],
           [-0.0078,  0.0123, -0.0113],
           [-0.0096, -0.0078,  0.0123]],

          [[-0.0479, -0.0763, -0.0722],
           [-0.0923, -0.0479, -0.0763],
           [-0.0747, -0.0923, -0.0479]],

          [[-0.0051,  0.0308,  0.0006],
           [ 0.0349, -0.0051,  0.0308],
           [ 0.0209,  0.0349, -0.0051]]],


         [[[ 0.0123, -0.0113, -0.0020],
           [-0.0078,  0.0123, -0.0113],
           [-0.0096, -0.0078,  0.0123]],

          [[-0.0479, -0.0763, -0.0722],
           [-0.0923, -0.0479, -0.0763],
           [-0.0747, -0.0923, -0.0479]],

          [[-0.0051,  0.0308,  0.0006],
           [ 0.0349, -0.0051,  0.0308],
           [ 0.0209,  0.0349, -0.0051]]],


         [[[ 0.0123, -0.0113, -0.0020],
           [-0.0078,  0.0123, -0.0113],
           [-0.0096, -0.0078,  0.0123]],

          [[-0.0479, -0.0763, -0.0722],
           [-0.0923, -0.0479, -0.0763],
           [-0.0747, -0.0923, -0.0479]],

          [[-0.0051,  0.0308,  0.0006],
           [ 0.0349, -0.0051,  0.0308],
           [ 0.0209,  0.0349, -0.0051]]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([1, 16, 6, 49, 49])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>attn.view</div>


: 

```python
attn = attn.view(-1, self.num_heads, N, N)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0123, -0.0113, -0.0020],
          [-0.0078,  0.0123, -0.0113],
          [-0.0096, -0.0078,  0.0123]],

         [[-0.0479, -0.0763, -0.0722],
          [-0.0923, -0.0479, -0.0763],
          [-0.0747, -0.0923, -0.0479]],

         [[-0.0051,  0.0308,  0.0006],
          [ 0.0349, -0.0051,  0.0308],
          [ 0.0209,  0.0349, -0.0051]]],


        [[[ 0.0123, -0.0113, -0.0020],
          [-0.0078,  0.0123, -0.0113],
          [-0.0096, -0.0078,  0.0123]],

         [[-0.0479, -0.0763, -0.0722],
          [-0.0923, -0.0479, -0.0763],
          [-0.0747, -0.0923, -0.0479]],

         [[-0.0051,  0.0308,  0.0006],
          [ 0.0349, -0.0051,  0.0308],
          [ 0.0209,  0.0349, -0.0051]]],


        [[[ 0.0123, -0.0113, -0.0020],
          [-0.0078,  0.0123, -0.0113],
          [-0.0096, -0.0078,  0.0123]],

         [[-0.0479, -0.0763, -0.0722],
          [-0.0923, -0.0479, -0.0763],
          [-0.0747, -0.0923, -0.0479]],

         [[-0.0051,  0.0308,  0.0006],
          [ 0.0349, -0.0051,  0.0308],
          [ 0.0209,  0.0349, -0.0051]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([16, 6, 49, 49])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>计算相似性权重</div>


: 

```python
attn = self.softmax(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0209, 0.0204, 0.0206],
          [0.0205, 0.0209, 0.0204],
          [0.0204, 0.0204, 0.0208]],

         [[0.0208, 0.0202, 0.0203],
          [0.0199, 0.0208, 0.0202],
          [0.0203, 0.0199, 0.0208]],

         [[0.0199, 0.0206, 0.0200],
          [0.0207, 0.0199, 0.0206],
          [0.0204, 0.0207, 0.0199]]],


        [[[0.0209, 0.0204, 0.0206],
          [0.0205, 0.0209, 0.0204],
          [0.0204, 0.0204, 0.0208]],

         [[0.0208, 0.0202, 0.0203],
          [0.0199, 0.0208, 0.0202],
          [0.0203, 0.0199, 0.0208]],

         [[0.0199, 0.0206, 0.0200],
          [0.0207, 0.0199, 0.0206],
          [0.0204, 0.0207, 0.0199]]],


        [[[0.0209, 0.0204, 0.0206],
          [0.0205, 0.0209, 0.0204],
          [0.0204, 0.0204, 0.0208]],

         [[0.0208, 0.0202, 0.0203],
          [0.0199, 0.0208, 0.0202],
          [0.0203, 0.0199, 0.0208]],

         [[0.0199, 0.0206, 0.0200],
          [0.0207, 0.0199, 0.0206],
          [0.0204, 0.0207, 0.0199]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([16, 6, 49, 49])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>drop操作</div>


: 

```python
attn = self.attn_drop(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0209, 0.0204, 0.0206],
          [0.0205, 0.0209, 0.0204],
          [0.0204, 0.0204, 0.0208]],

         [[0.0208, 0.0202, 0.0203],
          [0.0199, 0.0208, 0.0202],
          [0.0203, 0.0199, 0.0208]],

         [[0.0199, 0.0206, 0.0200],
          [0.0207, 0.0199, 0.0206],
          [0.0204, 0.0207, 0.0199]]],


        [[[0.0209, 0.0204, 0.0206],
          [0.0205, 0.0209, 0.0204],
          [0.0204, 0.0204, 0.0208]],

         [[0.0208, 0.0202, 0.0203],
          [0.0199, 0.0208, 0.0202],
          [0.0203, 0.0199, 0.0208]],

         [[0.0199, 0.0206, 0.0200],
          [0.0207, 0.0199, 0.0206],
          [0.0204, 0.0207, 0.0199]]],


        [[[0.0209, 0.0204, 0.0206],
          [0.0205, 0.0209, 0.0204],
          [0.0204, 0.0204, 0.0208]],

         [[0.0208, 0.0202, 0.0203],
          [0.0199, 0.0208, 0.0202],
          [0.0203, 0.0199, 0.0208]],

         [[0.0199, 0.0206, 0.0200],
          [0.0207, 0.0199, 0.0206],
          [0.0204, 0.0207, 0.0199]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([16, 6, 49, 49])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>相似性得分乘v加堆叠多头操作</div>


: 

```python
x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
```


x[:3,:3,:3]: 

```python
tensor([[[ 0.4360, -0.2199, -0.2909],
         [ 0.4360, -0.2199, -0.2909],
         [ 0.4360, -0.2199, -0.2909]],

        [[ 0.4360, -0.2199, -0.2909],
         [ 0.4360, -0.2199, -0.2909],
         [ 0.4360, -0.2199, -0.2909]],

        [[ 0.4360, -0.2199, -0.2909],
         [ 0.4360, -0.2199, -0.2909],
         [ 0.4360, -0.2199, -0.2909]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([16, 49, 192])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>投影操作</div>


: 

```python
x = self.proj(x)
```


x[:3,:3,:3]: 

```python
tensor([[[-0.0328, -0.0917, -0.1888],
         [-0.0328, -0.0917, -0.1888],
         [-0.0328, -0.0917, -0.1888]],

        [[-0.0328, -0.0917, -0.1888],
         [-0.0328, -0.0917, -0.1888],
         [-0.0328, -0.0917, -0.1888]],

        [[-0.0328, -0.0917, -0.1888],
         [-0.0328, -0.0917, -0.1888],
         [-0.0328, -0.0917, -0.1888]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.proj: 

```python
Linear(in_features=192, out_features=192, bias=True)
```


x.shape: 

```python
torch.Size([16, 49, 192])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>投影dropout操作</div>


: 

```python
x = self.proj(x)
```


self.proj_drop: 

```python
Dropout(p=0.0, inplace=False)
```


x.shape: 

```python
torch.Size([16, 49, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0328, -0.0917, -0.1888],
         [-0.0328, -0.0917, -0.1888],
         [-0.0328, -0.0917, -0.1888]],

        [[-0.0328, -0.0917, -0.1888],
         [-0.0328, -0.0917, -0.1888],
         [-0.0328, -0.0917, -0.1888]],

        [[-0.0328, -0.0917, -0.1888],
         [-0.0328, -0.0917, -0.1888],
         [-0.0328, -0.0917, -0.1888]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([16, 49, 192])
```


### WindowAttention 结束


<div style='color:#3296ee;font-weight:800;font-size:23px;'>W-MSA/SW-MSA</div>


: 

```python
attn_windows = self.attn(x_windows, mask=attn_mask)
```


attn_windows.shape: 

```python
torch.Size([16, 49, 192])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>merge windows（window_reverse）</div>


: 

```python
attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
```


attn_windows.shape: 

```python
torch.Size([16, 7, 7, 192])
```


### window_reverse 开始


<div style='color:#3296ee;font-weight:800;font-size:23px;'>将一个个window还原成一个feature map</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888]],

         [[-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888]],

         [[-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888]]],


        [[[-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888]],

         [[-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888]],

         [[-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888]]],


        [[[-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888]],

         [[-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888]],

         [[-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([16, 7, 7, 192])
```


: 

```python
B = int(windows.shape[0] / (H * W / window_size / window_size))
```


B: 

```python
1
```


: 

```python
x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
```


x.shape: 

```python
torch.Size([1, 4, 4, 7, 7, 192])
```


: 

```python
x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
```


x.shape: 

```python
torch.Size([1, 28, 28, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888]],

         [[-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888]],

         [[-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888],
          [-0.0328, -0.0917, -0.1888]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 28, 28, 192])
```


### window_reverse 结束


: 

```python
shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
```


shifted_x.shape: 

```python
torch.Size([1, 28, 28, 192])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>reverse cyclic shift</div>


: 

```python
x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
```


x.shape: 

```python
torch.Size([1, 28, 28, 192])
```


: 

```python
x = x.view(B, H * W, C)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>残差连接</div>


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0328, -0.0917, -0.1888],
         [-0.0328, -0.0917, -0.1888],
         [-0.0328, -0.0917, -0.1888]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0328, -0.0917, -0.1888],
         [-0.0328, -0.0917, -0.1888],
         [-0.0328, -0.0917, -0.1888]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[-0.0337, -0.0943, -0.1941,  ..., -0.1870,  0.1491, -0.1230],
         [-0.0337, -0.0943, -0.1941,  ..., -0.1870,  0.1491, -0.1230],
         [-0.0337, -0.0943, -0.1941,  ..., -0.1870,  0.1491, -0.1230],
         ...,
         [-0.0337, -0.0943, -0.1941,  ..., -0.1870,  0.1491, -0.1230],
         [-0.0337, -0.0943, -0.1941,  ..., -0.1870,  0.1491, -0.1230],
         [-0.0337, -0.0943, -0.1941,  ..., -0.1870,  0.1491, -0.1230]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 784, 192])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[-0.0337, -0.0943, -0.1941,  ..., -0.1870,  0.1491, -0.1230],
         [-0.0337, -0.0943, -0.1941,  ..., -0.1870,  0.1491, -0.1230],
         [-0.0337, -0.0943, -0.1941,  ..., -0.1870,  0.1491, -0.1230],
         ...,
         [-0.0337, -0.0943, -0.1941,  ..., -0.1870,  0.1491, -0.1230],
         [-0.0337, -0.0943, -0.1941,  ..., -0.1870,  0.1491, -0.1230],
         [-0.0337, -0.0943, -0.1941,  ..., -0.1870,  0.1491, -0.1230]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 784, 192])
```


### DropPath 结束


: 

```python
x = shortcut + self.drop_path(x)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


### Mlp 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.4509, -1.5654, -2.3473],
         [-0.4509, -1.5654, -2.3473],
         [-0.4509, -1.5654, -2.3473]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


self.fc1: 

```python
Linear(in_features=192, out_features=768, bias=True)
```


self.act: 

```python
GELU()
```


self.drop1: 

```python
Dropout(p=0.0, inplace=False)
```


self.fc2: 

```python
Linear(in_features=768, out_features=192, bias=True)
```


self.drop2: 

```python
Dropout(p=0.0, inplace=False)
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0621, -0.0134,  0.0485],
         [-0.0621, -0.0134,  0.0485],
         [-0.0621, -0.0134,  0.0485]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


### Mlp 结束


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0621, -0.0134,  0.0485],
         [-0.0621, -0.0134,  0.0485],
         [-0.0621, -0.0134,  0.0485]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0621, -0.0134,  0.0485],
         [-0.0621, -0.0134,  0.0485],
         [-0.0621, -0.0134,  0.0485]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[-0.0638, -0.0138,  0.0499,  ..., -0.0590,  0.1325, -0.0435],
         [-0.0638, -0.0138,  0.0499,  ..., -0.0590,  0.1325, -0.0435],
         [-0.0638, -0.0138,  0.0499,  ..., -0.0590,  0.1325, -0.0435],
         ...,
         [-0.0638, -0.0138,  0.0499,  ..., -0.0590,  0.1325, -0.0435],
         [-0.0638, -0.0138,  0.0499,  ..., -0.0590,  0.1325, -0.0435],
         [-0.0638, -0.0138,  0.0499,  ..., -0.0590,  0.1325, -0.0435]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 784, 192])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[-0.0638, -0.0138,  0.0499,  ..., -0.0590,  0.1325, -0.0435],
         [-0.0638, -0.0138,  0.0499,  ..., -0.0590,  0.1325, -0.0435],
         [-0.0638, -0.0138,  0.0499,  ..., -0.0590,  0.1325, -0.0435],
         ...,
         [-0.0638, -0.0138,  0.0499,  ..., -0.0590,  0.1325, -0.0435],
         [-0.0638, -0.0138,  0.0499,  ..., -0.0590,  0.1325, -0.0435],
         [-0.0638, -0.0138,  0.0499,  ..., -0.0590,  0.1325, -0.0435]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 784, 192])
```


### DropPath 结束


: 

```python
x = x + self.drop_path(self.mlp(self.norm2(x)))
```


self.norm2: 

```python
LayerNorm((192,), eps=1e-05, elementwise_affine=True)
```


self.mlp: 

```python
Mlp(
  (fc1): Linear(in_features=192, out_features=768, bias=True)
  (act): GELU()
  (drop1): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=768, out_features=192, bias=True)
  (drop2): Dropout(p=0.0, inplace=False)
)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.2716, -0.7012, -0.9741],
         [-0.2716, -0.7012, -0.9741],
         [-0.2716, -0.7012, -0.9741]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


## SwinTransformerBlock 结束


<div style='color:#fe618e;font-weight:800;font-size:23px;'>END: for blk in self.blocks:</div>


## downsample操作


## PatchMerging 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.2716, -0.7012, -0.9741],
         [-0.2716, -0.7012, -0.9741],
         [-0.2716, -0.7012, -0.9741]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 784, 192])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H, W, C)
```


x.shape: 

```python
torch.Size([1, 28, 28, 192])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>pad操作</div>


: 

```python
x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
```


x.shape: 

```python
torch.Size([1, 28, 28, 192])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>PatchMerging取元素操作</div>


: 

```python
x0 = x[:, 0::2, 0::2, :]
```


: 

```python
x1 = x[:, 1::2, 0::2, :]
```


: 

```python
x2 = x[:, 0::2, 1::2, :]
```


: 

```python
x3 = x[:, 1::2, 1::2, :]
```


x0.shape: 

```python
torch.Size([1, 14, 14, 192])
```


x1.shape: 

```python
torch.Size([1, 14, 14, 192])
```


x2.shape: 

```python
torch.Size([1, 14, 14, 192])
```


x3.shape: 

```python
torch.Size([1, 14, 14, 192])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>cat操作</div>


: 

```python
x = torch.cat([x0, x1, x2, x3], -1)
```


x.shape: 

```python
torch.Size([1, 14, 14, 768])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, -1, 4 * C) 
```


x.shape: 

```python
torch.Size([1, 196, 768])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>norm操作</div>


: 

```python
x = self.norm(x)
```


self.norm: 

```python
LayerNorm((768,), eps=1e-05, elementwise_affine=True)
```


x.shape: 

```python
torch.Size([1, 196, 768])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>reduction操作</div>


: 

```python
x = self.reduction(x)
```


self.reduction: 

```python
Linear(in_features=768, out_features=384, bias=False)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.5255, -0.0761,  0.7896],
         [ 0.5255, -0.0761,  0.7896],
         [ 0.5255, -0.0761,  0.7896]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


## PatchMerging 结束


<div style='color:#3296ee;font-weight:800;font-size:23px;'>downsample操作</div>


: 

```python
x = self.downsample(x, H, W)
```


self.downsample: 

```python
PatchMerging(
  (reduction): Linear(in_features=768, out_features=384, bias=False)
  (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.5255, -0.0761,  0.7896],
         [ 0.5255, -0.0761,  0.7896],
         [ 0.5255, -0.0761,  0.7896]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


H: 

```python
14
```


W: 

```python
14
```


# BasicLayer 结束


# BasicLayer 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>A basic Swin Transformer layer for one stage</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.5255, -0.0761,  0.7896],
         [ 0.5255, -0.0761,  0.7896],
         [ 0.5255, -0.0761,  0.7896]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


H: 

```python
14
```


W: 

```python
14
```


## 创建mask：self.create_mask(x, H, W)


### window_partition 开始


<div style='color:#3296ee;font-weight:800;font-size:23px;'>将feature map按照window_size划分成一个个没有重叠的window</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]]]], device='cuda:0')
```


x.shape: 

```python
torch.Size([1, 14, 14, 1])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
```


x.shape: 

```python
torch.Size([1, 2, 7, 2, 7, 1])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>permute操作</div>


: 

```python
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 1])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]],

         [[0.],
          [0.],
          [0.]]],


        [[[1.],
          [1.],
          [1.]],

         [[1.],
          [1.],
          [1.]],

         [[1.],
          [1.],
          [1.]]],


        [[[3.],
          [3.],
          [3.]],

         [[3.],
          [3.],
          [3.]],

         [[3.],
          [3.],
          [3.]]]], device='cuda:0')
```


windows.shape: 

```python
torch.Size([4, 7, 7, 1])
```


### window_partition 结束


<div style='color:#fe618e;font-weight:800;font-size:23px;'>创建mask</div>


: 

```python
attn_mask = self.create_mask(x, H, W)
```


attn_mask.shape: 

```python
torch.Size([4, 49, 49])
```


## for blk in self.blocks:


<div style='color:#fe618e;font-weight:800;font-size:23px;'>for blk in self.blocks:</div>


核心代码: 

```python
x = blk(x, attn_mask)
```


## SwinTransformerBlock 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.5255, -0.0761,  0.7896],
         [ 0.5255, -0.0761,  0.7896],
         [ 0.5255, -0.0761,  0.7896]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


attn_mask[:3,:3,:3]: 

```python
tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]], device='cuda:0')
```


attn_mask.shape: 

```python
torch.Size([4, 49, 49])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>保存输入便于做残差连接</div>


: 

```python
shortcut = x
```


shortcut.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>norm操作</div>


: 

```python
x = self.norm1(x)
```


self.norm1: 

```python
LayerNorm((384,), eps=1e-05, elementwise_affine=True)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H, W, C)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>把feature map给pad到window size的整数倍</div>


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>cyclic shift</div>


: 

```python
shifted_x = x
```


: 

```python
attn_mask = None
```


shifted_x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>partition windows</div>


### window_partition 开始


<div style='color:#3296ee;font-weight:800;font-size:23px;'>将feature map按照window_size划分成一个个没有重叠的window</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347]],

         [[ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347]],

         [[ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
```


x.shape: 

```python
torch.Size([1, 2, 7, 2, 7, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>permute操作</div>


: 

```python
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347]],

         [[ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347]],

         [[ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347]]],


        [[[ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347]],

         [[ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347]],

         [[ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347]]],


        [[[ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347]],

         [[ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347]],

         [[ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347],
          [ 0.9695, -0.0902,  1.4347]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


### window_partition 结束


: 

```python
x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
```


x_windows.shape: 

```python
torch.Size([4, 49, 384])
```


### WindowAttention 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.9695, -0.0902,  1.4347],
         [ 0.9695, -0.0902,  1.4347],
         [ 0.9695, -0.0902,  1.4347]],

        [[ 0.9695, -0.0902,  1.4347],
         [ 0.9695, -0.0902,  1.4347],
         [ 0.9695, -0.0902,  1.4347]],

        [[ 0.9695, -0.0902,  1.4347],
         [ 0.9695, -0.0902,  1.4347],
         [ 0.9695, -0.0902,  1.4347]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
```


self.qkv: 

```python
Linear(in_features=384, out_features=1152, bias=True)
```


qkv.shape: 

```python
torch.Size([3, 4, 12, 49, 32])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
q, k, v = qkv.unbind(0)
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.5254,  0.1297,  0.8754],
          [ 0.5254,  0.1297,  0.8754],
          [ 0.5254,  0.1297,  0.8754]],

         [[-0.7310,  0.2050,  0.2330],
          [-0.7310,  0.2050,  0.2330],
          [-0.7310,  0.2050,  0.2330]],

         [[-0.2744,  1.0572,  0.7040],
          [-0.2744,  1.0572,  0.7040],
          [-0.2744,  1.0572,  0.7040]]],


        [[[ 0.5254,  0.1297,  0.8754],
          [ 0.5254,  0.1297,  0.8754],
          [ 0.5254,  0.1297,  0.8754]],

         [[-0.7310,  0.2050,  0.2330],
          [-0.7310,  0.2050,  0.2330],
          [-0.7310,  0.2050,  0.2330]],

         [[-0.2744,  1.0572,  0.7040],
          [-0.2744,  1.0572,  0.7040],
          [-0.2744,  1.0572,  0.7040]]],


        [[[ 0.5254,  0.1297,  0.8754],
          [ 0.5254,  0.1297,  0.8754],
          [ 0.5254,  0.1297,  0.8754]],

         [[-0.7310,  0.2050,  0.2330],
          [-0.7310,  0.2050,  0.2330],
          [-0.7310,  0.2050,  0.2330]],

         [[-0.2744,  1.0572,  0.7040],
          [-0.2744,  1.0572,  0.7040],
          [-0.2744,  1.0572,  0.7040]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


q.shape: 

```python
torch.Size([4, 12, 49, 32])
```


k[:3,:3,:3,:3]: 

```python
tensor([[[[-0.2401,  0.2462,  0.4302],
          [-0.2401,  0.2462,  0.4302],
          [-0.2401,  0.2462,  0.4302]],

         [[ 0.2226,  0.1599, -0.1436],
          [ 0.2226,  0.1599, -0.1436],
          [ 0.2226,  0.1599, -0.1436]],

         [[-0.5468,  0.1617, -0.0321],
          [-0.5468,  0.1617, -0.0321],
          [-0.5468,  0.1617, -0.0321]]],


        [[[-0.2401,  0.2462,  0.4302],
          [-0.2401,  0.2462,  0.4302],
          [-0.2401,  0.2462,  0.4302]],

         [[ 0.2226,  0.1599, -0.1436],
          [ 0.2226,  0.1599, -0.1436],
          [ 0.2226,  0.1599, -0.1436]],

         [[-0.5468,  0.1617, -0.0321],
          [-0.5468,  0.1617, -0.0321],
          [-0.5468,  0.1617, -0.0321]]],


        [[[-0.2401,  0.2462,  0.4302],
          [-0.2401,  0.2462,  0.4302],
          [-0.2401,  0.2462,  0.4302]],

         [[ 0.2226,  0.1599, -0.1436],
          [ 0.2226,  0.1599, -0.1436],
          [ 0.2226,  0.1599, -0.1436]],

         [[-0.5468,  0.1617, -0.0321],
          [-0.5468,  0.1617, -0.0321],
          [-0.5468,  0.1617, -0.0321]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


k.shape: 

```python
torch.Size([4, 12, 49, 32])
```


v[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0027, -0.3665, -0.1739],
          [ 0.0027, -0.3665, -0.1739],
          [ 0.0027, -0.3665, -0.1739]],

         [[-0.3595, -0.3124, -0.2111],
          [-0.3595, -0.3124, -0.2111],
          [-0.3595, -0.3124, -0.2111]],

         [[ 0.1804, -0.0612,  0.4796],
          [ 0.1804, -0.0612,  0.4796],
          [ 0.1804, -0.0612,  0.4796]]],


        [[[ 0.0027, -0.3665, -0.1739],
          [ 0.0027, -0.3665, -0.1739],
          [ 0.0027, -0.3665, -0.1739]],

         [[-0.3595, -0.3124, -0.2111],
          [-0.3595, -0.3124, -0.2111],
          [-0.3595, -0.3124, -0.2111]],

         [[ 0.1804, -0.0612,  0.4796],
          [ 0.1804, -0.0612,  0.4796],
          [ 0.1804, -0.0612,  0.4796]]],


        [[[ 0.0027, -0.3665, -0.1739],
          [ 0.0027, -0.3665, -0.1739],
          [ 0.0027, -0.3665, -0.1739]],

         [[-0.3595, -0.3124, -0.2111],
          [-0.3595, -0.3124, -0.2111],
          [-0.3595, -0.3124, -0.2111]],

         [[ 0.1804, -0.0612,  0.4796],
          [ 0.1804, -0.0612,  0.4796],
          [ 0.1804, -0.0612,  0.4796]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


v.shape: 

```python
torch.Size([4, 12, 49, 32])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>q = q * self.scale</div>


: 

```python
q = q * self.scale
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0929,  0.0229,  0.1547],
          [ 0.0929,  0.0229,  0.1547],
          [ 0.0929,  0.0229,  0.1547]],

         [[-0.1292,  0.0362,  0.0412],
          [-0.1292,  0.0362,  0.0412],
          [-0.1292,  0.0362,  0.0412]],

         [[-0.0485,  0.1869,  0.1245],
          [-0.0485,  0.1869,  0.1245],
          [-0.0485,  0.1869,  0.1245]]],


        [[[ 0.0929,  0.0229,  0.1547],
          [ 0.0929,  0.0229,  0.1547],
          [ 0.0929,  0.0229,  0.1547]],

         [[-0.1292,  0.0362,  0.0412],
          [-0.1292,  0.0362,  0.0412],
          [-0.1292,  0.0362,  0.0412]],

         [[-0.0485,  0.1869,  0.1245],
          [-0.0485,  0.1869,  0.1245],
          [-0.0485,  0.1869,  0.1245]]],


        [[[ 0.0929,  0.0229,  0.1547],
          [ 0.0929,  0.0229,  0.1547],
          [ 0.0929,  0.0229,  0.1547]],

         [[-0.1292,  0.0362,  0.0412],
          [-0.1292,  0.0362,  0.0412],
          [-0.1292,  0.0362,  0.0412]],

         [[-0.0485,  0.1869,  0.1245],
          [-0.0485,  0.1869,  0.1245],
          [-0.0485,  0.1869,  0.1245]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.scale: 

```python
0.1767766952966369
```


q.shape: 

```python
torch.Size([4, 12, 49, 32])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>计算q和k相似性的得分</div>


: 

```python
attn = (q @ k.transpose(-2, -1))
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1396,  0.1396,  0.1396],
          [ 0.1396,  0.1396,  0.1396],
          [ 0.1396,  0.1396,  0.1396]],

         [[-0.1778, -0.1778, -0.1778],
          [-0.1778, -0.1778, -0.1778],
          [-0.1778, -0.1778, -0.1778]],

         [[ 0.0760,  0.0760,  0.0760],
          [ 0.0760,  0.0760,  0.0760],
          [ 0.0760,  0.0760,  0.0760]]],


        [[[ 0.1396,  0.1396,  0.1396],
          [ 0.1396,  0.1396,  0.1396],
          [ 0.1396,  0.1396,  0.1396]],

         [[-0.1778, -0.1778, -0.1778],
          [-0.1778, -0.1778, -0.1778],
          [-0.1778, -0.1778, -0.1778]],

         [[ 0.0760,  0.0760,  0.0760],
          [ 0.0760,  0.0760,  0.0760],
          [ 0.0760,  0.0760,  0.0760]]],


        [[[ 0.1396,  0.1396,  0.1396],
          [ 0.1396,  0.1396,  0.1396],
          [ 0.1396,  0.1396,  0.1396]],

         [[-0.1778, -0.1778, -0.1778],
          [-0.1778, -0.1778, -0.1778],
          [-0.1778, -0.1778, -0.1778]],

         [[ 0.0760,  0.0760,  0.0760],
          [ 0.0760,  0.0760,  0.0760],
          [ 0.0760,  0.0760,  0.0760]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>relative_position_bias</div>


: 

```python
relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
```


relative_position_bias[:3,:3,:3]: 

```python
tensor([[[ 0.0071,  0.0136,  0.0009],
         [-0.0328,  0.0071,  0.0136],
         [ 0.0243, -0.0328,  0.0071]],

        [[ 0.0301, -0.0226, -0.0099],
         [-0.0172,  0.0301, -0.0226],
         [-0.0198, -0.0172,  0.0301]],

        [[ 0.0143, -0.0026, -0.0030],
         [ 0.0068,  0.0143, -0.0026],
         [ 0.0362,  0.0068,  0.0143]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


relative_position_bias.shape: 

```python
torch.Size([12, 49, 49])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>attn加上bias操作</div>


: 

```python
attn = attn + relative_position_bias.unsqueeze(0)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1467,  0.1532,  0.1405],
          [ 0.1068,  0.1467,  0.1532],
          [ 0.1639,  0.1068,  0.1467]],

         [[-0.1477, -0.2005, -0.1877],
          [-0.1950, -0.1477, -0.2005],
          [-0.1976, -0.1950, -0.1477]],

         [[ 0.0903,  0.0734,  0.0730],
          [ 0.0828,  0.0903,  0.0734],
          [ 0.1123,  0.0828,  0.0903]]],


        [[[ 0.1467,  0.1532,  0.1405],
          [ 0.1068,  0.1467,  0.1532],
          [ 0.1639,  0.1068,  0.1467]],

         [[-0.1477, -0.2005, -0.1877],
          [-0.1950, -0.1477, -0.2005],
          [-0.1976, -0.1950, -0.1477]],

         [[ 0.0903,  0.0734,  0.0730],
          [ 0.0828,  0.0903,  0.0734],
          [ 0.1123,  0.0828,  0.0903]]],


        [[[ 0.1467,  0.1532,  0.1405],
          [ 0.1068,  0.1467,  0.1532],
          [ 0.1639,  0.1068,  0.1467]],

         [[-0.1477, -0.2005, -0.1877],
          [-0.1950, -0.1477, -0.2005],
          [-0.1976, -0.1950, -0.1477]],

         [[ 0.0903,  0.0734,  0.0730],
          [ 0.0828,  0.0903,  0.0734],
          [ 0.1123,  0.0828,  0.0903]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>mask操作</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>计算相似性权重</div>


: 

```python
attn = self.softmax(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0205, 0.0207, 0.0204],
          [0.0198, 0.0206, 0.0207],
          [0.0209, 0.0198, 0.0206]],

         [[0.0210, 0.0199, 0.0202],
          [0.0201, 0.0210, 0.0200],
          [0.0200, 0.0201, 0.0210]],

         [[0.0206, 0.0203, 0.0203],
          [0.0205, 0.0206, 0.0203],
          [0.0211, 0.0205, 0.0206]]],


        [[[0.0205, 0.0207, 0.0204],
          [0.0198, 0.0206, 0.0207],
          [0.0209, 0.0198, 0.0206]],

         [[0.0210, 0.0199, 0.0202],
          [0.0201, 0.0210, 0.0200],
          [0.0200, 0.0201, 0.0210]],

         [[0.0206, 0.0203, 0.0203],
          [0.0205, 0.0206, 0.0203],
          [0.0211, 0.0205, 0.0206]]],


        [[[0.0205, 0.0207, 0.0204],
          [0.0198, 0.0206, 0.0207],
          [0.0209, 0.0198, 0.0206]],

         [[0.0210, 0.0199, 0.0202],
          [0.0201, 0.0210, 0.0200],
          [0.0200, 0.0201, 0.0210]],

         [[0.0206, 0.0203, 0.0203],
          [0.0205, 0.0206, 0.0203],
          [0.0211, 0.0205, 0.0206]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>drop操作</div>


: 

```python
attn = self.attn_drop(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0205, 0.0207, 0.0204],
          [0.0198, 0.0206, 0.0207],
          [0.0209, 0.0198, 0.0206]],

         [[0.0210, 0.0199, 0.0202],
          [0.0201, 0.0210, 0.0200],
          [0.0200, 0.0201, 0.0210]],

         [[0.0206, 0.0203, 0.0203],
          [0.0205, 0.0206, 0.0203],
          [0.0211, 0.0205, 0.0206]]],


        [[[0.0205, 0.0207, 0.0204],
          [0.0198, 0.0206, 0.0207],
          [0.0209, 0.0198, 0.0206]],

         [[0.0210, 0.0199, 0.0202],
          [0.0201, 0.0210, 0.0200],
          [0.0200, 0.0201, 0.0210]],

         [[0.0206, 0.0203, 0.0203],
          [0.0205, 0.0206, 0.0203],
          [0.0211, 0.0205, 0.0206]]],


        [[[0.0205, 0.0207, 0.0204],
          [0.0198, 0.0206, 0.0207],
          [0.0209, 0.0198, 0.0206]],

         [[0.0210, 0.0199, 0.0202],
          [0.0201, 0.0210, 0.0200],
          [0.0200, 0.0201, 0.0210]],

         [[0.0206, 0.0203, 0.0203],
          [0.0205, 0.0206, 0.0203],
          [0.0211, 0.0205, 0.0206]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>相似性得分乘v加堆叠多头操作</div>


: 

```python
x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
```


x[:3,:3,:3]: 

```python
tensor([[[ 0.0027, -0.3665, -0.1739],
         [ 0.0027, -0.3665, -0.1739],
         [ 0.0027, -0.3665, -0.1739]],

        [[ 0.0027, -0.3665, -0.1739],
         [ 0.0027, -0.3665, -0.1739],
         [ 0.0027, -0.3665, -0.1739]],

        [[ 0.0027, -0.3665, -0.1739],
         [ 0.0027, -0.3665, -0.1739],
         [ 0.0027, -0.3665, -0.1739]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>投影操作</div>


: 

```python
x = self.proj(x)
```


x[:3,:3,:3]: 

```python
tensor([[[ 0.1460, -0.0292,  0.1676],
         [ 0.1460, -0.0292,  0.1676],
         [ 0.1460, -0.0292,  0.1676]],

        [[ 0.1460, -0.0292,  0.1676],
         [ 0.1460, -0.0292,  0.1676],
         [ 0.1460, -0.0292,  0.1676]],

        [[ 0.1460, -0.0292,  0.1676],
         [ 0.1460, -0.0292,  0.1676],
         [ 0.1460, -0.0292,  0.1676]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.proj: 

```python
Linear(in_features=384, out_features=384, bias=True)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>投影dropout操作</div>


: 

```python
x = self.proj(x)
```


self.proj_drop: 

```python
Dropout(p=0.0, inplace=False)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.1460, -0.0292,  0.1676],
         [ 0.1460, -0.0292,  0.1676],
         [ 0.1460, -0.0292,  0.1676]],

        [[ 0.1460, -0.0292,  0.1676],
         [ 0.1460, -0.0292,  0.1676],
         [ 0.1460, -0.0292,  0.1676]],

        [[ 0.1460, -0.0292,  0.1676],
         [ 0.1460, -0.0292,  0.1676],
         [ 0.1460, -0.0292,  0.1676]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


### WindowAttention 结束


<div style='color:#ff9702;font-weight:800;font-size:23px;'>W-MSA/SW-MSA</div>


: 

```python
attn_windows = self.attn(x_windows, mask=attn_mask)
```


attn_windows.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>merge windows（window_reverse）</div>


: 

```python
attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
```


attn_windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


### window_reverse 开始


<div style='color:#3296ee;font-weight:800;font-size:23px;'>将一个个window还原成一个feature map</div>


<div style='color:#ff9702;font-weight:800;font-size:23px;'>输入</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676]],

         [[ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676]],

         [[ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676]]],


        [[[ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676]],

         [[ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676]],

         [[ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676]]],


        [[[ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676]],

         [[ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676]],

         [[ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


: 

```python
B = int(windows.shape[0] / (H * W / window_size / window_size))
```


B: 

```python
1
```


: 

```python
x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
```


x.shape: 

```python
torch.Size([1, 2, 2, 7, 7, 384])
```


: 

```python
x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676]],

         [[ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676]],

         [[ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676],
          [ 0.1460, -0.0292,  0.1676]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


### window_reverse 结束


: 

```python
shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
```


shifted_x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>reverse cyclic shift</div>


: 

```python
x = shifted_x
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


: 

```python
x = x.view(B, H * W, C)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>残差连接</div>


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.1460, -0.0292,  0.1676],
         [ 0.1460, -0.0292,  0.1676],
         [ 0.1460, -0.0292,  0.1676]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.1460, -0.0292,  0.1676],
         [ 0.1460, -0.0292,  0.1676],
         [ 0.1460, -0.0292,  0.1676]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[0., -0., 0.,  ..., 0., -0., -0.],
         [0., -0., 0.,  ..., 0., -0., -0.],
         [0., -0., 0.,  ..., 0., -0., -0.],
         ...,
         [0., -0., 0.,  ..., 0., -0., -0.],
         [0., -0., 0.,  ..., 0., -0., -0.],
         [0., -0., 0.,  ..., 0., -0., -0.]]], device='cuda:0',
       grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[0., -0., 0.,  ..., 0., -0., -0.],
         [0., -0., 0.,  ..., 0., -0., -0.],
         [0., -0., 0.,  ..., 0., -0., -0.],
         ...,
         [0., -0., 0.,  ..., 0., -0., -0.],
         [0., -0., 0.,  ..., 0., -0., -0.],
         [0., -0., 0.,  ..., 0., -0., -0.]]], device='cuda:0',
       grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 196, 384])
```


### DropPath 结束


: 

```python
x = shortcut + self.drop_path(x)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


### Mlp 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.9695, -0.0902,  1.4347],
         [ 0.9695, -0.0902,  1.4347],
         [ 0.9695, -0.0902,  1.4347]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


self.fc1: 

```python
Linear(in_features=384, out_features=1536, bias=True)
```


self.act: 

```python
GELU()
```


self.drop1: 

```python
Dropout(p=0.0, inplace=False)
```


self.fc2: 

```python
Linear(in_features=1536, out_features=384, bias=True)
```


self.drop2: 

```python
Dropout(p=0.0, inplace=False)
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.3863,  0.2517,  0.2143],
         [-0.3863,  0.2517,  0.2143],
         [-0.3863,  0.2517,  0.2143]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


### Mlp 结束


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.3863,  0.2517,  0.2143],
         [-0.3863,  0.2517,  0.2143],
         [-0.3863,  0.2517,  0.2143]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.3863,  0.2517,  0.2143],
         [-0.3863,  0.2517,  0.2143],
         [-0.3863,  0.2517,  0.2143]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[-0.4008,  0.2612,  0.2223,  ..., -0.0478,  0.1103,  0.0993],
         [-0.4008,  0.2612,  0.2223,  ..., -0.0478,  0.1103,  0.0993],
         [-0.4008,  0.2612,  0.2223,  ..., -0.0478,  0.1103,  0.0993],
         ...,
         [-0.4008,  0.2612,  0.2223,  ..., -0.0478,  0.1103,  0.0993],
         [-0.4008,  0.2612,  0.2223,  ..., -0.0478,  0.1103,  0.0993],
         [-0.4008,  0.2612,  0.2223,  ..., -0.0478,  0.1103,  0.0993]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[-0.4008,  0.2612,  0.2223,  ..., -0.0478,  0.1103,  0.0993],
         [-0.4008,  0.2612,  0.2223,  ..., -0.0478,  0.1103,  0.0993],
         [-0.4008,  0.2612,  0.2223,  ..., -0.0478,  0.1103,  0.0993],
         ...,
         [-0.4008,  0.2612,  0.2223,  ..., -0.0478,  0.1103,  0.0993],
         [-0.4008,  0.2612,  0.2223,  ..., -0.0478,  0.1103,  0.0993],
         [-0.4008,  0.2612,  0.2223,  ..., -0.0478,  0.1103,  0.0993]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 196, 384])
```


### DropPath 结束


: 

```python
x = x + self.drop_path(self.mlp(self.norm2(x)))
```


self.norm2: 

```python
LayerNorm((384,), eps=1e-05, elementwise_affine=True)
```


self.mlp: 

```python
Mlp(
  (fc1): Linear(in_features=384, out_features=1536, bias=True)
  (act): GELU()
  (drop1): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=1536, out_features=384, bias=True)
  (drop2): Dropout(p=0.0, inplace=False)
)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[0.1246, 0.1850, 1.0119],
         [0.1246, 0.1850, 1.0119],
         [0.1246, 0.1850, 1.0119]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


## SwinTransformerBlock 结束


## SwinTransformerBlock 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.1246, 0.1850, 1.0119],
         [0.1246, 0.1850, 1.0119],
         [0.1246, 0.1850, 1.0119]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


attn_mask[:3,:3,:3]: 

```python
tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]], device='cuda:0')
```


attn_mask.shape: 

```python
torch.Size([4, 49, 49])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>保存输入便于做残差连接</div>


: 

```python
shortcut = x
```


shortcut.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>norm操作</div>


: 

```python
x = self.norm1(x)
```


self.norm1: 

```python
LayerNorm((384,), eps=1e-05, elementwise_affine=True)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H, W, C)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>把feature map给pad到window size的整数倍</div>


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>cyclic shift</div>


: 

```python
shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
```


self.shift_size: 

```python
3
```


shifted_x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>partition windows</div>


### window_partition 开始


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>将feature map按照window_size划分成一个个没有重叠的window</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482]],

         [[0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482]],

         [[0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
```


x.shape: 

```python
torch.Size([1, 2, 7, 2, 7, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>permute操作</div>


: 

```python
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482]],

         [[0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482]],

         [[0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482]]],


        [[[0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482]],

         [[0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482]],

         [[0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482]]],


        [[[0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482]],

         [[0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482]],

         [[0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482],
          [0.2540, 0.3557, 1.7482]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


### window_partition 结束


: 

```python
x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
```


x_windows.shape: 

```python
torch.Size([4, 49, 384])
```


### WindowAttention 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.2540, 0.3557, 1.7482],
         [0.2540, 0.3557, 1.7482],
         [0.2540, 0.3557, 1.7482]],

        [[0.2540, 0.3557, 1.7482],
         [0.2540, 0.3557, 1.7482],
         [0.2540, 0.3557, 1.7482]],

        [[0.2540, 0.3557, 1.7482],
         [0.2540, 0.3557, 1.7482],
         [0.2540, 0.3557, 1.7482]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
```


self.qkv: 

```python
Linear(in_features=384, out_features=1152, bias=True)
```


qkv.shape: 

```python
torch.Size([3, 4, 12, 49, 32])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
q, k, v = qkv.unbind(0)
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[-0.5214,  0.3434,  0.2836],
          [-0.5214,  0.3434,  0.2836],
          [-0.5214,  0.3434,  0.2836]],

         [[ 0.0355,  0.2588, -0.3956],
          [ 0.0355,  0.2588, -0.3956],
          [ 0.0355,  0.2588, -0.3956]],

         [[ 0.6915,  0.0788, -0.8741],
          [ 0.6915,  0.0788, -0.8741],
          [ 0.6915,  0.0788, -0.8741]]],


        [[[-0.5214,  0.3434,  0.2836],
          [-0.5214,  0.3434,  0.2836],
          [-0.5214,  0.3434,  0.2836]],

         [[ 0.0355,  0.2588, -0.3956],
          [ 0.0355,  0.2588, -0.3956],
          [ 0.0355,  0.2588, -0.3956]],

         [[ 0.6915,  0.0788, -0.8741],
          [ 0.6915,  0.0788, -0.8741],
          [ 0.6915,  0.0788, -0.8741]]],


        [[[-0.5214,  0.3434,  0.2836],
          [-0.5214,  0.3434,  0.2836],
          [-0.5214,  0.3434,  0.2836]],

         [[ 0.0355,  0.2588, -0.3956],
          [ 0.0355,  0.2588, -0.3956],
          [ 0.0355,  0.2588, -0.3956]],

         [[ 0.6915,  0.0788, -0.8741],
          [ 0.6915,  0.0788, -0.8741],
          [ 0.6915,  0.0788, -0.8741]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


q.shape: 

```python
torch.Size([4, 12, 49, 32])
```


k[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0741,  0.7138, -0.8528],
          [-0.0741,  0.7138, -0.8528],
          [-0.0741,  0.7138, -0.8528]],

         [[ 0.2769,  0.9507,  0.0556],
          [ 0.2769,  0.9507,  0.0556],
          [ 0.2769,  0.9507,  0.0556]],

         [[-0.0282,  0.0021,  0.3274],
          [-0.0282,  0.0021,  0.3274],
          [-0.0282,  0.0021,  0.3274]]],


        [[[-0.0741,  0.7138, -0.8528],
          [-0.0741,  0.7138, -0.8528],
          [-0.0741,  0.7138, -0.8528]],

         [[ 0.2769,  0.9507,  0.0556],
          [ 0.2769,  0.9507,  0.0556],
          [ 0.2769,  0.9507,  0.0556]],

         [[-0.0282,  0.0021,  0.3274],
          [-0.0282,  0.0021,  0.3274],
          [-0.0282,  0.0021,  0.3274]]],


        [[[-0.0741,  0.7138, -0.8528],
          [-0.0741,  0.7138, -0.8528],
          [-0.0741,  0.7138, -0.8528]],

         [[ 0.2769,  0.9507,  0.0556],
          [ 0.2769,  0.9507,  0.0556],
          [ 0.2769,  0.9507,  0.0556]],

         [[-0.0282,  0.0021,  0.3274],
          [-0.0282,  0.0021,  0.3274],
          [-0.0282,  0.0021,  0.3274]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


k.shape: 

```python
torch.Size([4, 12, 49, 32])
```


v[:3,:3,:3,:3]: 

```python
tensor([[[[-0.5639, -0.2143,  0.3942],
          [-0.5639, -0.2143,  0.3942],
          [-0.5639, -0.2143,  0.3942]],

         [[-0.2447, -0.1247, -0.2014],
          [-0.2447, -0.1247, -0.2014],
          [-0.2447, -0.1247, -0.2014]],

         [[ 0.3147, -0.8328, -0.8430],
          [ 0.3147, -0.8328, -0.8430],
          [ 0.3147, -0.8328, -0.8430]]],


        [[[-0.5639, -0.2143,  0.3942],
          [-0.5639, -0.2143,  0.3942],
          [-0.5639, -0.2143,  0.3942]],

         [[-0.2447, -0.1247, -0.2014],
          [-0.2447, -0.1247, -0.2014],
          [-0.2447, -0.1247, -0.2014]],

         [[ 0.3147, -0.8328, -0.8430],
          [ 0.3147, -0.8328, -0.8430],
          [ 0.3147, -0.8328, -0.8430]]],


        [[[-0.5639, -0.2143,  0.3942],
          [-0.5639, -0.2143,  0.3942],
          [-0.5639, -0.2143,  0.3942]],

         [[-0.2447, -0.1247, -0.2014],
          [-0.2447, -0.1247, -0.2014],
          [-0.2447, -0.1247, -0.2014]],

         [[ 0.3147, -0.8328, -0.8430],
          [ 0.3147, -0.8328, -0.8430],
          [ 0.3147, -0.8328, -0.8430]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


v.shape: 

```python
torch.Size([4, 12, 49, 32])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>q = q * self.scale</div>


: 

```python
q = q * self.scale
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0922,  0.0607,  0.0501],
          [-0.0922,  0.0607,  0.0501],
          [-0.0922,  0.0607,  0.0501]],

         [[ 0.0063,  0.0457, -0.0699],
          [ 0.0063,  0.0457, -0.0699],
          [ 0.0063,  0.0457, -0.0699]],

         [[ 0.1222,  0.0139, -0.1545],
          [ 0.1222,  0.0139, -0.1545],
          [ 0.1222,  0.0139, -0.1545]]],


        [[[-0.0922,  0.0607,  0.0501],
          [-0.0922,  0.0607,  0.0501],
          [-0.0922,  0.0607,  0.0501]],

         [[ 0.0063,  0.0457, -0.0699],
          [ 0.0063,  0.0457, -0.0699],
          [ 0.0063,  0.0457, -0.0699]],

         [[ 0.1222,  0.0139, -0.1545],
          [ 0.1222,  0.0139, -0.1545],
          [ 0.1222,  0.0139, -0.1545]]],


        [[[-0.0922,  0.0607,  0.0501],
          [-0.0922,  0.0607,  0.0501],
          [-0.0922,  0.0607,  0.0501]],

         [[ 0.0063,  0.0457, -0.0699],
          [ 0.0063,  0.0457, -0.0699],
          [ 0.0063,  0.0457, -0.0699]],

         [[ 0.1222,  0.0139, -0.1545],
          [ 0.1222,  0.0139, -0.1545],
          [ 0.1222,  0.0139, -0.1545]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.scale: 

```python
0.1767766952966369
```


q.shape: 

```python
torch.Size([4, 12, 49, 32])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>计算q和k相似性的得分</div>


: 

```python
attn = (q @ k.transpose(-2, -1))
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.2215,  0.2215,  0.2215],
          [ 0.2215,  0.2215,  0.2215],
          [ 0.2215,  0.2215,  0.2215]],

         [[-0.1398, -0.1398, -0.1398],
          [-0.1398, -0.1398, -0.1398],
          [-0.1398, -0.1398, -0.1398]],

         [[ 0.2779,  0.2779,  0.2779],
          [ 0.2779,  0.2779,  0.2779],
          [ 0.2779,  0.2779,  0.2779]]],


        [[[ 0.2215,  0.2215,  0.2215],
          [ 0.2215,  0.2215,  0.2215],
          [ 0.2215,  0.2215,  0.2215]],

         [[-0.1398, -0.1398, -0.1398],
          [-0.1398, -0.1398, -0.1398],
          [-0.1398, -0.1398, -0.1398]],

         [[ 0.2779,  0.2779,  0.2779],
          [ 0.2779,  0.2779,  0.2779],
          [ 0.2779,  0.2779,  0.2779]]],


        [[[ 0.2215,  0.2215,  0.2215],
          [ 0.2215,  0.2215,  0.2215],
          [ 0.2215,  0.2215,  0.2215]],

         [[-0.1398, -0.1398, -0.1398],
          [-0.1398, -0.1398, -0.1398],
          [-0.1398, -0.1398, -0.1398]],

         [[ 0.2779,  0.2779,  0.2779],
          [ 0.2779,  0.2779,  0.2779],
          [ 0.2779,  0.2779,  0.2779]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>relative_position_bias</div>


: 

```python
relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
```


relative_position_bias[:3,:3,:3]: 

```python
tensor([[[ 0.0212,  0.0261, -0.0122],
         [ 0.0088,  0.0212,  0.0261],
         [-0.0308,  0.0088,  0.0212]],

        [[ 0.0035,  0.0073,  0.0039],
         [-0.0093,  0.0035,  0.0073],
         [-0.0171, -0.0093,  0.0035]],

        [[-0.0065,  0.0192, -0.0219],
         [ 0.0058, -0.0065,  0.0192],
         [ 0.0393,  0.0058, -0.0065]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


relative_position_bias.shape: 

```python
torch.Size([12, 49, 49])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>attn加上bias操作</div>


: 

```python
attn = attn + relative_position_bias.unsqueeze(0)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.2428,  0.2477,  0.2093],
          [ 0.2303,  0.2428,  0.2477],
          [ 0.1907,  0.2303,  0.2428]],

         [[-0.1364, -0.1326, -0.1360],
          [-0.1492, -0.1364, -0.1326],
          [-0.1569, -0.1492, -0.1364]],

         [[ 0.2714,  0.2971,  0.2560],
          [ 0.2837,  0.2714,  0.2971],
          [ 0.3172,  0.2837,  0.2714]]],


        [[[ 0.2428,  0.2477,  0.2093],
          [ 0.2303,  0.2428,  0.2477],
          [ 0.1907,  0.2303,  0.2428]],

         [[-0.1364, -0.1326, -0.1360],
          [-0.1492, -0.1364, -0.1326],
          [-0.1569, -0.1492, -0.1364]],

         [[ 0.2714,  0.2971,  0.2560],
          [ 0.2837,  0.2714,  0.2971],
          [ 0.3172,  0.2837,  0.2714]]],


        [[[ 0.2428,  0.2477,  0.2093],
          [ 0.2303,  0.2428,  0.2477],
          [ 0.1907,  0.2303,  0.2428]],

         [[-0.1364, -0.1326, -0.1360],
          [-0.1492, -0.1364, -0.1326],
          [-0.1569, -0.1492, -0.1364]],

         [[ 0.2714,  0.2971,  0.2560],
          [ 0.2837,  0.2714,  0.2971],
          [ 0.3172,  0.2837,  0.2714]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>mask操作</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>mask矩阵</div>


mask: 

```python
tensor([[[   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         ...,
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         ...,
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         ...,
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         ...,
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.]]], device='cuda:0')
```


mask.shape: 

```python
torch.Size([4, 49, 49])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>mask操作</div>


: 

```python
attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
```


attn[:3,:3,:3,:3,:3]: 

```python
tensor([[[[[ 0.2428,  0.2477,  0.2093],
           [ 0.2303,  0.2428,  0.2477],
           [ 0.1907,  0.2303,  0.2428]],

          [[-0.1364, -0.1326, -0.1360],
           [-0.1492, -0.1364, -0.1326],
           [-0.1569, -0.1492, -0.1364]],

          [[ 0.2714,  0.2971,  0.2560],
           [ 0.2837,  0.2714,  0.2971],
           [ 0.3172,  0.2837,  0.2714]]],


         [[[ 0.2428,  0.2477,  0.2093],
           [ 0.2303,  0.2428,  0.2477],
           [ 0.1907,  0.2303,  0.2428]],

          [[-0.1364, -0.1326, -0.1360],
           [-0.1492, -0.1364, -0.1326],
           [-0.1569, -0.1492, -0.1364]],

          [[ 0.2714,  0.2971,  0.2560],
           [ 0.2837,  0.2714,  0.2971],
           [ 0.3172,  0.2837,  0.2714]]],


         [[[ 0.2428,  0.2477,  0.2093],
           [ 0.2303,  0.2428,  0.2477],
           [ 0.1907,  0.2303,  0.2428]],

          [[-0.1364, -0.1326, -0.1360],
           [-0.1492, -0.1364, -0.1326],
           [-0.1569, -0.1492, -0.1364]],

          [[ 0.2714,  0.2971,  0.2560],
           [ 0.2837,  0.2714,  0.2971],
           [ 0.3172,  0.2837,  0.2714]]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([1, 4, 12, 49, 49])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>attn.view</div>


: 

```python
attn = attn.view(-1, self.num_heads, N, N)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.2428,  0.2477,  0.2093],
          [ 0.2303,  0.2428,  0.2477],
          [ 0.1907,  0.2303,  0.2428]],

         [[-0.1364, -0.1326, -0.1360],
          [-0.1492, -0.1364, -0.1326],
          [-0.1569, -0.1492, -0.1364]],

         [[ 0.2714,  0.2971,  0.2560],
          [ 0.2837,  0.2714,  0.2971],
          [ 0.3172,  0.2837,  0.2714]]],


        [[[ 0.2428,  0.2477,  0.2093],
          [ 0.2303,  0.2428,  0.2477],
          [ 0.1907,  0.2303,  0.2428]],

         [[-0.1364, -0.1326, -0.1360],
          [-0.1492, -0.1364, -0.1326],
          [-0.1569, -0.1492, -0.1364]],

         [[ 0.2714,  0.2971,  0.2560],
          [ 0.2837,  0.2714,  0.2971],
          [ 0.3172,  0.2837,  0.2714]]],


        [[[ 0.2428,  0.2477,  0.2093],
          [ 0.2303,  0.2428,  0.2477],
          [ 0.1907,  0.2303,  0.2428]],

         [[-0.1364, -0.1326, -0.1360],
          [-0.1492, -0.1364, -0.1326],
          [-0.1569, -0.1492, -0.1364]],

         [[ 0.2714,  0.2971,  0.2560],
          [ 0.2837,  0.2714,  0.2971],
          [ 0.3172,  0.2837,  0.2714]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>计算相似性权重</div>


: 

```python
attn = self.softmax(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0209, 0.0210, 0.0202],
          [0.0206, 0.0209, 0.0210],
          [0.0199, 0.0207, 0.0210]],

         [[0.0205, 0.0206, 0.0205],
          [0.0203, 0.0205, 0.0206],
          [0.0201, 0.0203, 0.0205]],

         [[0.0201, 0.0206, 0.0198],
          [0.0204, 0.0202, 0.0207],
          [0.0211, 0.0204, 0.0202]]],


        [[[0.0367, 0.0369, 0.0355],
          [0.0361, 0.0366, 0.0367],
          [0.0348, 0.0362, 0.0367]],

         [[0.0360, 0.0361, 0.0360],
          [0.0356, 0.0360, 0.0362],
          [0.0352, 0.0354, 0.0359]],

         [[0.0353, 0.0362, 0.0347],
          [0.0357, 0.0353, 0.0362],
          [0.0370, 0.0358, 0.0354]]],


        [[[0.0364, 0.0366, 0.0352],
          [0.0360, 0.0365, 0.0367],
          [0.0348, 0.0362, 0.0367]],

         [[0.0358, 0.0359, 0.0358],
          [0.0354, 0.0358, 0.0360],
          [0.0351, 0.0354, 0.0358]],

         [[0.0352, 0.0361, 0.0346],
          [0.0357, 0.0352, 0.0362],
          [0.0370, 0.0357, 0.0353]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>drop操作</div>


: 

```python
attn = self.attn_drop(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0209, 0.0210, 0.0202],
          [0.0206, 0.0209, 0.0210],
          [0.0199, 0.0207, 0.0210]],

         [[0.0205, 0.0206, 0.0205],
          [0.0203, 0.0205, 0.0206],
          [0.0201, 0.0203, 0.0205]],

         [[0.0201, 0.0206, 0.0198],
          [0.0204, 0.0202, 0.0207],
          [0.0211, 0.0204, 0.0202]]],


        [[[0.0367, 0.0369, 0.0355],
          [0.0361, 0.0366, 0.0367],
          [0.0348, 0.0362, 0.0367]],

         [[0.0360, 0.0361, 0.0360],
          [0.0356, 0.0360, 0.0362],
          [0.0352, 0.0354, 0.0359]],

         [[0.0353, 0.0362, 0.0347],
          [0.0357, 0.0353, 0.0362],
          [0.0370, 0.0358, 0.0354]]],


        [[[0.0364, 0.0366, 0.0352],
          [0.0360, 0.0365, 0.0367],
          [0.0348, 0.0362, 0.0367]],

         [[0.0358, 0.0359, 0.0358],
          [0.0354, 0.0358, 0.0360],
          [0.0351, 0.0354, 0.0358]],

         [[0.0352, 0.0361, 0.0346],
          [0.0357, 0.0352, 0.0362],
          [0.0370, 0.0357, 0.0353]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>相似性得分乘v加堆叠多头操作</div>


: 

```python
x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
```


x[:3,:3,:3]: 

```python
tensor([[[-0.5639, -0.2143,  0.3942],
         [-0.5639, -0.2143,  0.3942],
         [-0.5639, -0.2143,  0.3942]],

        [[-0.5639, -0.2143,  0.3942],
         [-0.5639, -0.2143,  0.3942],
         [-0.5639, -0.2143,  0.3942]],

        [[-0.5639, -0.2143,  0.3942],
         [-0.5639, -0.2143,  0.3942],
         [-0.5639, -0.2143,  0.3942]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>投影操作</div>


: 

```python
x = self.proj(x)
```


x[:3,:3,:3]: 

```python
tensor([[[ 0.0174,  0.0537, -0.0262],
         [ 0.0174,  0.0537, -0.0262],
         [ 0.0174,  0.0537, -0.0262]],

        [[ 0.0174,  0.0537, -0.0262],
         [ 0.0174,  0.0537, -0.0262],
         [ 0.0174,  0.0537, -0.0262]],

        [[ 0.0174,  0.0537, -0.0262],
         [ 0.0174,  0.0537, -0.0262],
         [ 0.0174,  0.0537, -0.0262]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.proj: 

```python
Linear(in_features=384, out_features=384, bias=True)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>投影dropout操作</div>


: 

```python
x = self.proj(x)
```


self.proj_drop: 

```python
Dropout(p=0.0, inplace=False)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0174,  0.0537, -0.0262],
         [ 0.0174,  0.0537, -0.0262],
         [ 0.0174,  0.0537, -0.0262]],

        [[ 0.0174,  0.0537, -0.0262],
         [ 0.0174,  0.0537, -0.0262],
         [ 0.0174,  0.0537, -0.0262]],

        [[ 0.0174,  0.0537, -0.0262],
         [ 0.0174,  0.0537, -0.0262],
         [ 0.0174,  0.0537, -0.0262]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


### WindowAttention 结束


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>W-MSA/SW-MSA</div>


: 

```python
attn_windows = self.attn(x_windows, mask=attn_mask)
```


attn_windows.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>merge windows（window_reverse）</div>


: 

```python
attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
```


attn_windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


### window_reverse 开始


<div style='color:#3296ee;font-weight:800;font-size:23px;'>将一个个window还原成一个feature map</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262]],

         [[ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262]],

         [[ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262]]],


        [[[ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262]],

         [[ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262]],

         [[ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262]]],


        [[[ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262]],

         [[ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262]],

         [[ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


: 

```python
B = int(windows.shape[0] / (H * W / window_size / window_size))
```


B: 

```python
1
```


: 

```python
x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
```


x.shape: 

```python
torch.Size([1, 2, 2, 7, 7, 384])
```


: 

```python
x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262]],

         [[ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262]],

         [[ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262],
          [ 0.0174,  0.0537, -0.0262]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


### window_reverse 结束


: 

```python
shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
```


shifted_x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>reverse cyclic shift</div>


: 

```python
x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


: 

```python
x = x.view(B, H * W, C)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>残差连接</div>


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0174,  0.0537, -0.0262],
         [ 0.0174,  0.0537, -0.0262],
         [ 0.0174,  0.0537, -0.0262]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0174,  0.0537, -0.0262],
         [ 0.0174,  0.0537, -0.0262],
         [ 0.0174,  0.0537, -0.0262]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[ 0.0182,  0.0563, -0.0274,  ...,  0.1494, -0.0323,  0.2275],
         [ 0.0182,  0.0563, -0.0274,  ...,  0.1494, -0.0323,  0.2275],
         [ 0.0182,  0.0563, -0.0274,  ...,  0.1494, -0.0323,  0.2275],
         ...,
         [ 0.0182,  0.0563, -0.0274,  ...,  0.1494, -0.0323,  0.2275],
         [ 0.0182,  0.0563, -0.0274,  ...,  0.1494, -0.0323,  0.2275],
         [ 0.0182,  0.0563, -0.0274,  ...,  0.1494, -0.0323,  0.2275]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[ 0.0182,  0.0563, -0.0274,  ...,  0.1494, -0.0323,  0.2275],
         [ 0.0182,  0.0563, -0.0274,  ...,  0.1494, -0.0323,  0.2275],
         [ 0.0182,  0.0563, -0.0274,  ...,  0.1494, -0.0323,  0.2275],
         ...,
         [ 0.0182,  0.0563, -0.0274,  ...,  0.1494, -0.0323,  0.2275],
         [ 0.0182,  0.0563, -0.0274,  ...,  0.1494, -0.0323,  0.2275],
         [ 0.0182,  0.0563, -0.0274,  ...,  0.1494, -0.0323,  0.2275]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 196, 384])
```


### DropPath 结束


: 

```python
x = shortcut + self.drop_path(x)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


### Mlp 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.2680, 0.4289, 1.6437],
         [0.2680, 0.4289, 1.6437],
         [0.2680, 0.4289, 1.6437]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


self.fc1: 

```python
Linear(in_features=384, out_features=1536, bias=True)
```


self.act: 

```python
GELU()
```


self.drop1: 

```python
Dropout(p=0.0, inplace=False)
```


self.fc2: 

```python
Linear(in_features=1536, out_features=384, bias=True)
```


self.drop2: 

```python
Dropout(p=0.0, inplace=False)
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.2741, -0.0099, -0.1390],
         [ 0.2741, -0.0099, -0.1390],
         [ 0.2741, -0.0099, -0.1390]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


### Mlp 结束


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.2741, -0.0099, -0.1390],
         [ 0.2741, -0.0099, -0.1390],
         [ 0.2741, -0.0099, -0.1390]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.2741, -0.0099, -0.1390],
         [ 0.2741, -0.0099, -0.1390],
         [ 0.2741, -0.0099, -0.1390]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[ 0.2871, -0.0103, -0.1456,  ...,  0.0562, -0.0402,  0.0946],
         [ 0.2871, -0.0103, -0.1456,  ...,  0.0562, -0.0402,  0.0946],
         [ 0.2871, -0.0103, -0.1456,  ...,  0.0562, -0.0402,  0.0946],
         ...,
         [ 0.2871, -0.0103, -0.1456,  ...,  0.0562, -0.0402,  0.0946],
         [ 0.2871, -0.0103, -0.1456,  ...,  0.0562, -0.0402,  0.0946],
         [ 0.2871, -0.0103, -0.1456,  ...,  0.0562, -0.0402,  0.0946]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[ 0.2871, -0.0103, -0.1456,  ...,  0.0562, -0.0402,  0.0946],
         [ 0.2871, -0.0103, -0.1456,  ...,  0.0562, -0.0402,  0.0946],
         [ 0.2871, -0.0103, -0.1456,  ...,  0.0562, -0.0402,  0.0946],
         ...,
         [ 0.2871, -0.0103, -0.1456,  ...,  0.0562, -0.0402,  0.0946],
         [ 0.2871, -0.0103, -0.1456,  ...,  0.0562, -0.0402,  0.0946],
         [ 0.2871, -0.0103, -0.1456,  ...,  0.0562, -0.0402,  0.0946]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 196, 384])
```


### DropPath 结束


: 

```python
x = x + self.drop_path(self.mlp(self.norm2(x)))
```


self.norm2: 

```python
LayerNorm((384,), eps=1e-05, elementwise_affine=True)
```


self.mlp: 

```python
Mlp(
  (fc1): Linear(in_features=384, out_features=1536, bias=True)
  (act): GELU()
  (drop1): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=1536, out_features=384, bias=True)
  (drop2): Dropout(p=0.0, inplace=False)
)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[0.4299, 0.2310, 0.8389],
         [0.4299, 0.2310, 0.8389],
         [0.4299, 0.2310, 0.8389]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


## SwinTransformerBlock 结束


## SwinTransformerBlock 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.4299, 0.2310, 0.8389],
         [0.4299, 0.2310, 0.8389],
         [0.4299, 0.2310, 0.8389]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


attn_mask[:3,:3,:3]: 

```python
tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]], device='cuda:0')
```


attn_mask.shape: 

```python
torch.Size([4, 49, 49])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>保存输入便于做残差连接</div>


: 

```python
shortcut = x
```


shortcut.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>norm操作</div>


: 

```python
x = self.norm1(x)
```


self.norm1: 

```python
LayerNorm((384,), eps=1e-05, elementwise_affine=True)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H, W, C)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>把feature map给pad到window size的整数倍</div>


<div style='color:#fe1111;font-weight:800;font-size:23px;'>cyclic shift</div>


: 

```python
shifted_x = x
```


: 

```python
attn_mask = None
```


shifted_x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>partition windows</div>


### window_partition 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>将feature map按照window_size划分成一个个没有重叠的window</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690]],

         [[0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690]],

         [[0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
```


x.shape: 

```python
torch.Size([1, 2, 7, 2, 7, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>permute操作</div>


: 

```python
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690]],

         [[0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690]],

         [[0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690]]],


        [[[0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690]],

         [[0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690]],

         [[0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690]]],


        [[[0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690]],

         [[0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690]],

         [[0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690],
          [0.7244, 0.4108, 1.3690]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


### window_partition 结束


: 

```python
x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
```


x_windows.shape: 

```python
torch.Size([4, 49, 384])
```


### WindowAttention 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.7244, 0.4108, 1.3690],
         [0.7244, 0.4108, 1.3690],
         [0.7244, 0.4108, 1.3690]],

        [[0.7244, 0.4108, 1.3690],
         [0.7244, 0.4108, 1.3690],
         [0.7244, 0.4108, 1.3690]],

        [[0.7244, 0.4108, 1.3690],
         [0.7244, 0.4108, 1.3690],
         [0.7244, 0.4108, 1.3690]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
```


self.qkv: 

```python
Linear(in_features=384, out_features=1152, bias=True)
```


qkv.shape: 

```python
torch.Size([3, 4, 12, 49, 32])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
q, k, v = qkv.unbind(0)
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[-0.3252,  0.1060, -0.5215],
          [-0.3252,  0.1060, -0.5215],
          [-0.3252,  0.1060, -0.5215]],

         [[ 0.4235, -0.1027, -0.2132],
          [ 0.4235, -0.1027, -0.2132],
          [ 0.4235, -0.1027, -0.2132]],

         [[ 0.6025, -0.0414, -0.2785],
          [ 0.6025, -0.0414, -0.2785],
          [ 0.6025, -0.0414, -0.2785]]],


        [[[-0.3252,  0.1060, -0.5215],
          [-0.3252,  0.1060, -0.5215],
          [-0.3252,  0.1060, -0.5215]],

         [[ 0.4235, -0.1027, -0.2132],
          [ 0.4235, -0.1027, -0.2132],
          [ 0.4235, -0.1027, -0.2132]],

         [[ 0.6025, -0.0414, -0.2785],
          [ 0.6025, -0.0414, -0.2785],
          [ 0.6025, -0.0414, -0.2785]]],


        [[[-0.3252,  0.1060, -0.5215],
          [-0.3252,  0.1060, -0.5215],
          [-0.3252,  0.1060, -0.5215]],

         [[ 0.4235, -0.1027, -0.2132],
          [ 0.4235, -0.1027, -0.2132],
          [ 0.4235, -0.1027, -0.2132]],

         [[ 0.6025, -0.0414, -0.2785],
          [ 0.6025, -0.0414, -0.2785],
          [ 0.6025, -0.0414, -0.2785]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


q.shape: 

```python
torch.Size([4, 12, 49, 32])
```


k[:3,:3,:3,:3]: 

```python
tensor([[[[-0.6007, -0.0474,  0.3868],
          [-0.6007, -0.0474,  0.3868],
          [-0.6007, -0.0474,  0.3868]],

         [[ 0.4735,  0.0658,  0.1535],
          [ 0.4735,  0.0658,  0.1535],
          [ 0.4735,  0.0658,  0.1535]],

         [[ 0.7069,  0.7661,  0.6417],
          [ 0.7069,  0.7661,  0.6417],
          [ 0.7069,  0.7661,  0.6417]]],


        [[[-0.6007, -0.0474,  0.3868],
          [-0.6007, -0.0474,  0.3868],
          [-0.6007, -0.0474,  0.3868]],

         [[ 0.4735,  0.0658,  0.1535],
          [ 0.4735,  0.0658,  0.1535],
          [ 0.4735,  0.0658,  0.1535]],

         [[ 0.7069,  0.7661,  0.6417],
          [ 0.7069,  0.7661,  0.6417],
          [ 0.7069,  0.7661,  0.6417]]],


        [[[-0.6007, -0.0474,  0.3868],
          [-0.6007, -0.0474,  0.3868],
          [-0.6007, -0.0474,  0.3868]],

         [[ 0.4735,  0.0658,  0.1535],
          [ 0.4735,  0.0658,  0.1535],
          [ 0.4735,  0.0658,  0.1535]],

         [[ 0.7069,  0.7661,  0.6417],
          [ 0.7069,  0.7661,  0.6417],
          [ 0.7069,  0.7661,  0.6417]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


k.shape: 

```python
torch.Size([4, 12, 49, 32])
```


v[:3,:3,:3,:3]: 

```python
tensor([[[[-1.6177e-01, -2.6089e-01, -8.9739e-01],
          [-1.6177e-01, -2.6089e-01, -8.9739e-01],
          [-1.6177e-01, -2.6089e-01, -8.9739e-01]],

         [[-1.9721e-01,  3.4890e-01,  2.2060e-04],
          [-1.9721e-01,  3.4890e-01,  2.2057e-04],
          [-1.9721e-01,  3.4890e-01,  2.2056e-04]],

         [[-1.5418e-01, -2.3976e-01,  2.5173e-01],
          [-1.5418e-01, -2.3976e-01,  2.5173e-01],
          [-1.5418e-01, -2.3975e-01,  2.5173e-01]]],


        [[[-1.6177e-01, -2.6089e-01, -8.9739e-01],
          [-1.6177e-01, -2.6089e-01, -8.9739e-01],
          [-1.6177e-01, -2.6089e-01, -8.9739e-01]],

         [[-1.9721e-01,  3.4890e-01,  2.2062e-04],
          [-1.9721e-01,  3.4890e-01,  2.2050e-04],
          [-1.9721e-01,  3.4890e-01,  2.2056e-04]],

         [[-1.5418e-01, -2.3976e-01,  2.5173e-01],
          [-1.5418e-01, -2.3976e-01,  2.5173e-01],
          [-1.5418e-01, -2.3976e-01,  2.5173e-01]]],


        [[[-1.6177e-01, -2.6089e-01, -8.9739e-01],
          [-1.6177e-01, -2.6089e-01, -8.9739e-01],
          [-1.6177e-01, -2.6089e-01, -8.9739e-01]],

         [[-1.9721e-01,  3.4890e-01,  2.2055e-04],
          [-1.9721e-01,  3.4890e-01,  2.2064e-04],
          [-1.9721e-01,  3.4890e-01,  2.2060e-04]],

         [[-1.5418e-01, -2.3976e-01,  2.5173e-01],
          [-1.5418e-01, -2.3976e-01,  2.5173e-01],
          [-1.5418e-01, -2.3976e-01,  2.5173e-01]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


v.shape: 

```python
torch.Size([4, 12, 49, 32])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>q = q * self.scale</div>


: 

```python
q = q * self.scale
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0575,  0.0187, -0.0922],
          [-0.0575,  0.0187, -0.0922],
          [-0.0575,  0.0187, -0.0922]],

         [[ 0.0749, -0.0181, -0.0377],
          [ 0.0749, -0.0181, -0.0377],
          [ 0.0749, -0.0181, -0.0377]],

         [[ 0.1065, -0.0073, -0.0492],
          [ 0.1065, -0.0073, -0.0492],
          [ 0.1065, -0.0073, -0.0492]]],


        [[[-0.0575,  0.0187, -0.0922],
          [-0.0575,  0.0187, -0.0922],
          [-0.0575,  0.0187, -0.0922]],

         [[ 0.0749, -0.0181, -0.0377],
          [ 0.0749, -0.0181, -0.0377],
          [ 0.0749, -0.0181, -0.0377]],

         [[ 0.1065, -0.0073, -0.0492],
          [ 0.1065, -0.0073, -0.0492],
          [ 0.1065, -0.0073, -0.0492]]],


        [[[-0.0575,  0.0187, -0.0922],
          [-0.0575,  0.0187, -0.0922],
          [-0.0575,  0.0187, -0.0922]],

         [[ 0.0749, -0.0181, -0.0377],
          [ 0.0749, -0.0181, -0.0377],
          [ 0.0749, -0.0181, -0.0377]],

         [[ 0.1065, -0.0073, -0.0492],
          [ 0.1065, -0.0073, -0.0492],
          [ 0.1065, -0.0073, -0.0492]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.scale: 

```python
0.1767766952966369
```


q.shape: 

```python
torch.Size([4, 12, 49, 32])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>计算q和k相似性的得分</div>


: 

```python
attn = (q @ k.transpose(-2, -1))
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.1813, -0.1813, -0.1813],
          [-0.1813, -0.1813, -0.1813],
          [-0.1813, -0.1813, -0.1813]],

         [[ 0.0073,  0.0073,  0.0073],
          [ 0.0073,  0.0073,  0.0073],
          [ 0.0073,  0.0073,  0.0073]],

         [[-0.0133, -0.0133, -0.0133],
          [-0.0133, -0.0133, -0.0133],
          [-0.0133, -0.0133, -0.0133]]],


        [[[-0.1813, -0.1813, -0.1813],
          [-0.1813, -0.1813, -0.1813],
          [-0.1813, -0.1813, -0.1813]],

         [[ 0.0073,  0.0073,  0.0073],
          [ 0.0073,  0.0073,  0.0073],
          [ 0.0073,  0.0073,  0.0073]],

         [[-0.0133, -0.0133, -0.0133],
          [-0.0133, -0.0133, -0.0133],
          [-0.0133, -0.0133, -0.0133]]],


        [[[-0.1813, -0.1813, -0.1813],
          [-0.1813, -0.1813, -0.1813],
          [-0.1813, -0.1813, -0.1813]],

         [[ 0.0073,  0.0073,  0.0073],
          [ 0.0073,  0.0073,  0.0073],
          [ 0.0073,  0.0073,  0.0073]],

         [[-0.0133, -0.0133, -0.0133],
          [-0.0133, -0.0133, -0.0133],
          [-0.0133, -0.0133, -0.0133]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>relative_position_bias</div>


: 

```python
relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
```


relative_position_bias[:3,:3,:3]: 

```python
tensor([[[ 0.0257,  0.0451,  0.0309],
         [-0.0053,  0.0257,  0.0451],
         [-0.0141, -0.0053,  0.0257]],

        [[-0.0126, -0.0112, -0.0115],
         [-0.0074, -0.0126, -0.0112],
         [ 0.0070, -0.0074, -0.0126]],

        [[-0.0259, -0.0061,  0.0112],
         [ 0.0003, -0.0259, -0.0061],
         [ 0.0044,  0.0003, -0.0259]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


relative_position_bias.shape: 

```python
torch.Size([12, 49, 49])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>attn加上bias操作</div>


: 

```python
attn = attn + relative_position_bias.unsqueeze(0)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-1.5560e-01, -1.3622e-01, -1.5041e-01],
          [-1.8662e-01, -1.5560e-01, -1.3622e-01],
          [-1.9548e-01, -1.8662e-01, -1.5560e-01]],

         [[-5.3258e-03, -3.9279e-03, -4.2105e-03],
          [-6.2999e-05, -5.3258e-03, -3.9279e-03],
          [ 1.4346e-02, -6.2916e-05, -5.3257e-03]],

         [[-3.9170e-02, -1.9396e-02, -2.1076e-03],
          [-1.2944e-02, -3.9170e-02, -1.9396e-02],
          [-8.8463e-03, -1.2944e-02, -3.9170e-02]]],


        [[[-1.5560e-01, -1.3622e-01, -1.5041e-01],
          [-1.8662e-01, -1.5560e-01, -1.3622e-01],
          [-1.9548e-01, -1.8662e-01, -1.5560e-01]],

         [[-5.3258e-03, -3.9279e-03, -4.2104e-03],
          [-6.2988e-05, -5.3258e-03, -3.9278e-03],
          [ 1.4346e-02, -6.2936e-05, -5.3257e-03]],

         [[-3.9170e-02, -1.9396e-02, -2.1077e-03],
          [-1.2944e-02, -3.9170e-02, -1.9396e-02],
          [-8.8462e-03, -1.2944e-02, -3.9170e-02]]],


        [[[-1.5560e-01, -1.3622e-01, -1.5041e-01],
          [-1.8662e-01, -1.5560e-01, -1.3622e-01],
          [-1.9548e-01, -1.8662e-01, -1.5560e-01]],

         [[-5.3258e-03, -3.9279e-03, -4.2105e-03],
          [-6.2916e-05, -5.3258e-03, -3.9278e-03],
          [ 1.4346e-02, -6.2923e-05, -5.3257e-03]],

         [[-3.9170e-02, -1.9396e-02, -2.1076e-03],
          [-1.2944e-02, -3.9170e-02, -1.9396e-02],
          [-8.8462e-03, -1.2944e-02, -3.9170e-02]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>mask操作</div>


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>计算相似性权重</div>


: 

```python
attn = self.softmax(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0209, 0.0213, 0.0210],
          [0.0203, 0.0210, 0.0214],
          [0.0202, 0.0204, 0.0210]],

         [[0.0202, 0.0202, 0.0202],
          [0.0203, 0.0202, 0.0202],
          [0.0206, 0.0203, 0.0202]],

         [[0.0198, 0.0202, 0.0206],
          [0.0203, 0.0198, 0.0202],
          [0.0204, 0.0203, 0.0198]]],


        [[[0.0209, 0.0213, 0.0210],
          [0.0203, 0.0210, 0.0214],
          [0.0202, 0.0204, 0.0210]],

         [[0.0202, 0.0202, 0.0202],
          [0.0203, 0.0202, 0.0202],
          [0.0206, 0.0203, 0.0202]],

         [[0.0198, 0.0202, 0.0206],
          [0.0203, 0.0198, 0.0202],
          [0.0204, 0.0203, 0.0198]]],


        [[[0.0209, 0.0213, 0.0210],
          [0.0203, 0.0210, 0.0214],
          [0.0202, 0.0204, 0.0210]],

         [[0.0202, 0.0202, 0.0202],
          [0.0203, 0.0202, 0.0202],
          [0.0206, 0.0203, 0.0202]],

         [[0.0198, 0.0202, 0.0206],
          [0.0203, 0.0198, 0.0202],
          [0.0204, 0.0203, 0.0198]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>drop操作</div>


: 

```python
attn = self.attn_drop(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0209, 0.0213, 0.0210],
          [0.0203, 0.0210, 0.0214],
          [0.0202, 0.0204, 0.0210]],

         [[0.0202, 0.0202, 0.0202],
          [0.0203, 0.0202, 0.0202],
          [0.0206, 0.0203, 0.0202]],

         [[0.0198, 0.0202, 0.0206],
          [0.0203, 0.0198, 0.0202],
          [0.0204, 0.0203, 0.0198]]],


        [[[0.0209, 0.0213, 0.0210],
          [0.0203, 0.0210, 0.0214],
          [0.0202, 0.0204, 0.0210]],

         [[0.0202, 0.0202, 0.0202],
          [0.0203, 0.0202, 0.0202],
          [0.0206, 0.0203, 0.0202]],

         [[0.0198, 0.0202, 0.0206],
          [0.0203, 0.0198, 0.0202],
          [0.0204, 0.0203, 0.0198]]],


        [[[0.0209, 0.0213, 0.0210],
          [0.0203, 0.0210, 0.0214],
          [0.0202, 0.0204, 0.0210]],

         [[0.0202, 0.0202, 0.0202],
          [0.0203, 0.0202, 0.0202],
          [0.0206, 0.0203, 0.0202]],

         [[0.0198, 0.0202, 0.0206],
          [0.0203, 0.0198, 0.0202],
          [0.0204, 0.0203, 0.0198]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>相似性得分乘v加堆叠多头操作</div>


: 

```python
x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
```


x[:3,:3,:3]: 

```python
tensor([[[-0.1618, -0.2609, -0.8974],
         [-0.1618, -0.2609, -0.8974],
         [-0.1618, -0.2609, -0.8974]],

        [[-0.1618, -0.2609, -0.8974],
         [-0.1618, -0.2609, -0.8974],
         [-0.1618, -0.2609, -0.8974]],

        [[-0.1618, -0.2609, -0.8974],
         [-0.1618, -0.2609, -0.8974],
         [-0.1618, -0.2609, -0.8974]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>投影操作</div>


: 

```python
x = self.proj(x)
```


x[:3,:3,:3]: 

```python
tensor([[[ 0.1567, -0.0885,  0.0009],
         [ 0.1567, -0.0885,  0.0009],
         [ 0.1567, -0.0885,  0.0009]],

        [[ 0.1567, -0.0885,  0.0009],
         [ 0.1567, -0.0885,  0.0009],
         [ 0.1567, -0.0885,  0.0009]],

        [[ 0.1567, -0.0885,  0.0009],
         [ 0.1567, -0.0885,  0.0009],
         [ 0.1567, -0.0885,  0.0009]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.proj: 

```python
Linear(in_features=384, out_features=384, bias=True)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>投影dropout操作</div>


: 

```python
x = self.proj(x)
```


self.proj_drop: 

```python
Dropout(p=0.0, inplace=False)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.1567, -0.0885,  0.0009],
         [ 0.1567, -0.0885,  0.0009],
         [ 0.1567, -0.0885,  0.0009]],

        [[ 0.1567, -0.0885,  0.0009],
         [ 0.1567, -0.0885,  0.0009],
         [ 0.1567, -0.0885,  0.0009]],

        [[ 0.1567, -0.0885,  0.0009],
         [ 0.1567, -0.0885,  0.0009],
         [ 0.1567, -0.0885,  0.0009]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


### WindowAttention 结束


<div style='color:#3296ee;font-weight:800;font-size:23px;'>W-MSA/SW-MSA</div>


: 

```python
attn_windows = self.attn(x_windows, mask=attn_mask)
```


attn_windows.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>merge windows（window_reverse）</div>


: 

```python
attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
```


attn_windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


### window_reverse 开始


<div style='color:#3296ee;font-weight:800;font-size:23px;'>将一个个window还原成一个feature map</div>


<div style='color:#3296ee;font-weight:800;font-size:23px;'>输入</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009]],

         [[ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009]],

         [[ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009]]],


        [[[ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009]],

         [[ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009]],

         [[ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009]]],


        [[[ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009]],

         [[ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009]],

         [[ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


: 

```python
B = int(windows.shape[0] / (H * W / window_size / window_size))
```


B: 

```python
1
```


: 

```python
x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
```


x.shape: 

```python
torch.Size([1, 2, 2, 7, 7, 384])
```


: 

```python
x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009]],

         [[ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009]],

         [[ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009],
          [ 0.1567, -0.0885,  0.0009]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


### window_reverse 结束


: 

```python
shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
```


shifted_x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>reverse cyclic shift</div>


: 

```python
x = shifted_x
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


: 

```python
x = x.view(B, H * W, C)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>残差连接</div>


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.1567, -0.0885,  0.0009],
         [ 0.1567, -0.0885,  0.0009],
         [ 0.1567, -0.0885,  0.0009]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.1567, -0.0885,  0.0009],
         [ 0.1567, -0.0885,  0.0009],
         [ 0.1567, -0.0885,  0.0009]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[ 0.1657, -0.0936,  0.0009,  ...,  0.0826, -0.0623, -0.1847],
         [ 0.1657, -0.0936,  0.0009,  ...,  0.0826, -0.0623, -0.1847],
         [ 0.1657, -0.0936,  0.0009,  ...,  0.0826, -0.0623, -0.1847],
         ...,
         [ 0.1657, -0.0936,  0.0009,  ...,  0.0826, -0.0623, -0.1847],
         [ 0.1657, -0.0936,  0.0009,  ...,  0.0826, -0.0623, -0.1847],
         [ 0.1657, -0.0936,  0.0009,  ...,  0.0826, -0.0623, -0.1847]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[ 0.1657, -0.0936,  0.0009,  ...,  0.0826, -0.0623, -0.1847],
         [ 0.1657, -0.0936,  0.0009,  ...,  0.0826, -0.0623, -0.1847],
         [ 0.1657, -0.0936,  0.0009,  ...,  0.0826, -0.0623, -0.1847],
         ...,
         [ 0.1657, -0.0936,  0.0009,  ...,  0.0826, -0.0623, -0.1847],
         [ 0.1657, -0.0936,  0.0009,  ...,  0.0826, -0.0623, -0.1847],
         [ 0.1657, -0.0936,  0.0009,  ...,  0.0826, -0.0623, -0.1847]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 196, 384])
```


### DropPath 结束


: 

```python
x = shortcut + self.drop_path(x)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


### Mlp 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.9508, 0.2576, 1.3201],
         [0.9508, 0.2576, 1.3201],
         [0.9508, 0.2576, 1.3201]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


self.fc1: 

```python
Linear(in_features=384, out_features=1536, bias=True)
```


self.act: 

```python
GELU()
```


self.drop1: 

```python
Dropout(p=0.0, inplace=False)
```


self.fc2: 

```python
Linear(in_features=1536, out_features=384, bias=True)
```


self.drop2: 

```python
Dropout(p=0.0, inplace=False)
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0474, -0.0641,  0.1570],
         [ 0.0474, -0.0641,  0.1570],
         [ 0.0474, -0.0641,  0.1570]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


### Mlp 结束


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0474, -0.0641,  0.1570],
         [ 0.0474, -0.0641,  0.1570],
         [ 0.0474, -0.0641,  0.1570]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0474, -0.0641,  0.1570],
         [ 0.0474, -0.0641,  0.1570],
         [ 0.0474, -0.0641,  0.1570]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[ 0.0502, -0.0678,  0.1660,  ..., -0.3171, -0.2118,  0.1744],
         [ 0.0502, -0.0678,  0.1660,  ..., -0.3171, -0.2118,  0.1744],
         [ 0.0502, -0.0678,  0.1660,  ..., -0.3171, -0.2118,  0.1744],
         ...,
         [ 0.0502, -0.0678,  0.1660,  ..., -0.3171, -0.2118,  0.1744],
         [ 0.0502, -0.0678,  0.1660,  ..., -0.3171, -0.2118,  0.1744],
         [ 0.0502, -0.0678,  0.1660,  ..., -0.3171, -0.2118,  0.1744]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[ 0.0502, -0.0678,  0.1660,  ..., -0.3171, -0.2118,  0.1744],
         [ 0.0502, -0.0678,  0.1660,  ..., -0.3171, -0.2118,  0.1744],
         [ 0.0502, -0.0678,  0.1660,  ..., -0.3171, -0.2118,  0.1744],
         ...,
         [ 0.0502, -0.0678,  0.1660,  ..., -0.3171, -0.2118,  0.1744],
         [ 0.0502, -0.0678,  0.1660,  ..., -0.3171, -0.2118,  0.1744],
         [ 0.0502, -0.0678,  0.1660,  ..., -0.3171, -0.2118,  0.1744]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 196, 384])
```


### DropPath 结束


: 

```python
x = x + self.drop_path(self.mlp(self.norm2(x)))
```


self.norm2: 

```python
LayerNorm((384,), eps=1e-05, elementwise_affine=True)
```


self.mlp: 

```python
Mlp(
  (fc1): Linear(in_features=384, out_features=1536, bias=True)
  (act): GELU()
  (drop1): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=1536, out_features=384, bias=True)
  (drop2): Dropout(p=0.0, inplace=False)
)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[0.6459, 0.0695, 1.0058],
         [0.6459, 0.0695, 1.0058],
         [0.6459, 0.0695, 1.0058]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


## SwinTransformerBlock 结束


## SwinTransformerBlock 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.6459, 0.0695, 1.0058],
         [0.6459, 0.0695, 1.0058],
         [0.6459, 0.0695, 1.0058]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


attn_mask[:3,:3,:3]: 

```python
tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]], device='cuda:0')
```


attn_mask.shape: 

```python
torch.Size([4, 49, 49])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>保存输入便于做残差连接</div>


: 

```python
shortcut = x
```


shortcut.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>norm操作</div>


: 

```python
x = self.norm1(x)
```


self.norm1: 

```python
LayerNorm((384,), eps=1e-05, elementwise_affine=True)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H, W, C)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>把feature map给pad到window size的整数倍</div>


<div style='color:#fd7949;font-weight:800;font-size:23px;'>cyclic shift</div>


: 

```python
shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
```


self.shift_size: 

```python
3
```


shifted_x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>partition windows</div>


### window_partition 开始


<div style='color:#fd7949;font-weight:800;font-size:23px;'>将feature map按照window_size划分成一个个没有重叠的window</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160]],

         [[0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160]],

         [[0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
```


x.shape: 

```python
torch.Size([1, 2, 7, 2, 7, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>permute操作</div>


: 

```python
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160]],

         [[0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160]],

         [[0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160]]],


        [[[0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160]],

         [[0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160]],

         [[0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160]]],


        [[[0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160]],

         [[0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160]],

         [[0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160],
          [0.9931, 0.1560, 1.5160]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


### window_partition 结束


: 

```python
x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
```


x_windows.shape: 

```python
torch.Size([4, 49, 384])
```


### WindowAttention 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.9931, 0.1560, 1.5160],
         [0.9931, 0.1560, 1.5160],
         [0.9931, 0.1560, 1.5160]],

        [[0.9931, 0.1560, 1.5160],
         [0.9931, 0.1560, 1.5160],
         [0.9931, 0.1560, 1.5160]],

        [[0.9931, 0.1560, 1.5160],
         [0.9931, 0.1560, 1.5160],
         [0.9931, 0.1560, 1.5160]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
```


self.qkv: 

```python
Linear(in_features=384, out_features=1152, bias=True)
```


qkv.shape: 

```python
torch.Size([3, 4, 12, 49, 32])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
q, k, v = qkv.unbind(0)
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1090, -0.6879, -0.4784],
          [ 0.1090, -0.6879, -0.4784],
          [ 0.1090, -0.6879, -0.4784]],

         [[ 0.0455, -0.6347, -0.0843],
          [ 0.0455, -0.6347, -0.0843],
          [ 0.0455, -0.6347, -0.0843]],

         [[-0.4820,  0.1225,  0.3231],
          [-0.4820,  0.1225,  0.3231],
          [-0.4820,  0.1225,  0.3231]]],


        [[[ 0.1090, -0.6879, -0.4784],
          [ 0.1090, -0.6879, -0.4784],
          [ 0.1090, -0.6879, -0.4784]],

         [[ 0.0455, -0.6347, -0.0843],
          [ 0.0455, -0.6347, -0.0843],
          [ 0.0455, -0.6347, -0.0843]],

         [[-0.4820,  0.1225,  0.3231],
          [-0.4820,  0.1225,  0.3231],
          [-0.4820,  0.1225,  0.3231]]],


        [[[ 0.1090, -0.6879, -0.4784],
          [ 0.1090, -0.6879, -0.4784],
          [ 0.1090, -0.6879, -0.4784]],

         [[ 0.0455, -0.6347, -0.0843],
          [ 0.0455, -0.6347, -0.0843],
          [ 0.0455, -0.6347, -0.0843]],

         [[-0.4820,  0.1225,  0.3231],
          [-0.4820,  0.1225,  0.3231],
          [-0.4820,  0.1225,  0.3231]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


q.shape: 

```python
torch.Size([4, 12, 49, 32])
```


k[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0995,  0.6862,  0.5888],
          [-0.0995,  0.6862,  0.5888],
          [-0.0995,  0.6862,  0.5888]],

         [[ 0.5330, -0.2899,  0.4158],
          [ 0.5330, -0.2899,  0.4158],
          [ 0.5330, -0.2899,  0.4158]],

         [[-0.2995,  0.2086,  0.0412],
          [-0.2995,  0.2086,  0.0412],
          [-0.2995,  0.2086,  0.0412]]],


        [[[-0.0995,  0.6862,  0.5888],
          [-0.0995,  0.6862,  0.5888],
          [-0.0995,  0.6862,  0.5888]],

         [[ 0.5330, -0.2899,  0.4158],
          [ 0.5330, -0.2899,  0.4158],
          [ 0.5330, -0.2899,  0.4158]],

         [[-0.2995,  0.2086,  0.0412],
          [-0.2995,  0.2086,  0.0412],
          [-0.2995,  0.2086,  0.0412]]],


        [[[-0.0995,  0.6862,  0.5888],
          [-0.0995,  0.6862,  0.5888],
          [-0.0995,  0.6862,  0.5888]],

         [[ 0.5330, -0.2899,  0.4158],
          [ 0.5330, -0.2899,  0.4158],
          [ 0.5330, -0.2899,  0.4158]],

         [[-0.2995,  0.2086,  0.0412],
          [-0.2995,  0.2086,  0.0412],
          [-0.2995,  0.2086,  0.0412]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


k.shape: 

```python
torch.Size([4, 12, 49, 32])
```


v[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1278, -0.1895,  0.0467],
          [ 0.1278, -0.1895,  0.0467],
          [ 0.1278, -0.1895,  0.0467]],

         [[-0.5087, -0.1139,  0.1725],
          [-0.5087, -0.1139,  0.1725],
          [-0.5087, -0.1139,  0.1725]],

         [[-0.4529,  0.1175,  0.4441],
          [-0.4529,  0.1175,  0.4441],
          [-0.4529,  0.1175,  0.4441]]],


        [[[ 0.1278, -0.1895,  0.0467],
          [ 0.1278, -0.1895,  0.0467],
          [ 0.1278, -0.1895,  0.0467]],

         [[-0.5087, -0.1139,  0.1725],
          [-0.5087, -0.1139,  0.1725],
          [-0.5087, -0.1139,  0.1725]],

         [[-0.4529,  0.1175,  0.4441],
          [-0.4529,  0.1175,  0.4441],
          [-0.4529,  0.1175,  0.4441]]],


        [[[ 0.1278, -0.1895,  0.0467],
          [ 0.1278, -0.1895,  0.0467],
          [ 0.1278, -0.1895,  0.0467]],

         [[-0.5087, -0.1139,  0.1725],
          [-0.5087, -0.1139,  0.1725],
          [-0.5087, -0.1139,  0.1725]],

         [[-0.4529,  0.1175,  0.4441],
          [-0.4529,  0.1175,  0.4441],
          [-0.4529,  0.1175,  0.4441]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


v.shape: 

```python
torch.Size([4, 12, 49, 32])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>q = q * self.scale</div>


: 

```python
q = q * self.scale
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0193, -0.1216, -0.0846],
          [ 0.0193, -0.1216, -0.0846],
          [ 0.0193, -0.1216, -0.0846]],

         [[ 0.0080, -0.1122, -0.0149],
          [ 0.0080, -0.1122, -0.0149],
          [ 0.0080, -0.1122, -0.0149]],

         [[-0.0852,  0.0217,  0.0571],
          [-0.0852,  0.0217,  0.0571],
          [-0.0852,  0.0217,  0.0571]]],


        [[[ 0.0193, -0.1216, -0.0846],
          [ 0.0193, -0.1216, -0.0846],
          [ 0.0193, -0.1216, -0.0846]],

         [[ 0.0080, -0.1122, -0.0149],
          [ 0.0080, -0.1122, -0.0149],
          [ 0.0080, -0.1122, -0.0149]],

         [[-0.0852,  0.0217,  0.0571],
          [-0.0852,  0.0217,  0.0571],
          [-0.0852,  0.0217,  0.0571]]],


        [[[ 0.0193, -0.1216, -0.0846],
          [ 0.0193, -0.1216, -0.0846],
          [ 0.0193, -0.1216, -0.0846]],

         [[ 0.0080, -0.1122, -0.0149],
          [ 0.0080, -0.1122, -0.0149],
          [ 0.0080, -0.1122, -0.0149]],

         [[-0.0852,  0.0217,  0.0571],
          [-0.0852,  0.0217,  0.0571],
          [-0.0852,  0.0217,  0.0571]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.scale: 

```python
0.1767766952966369
```


q.shape: 

```python
torch.Size([4, 12, 49, 32])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>计算q和k相似性的得分</div>


: 

```python
attn = (q @ k.transpose(-2, -1))
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.2655, -0.2655, -0.2655],
          [-0.2655, -0.2655, -0.2655],
          [-0.2655, -0.2655, -0.2655]],

         [[ 0.0115,  0.0115,  0.0115],
          [ 0.0115,  0.0115,  0.0115],
          [ 0.0115,  0.0115,  0.0115]],

         [[ 0.2236,  0.2236,  0.2236],
          [ 0.2236,  0.2236,  0.2236],
          [ 0.2236,  0.2236,  0.2236]]],


        [[[-0.2655, -0.2655, -0.2655],
          [-0.2655, -0.2655, -0.2655],
          [-0.2655, -0.2655, -0.2655]],

         [[ 0.0115,  0.0115,  0.0115],
          [ 0.0115,  0.0115,  0.0115],
          [ 0.0115,  0.0115,  0.0115]],

         [[ 0.2236,  0.2236,  0.2236],
          [ 0.2236,  0.2236,  0.2236],
          [ 0.2236,  0.2236,  0.2236]]],


        [[[-0.2655, -0.2655, -0.2655],
          [-0.2655, -0.2655, -0.2655],
          [-0.2655, -0.2655, -0.2655]],

         [[ 0.0115,  0.0115,  0.0115],
          [ 0.0115,  0.0115,  0.0115],
          [ 0.0115,  0.0115,  0.0115]],

         [[ 0.2236,  0.2236,  0.2236],
          [ 0.2236,  0.2236,  0.2236],
          [ 0.2236,  0.2236,  0.2236]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>relative_position_bias</div>


: 

```python
relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
```


relative_position_bias[:3,:3,:3]: 

```python
tensor([[[-0.0113, -0.0111, -0.0036],
         [ 0.0026, -0.0113, -0.0111],
         [ 0.0464,  0.0026, -0.0113]],

        [[-0.0003, -0.0076, -0.0022],
         [ 0.0159, -0.0003, -0.0076],
         [ 0.0047,  0.0159, -0.0003]],

        [[ 0.0107, -0.0244, -0.0179],
         [-0.0418,  0.0107, -0.0244],
         [ 0.0037, -0.0418,  0.0107]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


relative_position_bias.shape: 

```python
torch.Size([12, 49, 49])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>attn加上bias操作</div>


: 

```python
attn = attn + relative_position_bias.unsqueeze(0)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.2768, -0.2766, -0.2691],
          [-0.2630, -0.2768, -0.2766],
          [-0.2191, -0.2630, -0.2768]],

         [[ 0.0112,  0.0039,  0.0093],
          [ 0.0274,  0.0112,  0.0039],
          [ 0.0162,  0.0274,  0.0112]],

         [[ 0.2343,  0.1992,  0.2057],
          [ 0.1818,  0.2343,  0.1992],
          [ 0.2273,  0.1818,  0.2343]]],


        [[[-0.2768, -0.2766, -0.2691],
          [-0.2630, -0.2768, -0.2766],
          [-0.2191, -0.2630, -0.2768]],

         [[ 0.0112,  0.0039,  0.0093],
          [ 0.0274,  0.0112,  0.0039],
          [ 0.0162,  0.0274,  0.0112]],

         [[ 0.2343,  0.1992,  0.2057],
          [ 0.1818,  0.2343,  0.1992],
          [ 0.2273,  0.1818,  0.2343]]],


        [[[-0.2768, -0.2766, -0.2691],
          [-0.2630, -0.2768, -0.2766],
          [-0.2191, -0.2630, -0.2768]],

         [[ 0.0112,  0.0039,  0.0093],
          [ 0.0274,  0.0112,  0.0039],
          [ 0.0162,  0.0274,  0.0112]],

         [[ 0.2343,  0.1992,  0.2057],
          [ 0.1818,  0.2343,  0.1992],
          [ 0.2273,  0.1818,  0.2343]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>mask操作</div>


<div style='color:#fd7949;font-weight:800;font-size:23px;'>mask矩阵</div>


mask: 

```python
tensor([[[   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         ...,
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         ...,
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         ...,
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         ...,
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.]]], device='cuda:0')
```


mask.shape: 

```python
torch.Size([4, 49, 49])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>mask操作</div>


: 

```python
attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
```


attn[:3,:3,:3,:3,:3]: 

```python
tensor([[[[[-0.2768, -0.2766, -0.2691],
           [-0.2630, -0.2768, -0.2766],
           [-0.2191, -0.2630, -0.2768]],

          [[ 0.0112,  0.0039,  0.0093],
           [ 0.0274,  0.0112,  0.0039],
           [ 0.0162,  0.0274,  0.0112]],

          [[ 0.2343,  0.1992,  0.2057],
           [ 0.1818,  0.2343,  0.1992],
           [ 0.2273,  0.1818,  0.2343]]],


         [[[-0.2768, -0.2766, -0.2691],
           [-0.2630, -0.2768, -0.2766],
           [-0.2191, -0.2630, -0.2768]],

          [[ 0.0112,  0.0039,  0.0093],
           [ 0.0274,  0.0112,  0.0039],
           [ 0.0162,  0.0274,  0.0112]],

          [[ 0.2343,  0.1992,  0.2057],
           [ 0.1818,  0.2343,  0.1992],
           [ 0.2273,  0.1818,  0.2343]]],


         [[[-0.2768, -0.2766, -0.2691],
           [-0.2630, -0.2768, -0.2766],
           [-0.2191, -0.2630, -0.2768]],

          [[ 0.0112,  0.0039,  0.0093],
           [ 0.0274,  0.0112,  0.0039],
           [ 0.0162,  0.0274,  0.0112]],

          [[ 0.2343,  0.1992,  0.2057],
           [ 0.1818,  0.2343,  0.1992],
           [ 0.2273,  0.1818,  0.2343]]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([1, 4, 12, 49, 49])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>attn.view</div>


: 

```python
attn = attn.view(-1, self.num_heads, N, N)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.2768, -0.2766, -0.2691],
          [-0.2630, -0.2768, -0.2766],
          [-0.2191, -0.2630, -0.2768]],

         [[ 0.0112,  0.0039,  0.0093],
          [ 0.0274,  0.0112,  0.0039],
          [ 0.0162,  0.0274,  0.0112]],

         [[ 0.2343,  0.1992,  0.2057],
          [ 0.1818,  0.2343,  0.1992],
          [ 0.2273,  0.1818,  0.2343]]],


        [[[-0.2768, -0.2766, -0.2691],
          [-0.2630, -0.2768, -0.2766],
          [-0.2191, -0.2630, -0.2768]],

         [[ 0.0112,  0.0039,  0.0093],
          [ 0.0274,  0.0112,  0.0039],
          [ 0.0162,  0.0274,  0.0112]],

         [[ 0.2343,  0.1992,  0.2057],
          [ 0.1818,  0.2343,  0.1992],
          [ 0.2273,  0.1818,  0.2343]]],


        [[[-0.2768, -0.2766, -0.2691],
          [-0.2630, -0.2768, -0.2766],
          [-0.2191, -0.2630, -0.2768]],

         [[ 0.0112,  0.0039,  0.0093],
          [ 0.0274,  0.0112,  0.0039],
          [ 0.0162,  0.0274,  0.0112]],

         [[ 0.2343,  0.1992,  0.2057],
          [ 0.1818,  0.2343,  0.1992],
          [ 0.2273,  0.1818,  0.2343]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>计算相似性权重</div>


: 

```python
attn = self.softmax(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0202, 0.0202, 0.0203],
          [0.0204, 0.0201, 0.0201],
          [0.0214, 0.0205, 0.0202]],

         [[0.0205, 0.0203, 0.0204],
          [0.0208, 0.0205, 0.0203],
          [0.0206, 0.0208, 0.0205]],

         [[0.0207, 0.0200, 0.0201],
          [0.0197, 0.0208, 0.0200],
          [0.0206, 0.0197, 0.0207]]],


        [[[0.0354, 0.0354, 0.0356],
          [0.0358, 0.0353, 0.0353],
          [0.0373, 0.0357, 0.0352]],

         [[0.0358, 0.0356, 0.0357],
          [0.0364, 0.0358, 0.0356],
          [0.0360, 0.0364, 0.0358]],

         [[0.0363, 0.0350, 0.0353],
          [0.0345, 0.0364, 0.0351],
          [0.0361, 0.0345, 0.0363]]],


        [[[0.0355, 0.0355, 0.0358],
          [0.0359, 0.0355, 0.0355],
          [0.0376, 0.0360, 0.0355]],

         [[0.0358, 0.0355, 0.0357],
          [0.0364, 0.0358, 0.0356],
          [0.0360, 0.0364, 0.0359]],

         [[0.0364, 0.0351, 0.0354],
          [0.0346, 0.0365, 0.0352],
          [0.0361, 0.0345, 0.0364]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>drop操作</div>


: 

```python
attn = self.attn_drop(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0202, 0.0202, 0.0203],
          [0.0204, 0.0201, 0.0201],
          [0.0214, 0.0205, 0.0202]],

         [[0.0205, 0.0203, 0.0204],
          [0.0208, 0.0205, 0.0203],
          [0.0206, 0.0208, 0.0205]],

         [[0.0207, 0.0200, 0.0201],
          [0.0197, 0.0208, 0.0200],
          [0.0206, 0.0197, 0.0207]]],


        [[[0.0354, 0.0354, 0.0356],
          [0.0358, 0.0353, 0.0353],
          [0.0373, 0.0357, 0.0352]],

         [[0.0358, 0.0356, 0.0357],
          [0.0364, 0.0358, 0.0356],
          [0.0360, 0.0364, 0.0358]],

         [[0.0363, 0.0350, 0.0353],
          [0.0345, 0.0364, 0.0351],
          [0.0361, 0.0345, 0.0363]]],


        [[[0.0355, 0.0355, 0.0358],
          [0.0359, 0.0355, 0.0355],
          [0.0376, 0.0360, 0.0355]],

         [[0.0358, 0.0355, 0.0357],
          [0.0364, 0.0358, 0.0356],
          [0.0360, 0.0364, 0.0359]],

         [[0.0364, 0.0351, 0.0354],
          [0.0346, 0.0365, 0.0352],
          [0.0361, 0.0345, 0.0364]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>相似性得分乘v加堆叠多头操作</div>


: 

```python
x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
```


x[:3,:3,:3]: 

```python
tensor([[[ 0.1278, -0.1895,  0.0467],
         [ 0.1278, -0.1895,  0.0467],
         [ 0.1278, -0.1895,  0.0467]],

        [[ 0.1278, -0.1895,  0.0467],
         [ 0.1278, -0.1895,  0.0467],
         [ 0.1278, -0.1895,  0.0467]],

        [[ 0.1278, -0.1895,  0.0467],
         [ 0.1278, -0.1895,  0.0467],
         [ 0.1278, -0.1895,  0.0467]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>投影操作</div>


: 

```python
x = self.proj(x)
```


x[:3,:3,:3]: 

```python
tensor([[[ 0.0097, -0.1470, -0.0807],
         [ 0.0097, -0.1470, -0.0807],
         [ 0.0097, -0.1470, -0.0807]],

        [[ 0.0097, -0.1470, -0.0807],
         [ 0.0097, -0.1470, -0.0807],
         [ 0.0097, -0.1470, -0.0807]],

        [[ 0.0097, -0.1470, -0.0807],
         [ 0.0097, -0.1470, -0.0807],
         [ 0.0097, -0.1470, -0.0807]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.proj: 

```python
Linear(in_features=384, out_features=384, bias=True)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>投影dropout操作</div>


: 

```python
x = self.proj(x)
```


self.proj_drop: 

```python
Dropout(p=0.0, inplace=False)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0097, -0.1470, -0.0807],
         [ 0.0097, -0.1470, -0.0807],
         [ 0.0097, -0.1470, -0.0807]],

        [[ 0.0097, -0.1470, -0.0807],
         [ 0.0097, -0.1470, -0.0807],
         [ 0.0097, -0.1470, -0.0807]],

        [[ 0.0097, -0.1470, -0.0807],
         [ 0.0097, -0.1470, -0.0807],
         [ 0.0097, -0.1470, -0.0807]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


### WindowAttention 结束


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>W-MSA/SW-MSA</div>


: 

```python
attn_windows = self.attn(x_windows, mask=attn_mask)
```


attn_windows.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>merge windows（window_reverse）</div>


: 

```python
attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
```


attn_windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


### window_reverse 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>将一个个window还原成一个feature map</div>


<div style='color:#fe1111;font-weight:800;font-size:23px;'>输入</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807]],

         [[ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807]],

         [[ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807]]],


        [[[ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807]],

         [[ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807]],

         [[ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807]]],


        [[[ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807]],

         [[ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807]],

         [[ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


: 

```python
B = int(windows.shape[0] / (H * W / window_size / window_size))
```


B: 

```python
1
```


: 

```python
x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
```


x.shape: 

```python
torch.Size([1, 2, 2, 7, 7, 384])
```


: 

```python
x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807]],

         [[ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807]],

         [[ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807],
          [ 0.0097, -0.1470, -0.0807]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


### window_reverse 结束


: 

```python
shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
```


shifted_x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>reverse cyclic shift</div>


: 

```python
x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


: 

```python
x = x.view(B, H * W, C)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>残差连接</div>


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0097, -0.1470, -0.0807],
         [ 0.0097, -0.1470, -0.0807],
         [ 0.0097, -0.1470, -0.0807]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0097, -0.1470, -0.0807],
         [ 0.0097, -0.1470, -0.0807],
         [ 0.0097, -0.1470, -0.0807]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[ 0.0104, -0.1570, -0.0862,  ...,  0.1968,  0.1171, -0.3403],
         [ 0.0104, -0.1570, -0.0862,  ...,  0.1968,  0.1171, -0.3403],
         [ 0.0104, -0.1570, -0.0862,  ...,  0.1968,  0.1171, -0.3403],
         ...,
         [ 0.0104, -0.1570, -0.0862,  ...,  0.1968,  0.1171, -0.3403],
         [ 0.0104, -0.1570, -0.0862,  ...,  0.1968,  0.1171, -0.3403],
         [ 0.0104, -0.1570, -0.0862,  ...,  0.1968,  0.1171, -0.3403]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[ 0.0104, -0.1570, -0.0862,  ...,  0.1968,  0.1171, -0.3403],
         [ 0.0104, -0.1570, -0.0862,  ...,  0.1968,  0.1171, -0.3403],
         [ 0.0104, -0.1570, -0.0862,  ...,  0.1968,  0.1171, -0.3403],
         ...,
         [ 0.0104, -0.1570, -0.0862,  ...,  0.1968,  0.1171, -0.3403],
         [ 0.0104, -0.1570, -0.0862,  ...,  0.1968,  0.1171, -0.3403],
         [ 0.0104, -0.1570, -0.0862,  ...,  0.1968,  0.1171, -0.3403]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 196, 384])
```


### DropPath 结束


: 

```python
x = shortcut + self.drop_path(x)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


### Mlp 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.9822, -0.0563,  1.3500],
         [ 0.9822, -0.0563,  1.3500],
         [ 0.9822, -0.0563,  1.3500]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


self.fc1: 

```python
Linear(in_features=384, out_features=1536, bias=True)
```


self.act: 

```python
GELU()
```


self.drop1: 

```python
Dropout(p=0.0, inplace=False)
```


self.fc2: 

```python
Linear(in_features=1536, out_features=384, bias=True)
```


self.drop2: 

```python
Dropout(p=0.0, inplace=False)
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[0.0294, 0.0391, 0.0358],
         [0.0294, 0.0391, 0.0358],
         [0.0294, 0.0391, 0.0358]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


### Mlp 结束


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.0294, 0.0391, 0.0358],
         [0.0294, 0.0391, 0.0358],
         [0.0294, 0.0391, 0.0358]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.0294, 0.0391, 0.0358],
         [0.0294, 0.0391, 0.0358],
         [0.0294, 0.0391, 0.0358]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[ 0.0314,  0.0418,  0.0382,  ..., -0.0321,  0.1322,  0.0076],
         [ 0.0314,  0.0418,  0.0382,  ..., -0.0321,  0.1322,  0.0076],
         [ 0.0314,  0.0418,  0.0382,  ..., -0.0321,  0.1322,  0.0076],
         ...,
         [ 0.0314,  0.0418,  0.0382,  ..., -0.0321,  0.1322,  0.0076],
         [ 0.0314,  0.0418,  0.0382,  ..., -0.0321,  0.1322,  0.0076],
         [ 0.0314,  0.0418,  0.0382,  ..., -0.0321,  0.1322,  0.0076]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[ 0.0314,  0.0418,  0.0382,  ..., -0.0321,  0.1322,  0.0076],
         [ 0.0314,  0.0418,  0.0382,  ..., -0.0321,  0.1322,  0.0076],
         [ 0.0314,  0.0418,  0.0382,  ..., -0.0321,  0.1322,  0.0076],
         ...,
         [ 0.0314,  0.0418,  0.0382,  ..., -0.0321,  0.1322,  0.0076],
         [ 0.0314,  0.0418,  0.0382,  ..., -0.0321,  0.1322,  0.0076],
         [ 0.0314,  0.0418,  0.0382,  ..., -0.0321,  0.1322,  0.0076]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 196, 384])
```


### DropPath 结束


: 

```python
x = x + self.drop_path(self.mlp(self.norm2(x)))
```


self.norm2: 

```python
LayerNorm((384,), eps=1e-05, elementwise_affine=True)
```


self.mlp: 

```python
Mlp(
  (fc1): Linear(in_features=384, out_features=1536, bias=True)
  (act): GELU()
  (drop1): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=1536, out_features=384, bias=True)
  (drop2): Dropout(p=0.0, inplace=False)
)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.6876, -0.0456,  0.9579],
         [ 0.6876, -0.0456,  0.9579],
         [ 0.6876, -0.0456,  0.9579]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


## SwinTransformerBlock 结束


## SwinTransformerBlock 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.6876, -0.0456,  0.9579],
         [ 0.6876, -0.0456,  0.9579],
         [ 0.6876, -0.0456,  0.9579]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


attn_mask[:3,:3,:3]: 

```python
tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]], device='cuda:0')
```


attn_mask.shape: 

```python
torch.Size([4, 49, 49])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>保存输入便于做残差连接</div>


: 

```python
shortcut = x
```


shortcut.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>norm操作</div>


: 

```python
x = self.norm1(x)
```


self.norm1: 

```python
LayerNorm((384,), eps=1e-05, elementwise_affine=True)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H, W, C)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>把feature map给pad到window size的整数倍</div>


<div style='color:#3296ee;font-weight:800;font-size:23px;'>cyclic shift</div>


: 

```python
shifted_x = x
```


: 

```python
attn_mask = None
```


shifted_x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>partition windows</div>


### window_partition 开始


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>将feature map按照window_size划分成一个个没有重叠的window</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555]],

         [[ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555]],

         [[ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
```


x.shape: 

```python
torch.Size([1, 2, 7, 2, 7, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>permute操作</div>


: 

```python
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555]],

         [[ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555]],

         [[ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555]]],


        [[[ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555]],

         [[ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555]],

         [[ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555]]],


        [[[ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555]],

         [[ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555]],

         [[ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555],
          [ 0.9867, -0.0139,  1.3555]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


### window_partition 结束


: 

```python
x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
```


x_windows.shape: 

```python
torch.Size([4, 49, 384])
```


### WindowAttention 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.9867, -0.0139,  1.3555],
         [ 0.9867, -0.0139,  1.3555],
         [ 0.9867, -0.0139,  1.3555]],

        [[ 0.9867, -0.0139,  1.3555],
         [ 0.9867, -0.0139,  1.3555],
         [ 0.9867, -0.0139,  1.3555]],

        [[ 0.9867, -0.0139,  1.3555],
         [ 0.9867, -0.0139,  1.3555],
         [ 0.9867, -0.0139,  1.3555]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
```


self.qkv: 

```python
Linear(in_features=384, out_features=1152, bias=True)
```


qkv.shape: 

```python
torch.Size([3, 4, 12, 49, 32])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
q, k, v = qkv.unbind(0)
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1806, -0.2826, -0.2670],
          [ 0.1806, -0.2826, -0.2670],
          [ 0.1806, -0.2826, -0.2670]],

         [[ 0.2301,  0.6648,  0.7170],
          [ 0.2301,  0.6648,  0.7170],
          [ 0.2301,  0.6648,  0.7170]],

         [[ 0.6355, -0.0657,  0.1320],
          [ 0.6355, -0.0657,  0.1320],
          [ 0.6355, -0.0657,  0.1320]]],


        [[[ 0.1806, -0.2826, -0.2670],
          [ 0.1806, -0.2826, -0.2670],
          [ 0.1806, -0.2826, -0.2670]],

         [[ 0.2301,  0.6648,  0.7170],
          [ 0.2301,  0.6648,  0.7170],
          [ 0.2301,  0.6648,  0.7170]],

         [[ 0.6355, -0.0657,  0.1320],
          [ 0.6355, -0.0657,  0.1320],
          [ 0.6355, -0.0657,  0.1320]]],


        [[[ 0.1806, -0.2826, -0.2670],
          [ 0.1806, -0.2826, -0.2670],
          [ 0.1806, -0.2826, -0.2670]],

         [[ 0.2301,  0.6648,  0.7170],
          [ 0.2301,  0.6648,  0.7170],
          [ 0.2301,  0.6648,  0.7170]],

         [[ 0.6355, -0.0657,  0.1320],
          [ 0.6355, -0.0657,  0.1320],
          [ 0.6355, -0.0657,  0.1320]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


q.shape: 

```python
torch.Size([4, 12, 49, 32])
```


k[:3,:3,:3,:3]: 

```python
tensor([[[[-0.3139,  0.1388,  0.2388],
          [-0.3139,  0.1388,  0.2388],
          [-0.3139,  0.1388,  0.2388]],

         [[ 0.3008,  0.7725,  0.2117],
          [ 0.3008,  0.7725,  0.2117],
          [ 0.3008,  0.7725,  0.2117]],

         [[-0.0656,  0.2647,  0.1689],
          [-0.0656,  0.2647,  0.1689],
          [-0.0656,  0.2647,  0.1689]]],


        [[[-0.3139,  0.1388,  0.2388],
          [-0.3139,  0.1388,  0.2388],
          [-0.3139,  0.1388,  0.2388]],

         [[ 0.3008,  0.7725,  0.2117],
          [ 0.3008,  0.7725,  0.2117],
          [ 0.3008,  0.7725,  0.2117]],

         [[-0.0656,  0.2647,  0.1689],
          [-0.0656,  0.2647,  0.1689],
          [-0.0656,  0.2647,  0.1689]]],


        [[[-0.3139,  0.1388,  0.2388],
          [-0.3139,  0.1388,  0.2388],
          [-0.3139,  0.1388,  0.2388]],

         [[ 0.3008,  0.7725,  0.2117],
          [ 0.3008,  0.7725,  0.2117],
          [ 0.3008,  0.7725,  0.2117]],

         [[-0.0656,  0.2647,  0.1689],
          [-0.0656,  0.2647,  0.1689],
          [-0.0656,  0.2647,  0.1689]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


k.shape: 

```python
torch.Size([4, 12, 49, 32])
```


v[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1296, -0.0493,  0.1939],
          [ 0.1296, -0.0493,  0.1939],
          [ 0.1296, -0.0493,  0.1939]],

         [[-0.4836,  0.5298,  0.6458],
          [-0.4836,  0.5298,  0.6458],
          [-0.4836,  0.5298,  0.6458]],

         [[ 0.5017,  0.1170,  0.0979],
          [ 0.5017,  0.1170,  0.0979],
          [ 0.5017,  0.1170,  0.0979]]],


        [[[ 0.1296, -0.0493,  0.1939],
          [ 0.1296, -0.0493,  0.1939],
          [ 0.1296, -0.0493,  0.1939]],

         [[-0.4836,  0.5298,  0.6458],
          [-0.4836,  0.5298,  0.6458],
          [-0.4836,  0.5298,  0.6458]],

         [[ 0.5017,  0.1170,  0.0979],
          [ 0.5017,  0.1170,  0.0979],
          [ 0.5017,  0.1170,  0.0979]]],


        [[[ 0.1296, -0.0493,  0.1939],
          [ 0.1296, -0.0493,  0.1939],
          [ 0.1296, -0.0493,  0.1939]],

         [[-0.4836,  0.5298,  0.6458],
          [-0.4836,  0.5298,  0.6458],
          [-0.4836,  0.5298,  0.6458]],

         [[ 0.5017,  0.1170,  0.0979],
          [ 0.5017,  0.1170,  0.0979],
          [ 0.5017,  0.1170,  0.0979]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


v.shape: 

```python
torch.Size([4, 12, 49, 32])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>q = q * self.scale</div>


: 

```python
q = q * self.scale
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0319, -0.0499, -0.0472],
          [ 0.0319, -0.0499, -0.0472],
          [ 0.0319, -0.0499, -0.0472]],

         [[ 0.0407,  0.1175,  0.1267],
          [ 0.0407,  0.1175,  0.1267],
          [ 0.0407,  0.1175,  0.1267]],

         [[ 0.1123, -0.0116,  0.0233],
          [ 0.1123, -0.0116,  0.0233],
          [ 0.1123, -0.0116,  0.0233]]],


        [[[ 0.0319, -0.0499, -0.0472],
          [ 0.0319, -0.0499, -0.0472],
          [ 0.0319, -0.0499, -0.0472]],

         [[ 0.0407,  0.1175,  0.1267],
          [ 0.0407,  0.1175,  0.1267],
          [ 0.0407,  0.1175,  0.1267]],

         [[ 0.1123, -0.0116,  0.0233],
          [ 0.1123, -0.0116,  0.0233],
          [ 0.1123, -0.0116,  0.0233]]],


        [[[ 0.0319, -0.0499, -0.0472],
          [ 0.0319, -0.0499, -0.0472],
          [ 0.0319, -0.0499, -0.0472]],

         [[ 0.0407,  0.1175,  0.1267],
          [ 0.0407,  0.1175,  0.1267],
          [ 0.0407,  0.1175,  0.1267]],

         [[ 0.1123, -0.0116,  0.0233],
          [ 0.1123, -0.0116,  0.0233],
          [ 0.1123, -0.0116,  0.0233]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.scale: 

```python
0.1767766952966369
```


q.shape: 

```python
torch.Size([4, 12, 49, 32])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>计算q和k相似性的得分</div>


: 

```python
attn = (q @ k.transpose(-2, -1))
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.1015, -0.1015, -0.1015],
          [-0.1015, -0.1015, -0.1015],
          [-0.1015, -0.1015, -0.1015]],

         [[ 0.3183,  0.3183,  0.3183],
          [ 0.3183,  0.3183,  0.3183],
          [ 0.3183,  0.3183,  0.3183]],

         [[-0.0357, -0.0357, -0.0357],
          [-0.0357, -0.0357, -0.0357],
          [-0.0357, -0.0357, -0.0357]]],


        [[[-0.1015, -0.1015, -0.1015],
          [-0.1015, -0.1015, -0.1015],
          [-0.1015, -0.1015, -0.1015]],

         [[ 0.3183,  0.3183,  0.3183],
          [ 0.3183,  0.3183,  0.3183],
          [ 0.3183,  0.3183,  0.3183]],

         [[-0.0357, -0.0357, -0.0357],
          [-0.0357, -0.0357, -0.0357],
          [-0.0357, -0.0357, -0.0357]]],


        [[[-0.1015, -0.1015, -0.1015],
          [-0.1015, -0.1015, -0.1015],
          [-0.1015, -0.1015, -0.1015]],

         [[ 0.3183,  0.3183,  0.3183],
          [ 0.3183,  0.3183,  0.3183],
          [ 0.3183,  0.3183,  0.3183]],

         [[-0.0357, -0.0357, -0.0357],
          [-0.0357, -0.0357, -0.0357],
          [-0.0357, -0.0357, -0.0357]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>relative_position_bias</div>


: 

```python
relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
```


relative_position_bias[:3,:3,:3]: 

```python
tensor([[[-0.0274, -0.0357, -0.0134],
         [ 0.0250, -0.0274, -0.0357],
         [-0.0371,  0.0250, -0.0274]],

        [[ 0.0219, -0.0290, -0.0020],
         [ 0.0170,  0.0219, -0.0290],
         [-0.0353,  0.0170,  0.0219]],

        [[ 0.0208,  0.0035,  0.0111],
         [-0.0268,  0.0208,  0.0035],
         [-0.0184, -0.0268,  0.0208]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


relative_position_bias.shape: 

```python
torch.Size([12, 49, 49])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>attn加上bias操作</div>


: 

```python
attn = attn + relative_position_bias.unsqueeze(0)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.1289, -0.1372, -0.1149],
          [-0.0765, -0.1289, -0.1372],
          [-0.1385, -0.0765, -0.1289]],

         [[ 0.3402,  0.2893,  0.3163],
          [ 0.3353,  0.3402,  0.2893],
          [ 0.2830,  0.3353,  0.3402]],

         [[-0.0150, -0.0323, -0.0246],
          [-0.0626, -0.0150, -0.0323],
          [-0.0542, -0.0626, -0.0150]]],


        [[[-0.1289, -0.1372, -0.1149],
          [-0.0765, -0.1289, -0.1372],
          [-0.1385, -0.0765, -0.1289]],

         [[ 0.3402,  0.2893,  0.3163],
          [ 0.3353,  0.3402,  0.2893],
          [ 0.2830,  0.3353,  0.3402]],

         [[-0.0150, -0.0323, -0.0246],
          [-0.0626, -0.0150, -0.0323],
          [-0.0542, -0.0626, -0.0150]]],


        [[[-0.1289, -0.1372, -0.1149],
          [-0.0765, -0.1289, -0.1372],
          [-0.1385, -0.0765, -0.1289]],

         [[ 0.3402,  0.2893,  0.3163],
          [ 0.3353,  0.3402,  0.2893],
          [ 0.2830,  0.3353,  0.3402]],

         [[-0.0150, -0.0323, -0.0246],
          [-0.0626, -0.0150, -0.0323],
          [-0.0542, -0.0626, -0.0150]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>mask操作</div>


<div style='color:#fd7949;font-weight:800;font-size:23px;'>计算相似性权重</div>


: 

```python
attn = self.softmax(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0198, 0.0197, 0.0201],
          [0.0209, 0.0198, 0.0197],
          [0.0196, 0.0209, 0.0198]],

         [[0.0209, 0.0198, 0.0204],
          [0.0207, 0.0209, 0.0198],
          [0.0197, 0.0208, 0.0209]],

         [[0.0209, 0.0205, 0.0207],
          [0.0199, 0.0209, 0.0206],
          [0.0201, 0.0200, 0.0209]]],


        [[[0.0198, 0.0197, 0.0201],
          [0.0209, 0.0198, 0.0197],
          [0.0196, 0.0209, 0.0198]],

         [[0.0209, 0.0198, 0.0204],
          [0.0207, 0.0209, 0.0198],
          [0.0197, 0.0208, 0.0209]],

         [[0.0209, 0.0205, 0.0207],
          [0.0199, 0.0209, 0.0206],
          [0.0201, 0.0200, 0.0209]]],


        [[[0.0198, 0.0197, 0.0201],
          [0.0209, 0.0198, 0.0197],
          [0.0196, 0.0209, 0.0198]],

         [[0.0209, 0.0198, 0.0204],
          [0.0207, 0.0209, 0.0198],
          [0.0197, 0.0208, 0.0209]],

         [[0.0209, 0.0205, 0.0207],
          [0.0199, 0.0209, 0.0206],
          [0.0201, 0.0200, 0.0209]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>drop操作</div>


: 

```python
attn = self.attn_drop(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0198, 0.0197, 0.0201],
          [0.0209, 0.0198, 0.0197],
          [0.0196, 0.0209, 0.0198]],

         [[0.0209, 0.0198, 0.0204],
          [0.0207, 0.0209, 0.0198],
          [0.0197, 0.0208, 0.0209]],

         [[0.0209, 0.0205, 0.0207],
          [0.0199, 0.0209, 0.0206],
          [0.0201, 0.0200, 0.0209]]],


        [[[0.0198, 0.0197, 0.0201],
          [0.0209, 0.0198, 0.0197],
          [0.0196, 0.0209, 0.0198]],

         [[0.0209, 0.0198, 0.0204],
          [0.0207, 0.0209, 0.0198],
          [0.0197, 0.0208, 0.0209]],

         [[0.0209, 0.0205, 0.0207],
          [0.0199, 0.0209, 0.0206],
          [0.0201, 0.0200, 0.0209]]],


        [[[0.0198, 0.0197, 0.0201],
          [0.0209, 0.0198, 0.0197],
          [0.0196, 0.0209, 0.0198]],

         [[0.0209, 0.0198, 0.0204],
          [0.0207, 0.0209, 0.0198],
          [0.0197, 0.0208, 0.0209]],

         [[0.0209, 0.0205, 0.0207],
          [0.0199, 0.0209, 0.0206],
          [0.0201, 0.0200, 0.0209]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>相似性得分乘v加堆叠多头操作</div>


: 

```python
x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
```


x[:3,:3,:3]: 

```python
tensor([[[ 0.1296, -0.0493,  0.1939],
         [ 0.1296, -0.0493,  0.1939],
         [ 0.1296, -0.0493,  0.1939]],

        [[ 0.1296, -0.0493,  0.1939],
         [ 0.1296, -0.0493,  0.1939],
         [ 0.1296, -0.0493,  0.1939]],

        [[ 0.1296, -0.0493,  0.1939],
         [ 0.1296, -0.0493,  0.1939],
         [ 0.1296, -0.0493,  0.1939]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>投影操作</div>


: 

```python
x = self.proj(x)
```


x[:3,:3,:3]: 

```python
tensor([[[0.1277, 0.1201, 0.0278],
         [0.1277, 0.1201, 0.0278],
         [0.1277, 0.1201, 0.0278]],

        [[0.1277, 0.1201, 0.0278],
         [0.1277, 0.1201, 0.0278],
         [0.1277, 0.1201, 0.0278]],

        [[0.1277, 0.1201, 0.0278],
         [0.1277, 0.1201, 0.0278],
         [0.1277, 0.1201, 0.0278]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


self.proj: 

```python
Linear(in_features=384, out_features=384, bias=True)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>投影dropout操作</div>


: 

```python
x = self.proj(x)
```


self.proj_drop: 

```python
Dropout(p=0.0, inplace=False)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[0.1277, 0.1201, 0.0278],
         [0.1277, 0.1201, 0.0278],
         [0.1277, 0.1201, 0.0278]],

        [[0.1277, 0.1201, 0.0278],
         [0.1277, 0.1201, 0.0278],
         [0.1277, 0.1201, 0.0278]],

        [[0.1277, 0.1201, 0.0278],
         [0.1277, 0.1201, 0.0278],
         [0.1277, 0.1201, 0.0278]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


### WindowAttention 结束


<div style='color:#ff9702;font-weight:800;font-size:23px;'>W-MSA/SW-MSA</div>


: 

```python
attn_windows = self.attn(x_windows, mask=attn_mask)
```


attn_windows.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>merge windows（window_reverse）</div>


: 

```python
attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
```


attn_windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


### window_reverse 开始


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>将一个个window还原成一个feature map</div>


<div style='color:#fe1111;font-weight:800;font-size:23px;'>输入</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278]],

         [[0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278]],

         [[0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278]]],


        [[[0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278]],

         [[0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278]],

         [[0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278]]],


        [[[0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278]],

         [[0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278]],

         [[0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


: 

```python
B = int(windows.shape[0] / (H * W / window_size / window_size))
```


B: 

```python
1
```


: 

```python
x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
```


x.shape: 

```python
torch.Size([1, 2, 2, 7, 7, 384])
```


: 

```python
x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278]],

         [[0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278]],

         [[0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278],
          [0.1277, 0.1201, 0.0278]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


### window_reverse 结束


: 

```python
shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
```


shifted_x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>reverse cyclic shift</div>


: 

```python
x = shifted_x
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


: 

```python
x = x.view(B, H * W, C)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>残差连接</div>


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.1277, 0.1201, 0.0278],
         [0.1277, 0.1201, 0.0278],
         [0.1277, 0.1201, 0.0278]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[0.1277, 0.1201, 0.0278],
         [0.1277, 0.1201, 0.0278],
         [0.1277, 0.1201, 0.0278]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[ 0.1378,  0.1295,  0.0300,  ...,  0.2647, -0.2731,  0.1193],
         [ 0.1378,  0.1295,  0.0300,  ...,  0.2647, -0.2731,  0.1193],
         [ 0.1378,  0.1295,  0.0300,  ...,  0.2647, -0.2731,  0.1193],
         ...,
         [ 0.1378,  0.1295,  0.0300,  ...,  0.2647, -0.2731,  0.1193],
         [ 0.1378,  0.1295,  0.0300,  ...,  0.2647, -0.2731,  0.1193],
         [ 0.1378,  0.1295,  0.0300,  ...,  0.2647, -0.2731,  0.1193]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[ 0.1378,  0.1295,  0.0300,  ...,  0.2647, -0.2731,  0.1193],
         [ 0.1378,  0.1295,  0.0300,  ...,  0.2647, -0.2731,  0.1193],
         [ 0.1378,  0.1295,  0.0300,  ...,  0.2647, -0.2731,  0.1193],
         ...,
         [ 0.1378,  0.1295,  0.0300,  ...,  0.2647, -0.2731,  0.1193],
         [ 0.1378,  0.1295,  0.0300,  ...,  0.2647, -0.2731,  0.1193],
         [ 0.1378,  0.1295,  0.0300,  ...,  0.2647, -0.2731,  0.1193]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 196, 384])
```


### DropPath 结束


: 

```python
x = shortcut + self.drop_path(x)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


### Mlp 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[1.1580, 0.1608, 1.3765],
         [1.1580, 0.1608, 1.3765],
         [1.1580, 0.1608, 1.3765]]], device='cuda:0', grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


self.fc1: 

```python
Linear(in_features=384, out_features=1536, bias=True)
```


self.act: 

```python
GELU()
```


self.drop1: 

```python
Dropout(p=0.0, inplace=False)
```


self.fc2: 

```python
Linear(in_features=1536, out_features=384, bias=True)
```


self.drop2: 

```python
Dropout(p=0.0, inplace=False)
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.4392, -0.1244, -0.0876],
         [-0.4392, -0.1244, -0.0876],
         [-0.4392, -0.1244, -0.0876]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


### Mlp 结束


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.4392, -0.1244, -0.0876],
         [-0.4392, -0.1244, -0.0876],
         [-0.4392, -0.1244, -0.0876]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.4392, -0.1244, -0.0876],
         [-0.4392, -0.1244, -0.0876],
         [-0.4392, -0.1244, -0.0876]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[-0.4737, -0.1341, -0.0945,  ...,  0.0585, -0.3040,  0.0569],
         [-0.4737, -0.1341, -0.0945,  ...,  0.0585, -0.3040,  0.0569],
         [-0.4737, -0.1341, -0.0945,  ...,  0.0585, -0.3040,  0.0569],
         ...,
         [-0.4737, -0.1341, -0.0945,  ...,  0.0585, -0.3040,  0.0569],
         [-0.4737, -0.1341, -0.0945,  ...,  0.0585, -0.3040,  0.0569],
         [-0.4737, -0.1341, -0.0945,  ...,  0.0585, -0.3040,  0.0569]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[-0.4737, -0.1341, -0.0945,  ...,  0.0585, -0.3040,  0.0569],
         [-0.4737, -0.1341, -0.0945,  ...,  0.0585, -0.3040,  0.0569],
         [-0.4737, -0.1341, -0.0945,  ...,  0.0585, -0.3040,  0.0569],
         ...,
         [-0.4737, -0.1341, -0.0945,  ...,  0.0585, -0.3040,  0.0569],
         [-0.4737, -0.1341, -0.0945,  ...,  0.0585, -0.3040,  0.0569],
         [-0.4737, -0.1341, -0.0945,  ...,  0.0585, -0.3040,  0.0569]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 196, 384])
```


### DropPath 结束


: 

```python
x = x + self.drop_path(self.mlp(self.norm2(x)))
```


self.norm2: 

```python
LayerNorm((384,), eps=1e-05, elementwise_affine=True)
```


self.mlp: 

```python
Mlp(
  (fc1): Linear(in_features=384, out_features=1536, bias=True)
  (act): GELU()
  (drop1): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=1536, out_features=384, bias=True)
  (drop2): Dropout(p=0.0, inplace=False)
)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.3517, -0.0502,  0.8933],
         [ 0.3517, -0.0502,  0.8933],
         [ 0.3517, -0.0502,  0.8933]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


## SwinTransformerBlock 结束


## SwinTransformerBlock 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.3517, -0.0502,  0.8933],
         [ 0.3517, -0.0502,  0.8933],
         [ 0.3517, -0.0502,  0.8933]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


attn_mask[:3,:3,:3]: 

```python
tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]], device='cuda:0')
```


attn_mask.shape: 

```python
torch.Size([4, 49, 49])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>保存输入便于做残差连接</div>


: 

```python
shortcut = x
```


shortcut.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>norm操作</div>


: 

```python
x = self.norm1(x)
```


self.norm1: 

```python
LayerNorm((384,), eps=1e-05, elementwise_affine=True)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H, W, C)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>把feature map给pad到window size的整数倍</div>


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>cyclic shift</div>


: 

```python
shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
```


self.shift_size: 

```python
3
```


shifted_x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>partition windows</div>


### window_partition 开始


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>将feature map按照window_size划分成一个个没有重叠的window</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036]],

         [[ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036]],

         [[ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
```


x.shape: 

```python
torch.Size([1, 2, 7, 2, 7, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>permute操作</div>


: 

```python
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036]],

         [[ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036]],

         [[ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036]]],


        [[[ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036]],

         [[ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036]],

         [[ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036]]],


        [[[ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036]],

         [[ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036]],

         [[ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036],
          [ 0.4970, -0.0272,  1.2036]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


### window_partition 结束


: 

```python
x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
```


x_windows.shape: 

```python
torch.Size([4, 49, 384])
```


### WindowAttention 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.4970, -0.0272,  1.2036],
         [ 0.4970, -0.0272,  1.2036],
         [ 0.4970, -0.0272,  1.2036]],

        [[ 0.4970, -0.0272,  1.2036],
         [ 0.4970, -0.0272,  1.2036],
         [ 0.4970, -0.0272,  1.2036]],

        [[ 0.4970, -0.0272,  1.2036],
         [ 0.4970, -0.0272,  1.2036],
         [ 0.4970, -0.0272,  1.2036]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
```


self.qkv: 

```python
Linear(in_features=384, out_features=1152, bias=True)
```


qkv.shape: 

```python
torch.Size([3, 4, 12, 49, 32])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
q, k, v = qkv.unbind(0)
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[-0.2159,  0.4082, -0.0559],
          [-0.2159,  0.4082, -0.0559],
          [-0.2159,  0.4082, -0.0559]],

         [[-0.7495,  0.0295,  0.1770],
          [-0.7495,  0.0295,  0.1770],
          [-0.7495,  0.0295,  0.1770]],

         [[ 0.2641, -0.4324, -0.3819],
          [ 0.2641, -0.4324, -0.3819],
          [ 0.2641, -0.4324, -0.3819]]],


        [[[-0.2159,  0.4082, -0.0559],
          [-0.2159,  0.4082, -0.0559],
          [-0.2159,  0.4082, -0.0559]],

         [[-0.7495,  0.0295,  0.1770],
          [-0.7495,  0.0295,  0.1770],
          [-0.7495,  0.0295,  0.1770]],

         [[ 0.2641, -0.4324, -0.3819],
          [ 0.2641, -0.4324, -0.3819],
          [ 0.2641, -0.4324, -0.3819]]],


        [[[-0.2159,  0.4082, -0.0559],
          [-0.2159,  0.4082, -0.0559],
          [-0.2159,  0.4082, -0.0559]],

         [[-0.7495,  0.0295,  0.1770],
          [-0.7495,  0.0295,  0.1770],
          [-0.7495,  0.0295,  0.1770]],

         [[ 0.2641, -0.4324, -0.3819],
          [ 0.2641, -0.4324, -0.3819],
          [ 0.2641, -0.4324, -0.3819]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


q.shape: 

```python
torch.Size([4, 12, 49, 32])
```


k[:3,:3,:3,:3]: 

```python
tensor([[[[-0.4336, -0.0856, -0.1348],
          [-0.4336, -0.0856, -0.1348],
          [-0.4336, -0.0856, -0.1348]],

         [[ 0.8746, -0.0734, -0.4801],
          [ 0.8746, -0.0734, -0.4801],
          [ 0.8746, -0.0734, -0.4801]],

         [[ 0.6001, -0.2563,  0.5306],
          [ 0.6001, -0.2563,  0.5306],
          [ 0.6001, -0.2563,  0.5306]]],


        [[[-0.4336, -0.0856, -0.1348],
          [-0.4336, -0.0856, -0.1348],
          [-0.4336, -0.0856, -0.1348]],

         [[ 0.8746, -0.0734, -0.4801],
          [ 0.8746, -0.0734, -0.4801],
          [ 0.8746, -0.0734, -0.4801]],

         [[ 0.6001, -0.2563,  0.5306],
          [ 0.6001, -0.2563,  0.5306],
          [ 0.6001, -0.2563,  0.5306]]],


        [[[-0.4336, -0.0856, -0.1348],
          [-0.4336, -0.0856, -0.1348],
          [-0.4336, -0.0856, -0.1348]],

         [[ 0.8746, -0.0734, -0.4801],
          [ 0.8746, -0.0734, -0.4801],
          [ 0.8746, -0.0734, -0.4801]],

         [[ 0.6001, -0.2563,  0.5306],
          [ 0.6001, -0.2563,  0.5306],
          [ 0.6001, -0.2563,  0.5306]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


k.shape: 

```python
torch.Size([4, 12, 49, 32])
```


v[:3,:3,:3,:3]: 

```python
tensor([[[[-0.1971,  0.1862, -0.2834],
          [-0.1971,  0.1862, -0.2834],
          [-0.1971,  0.1862, -0.2834]],

         [[-0.5007,  0.0328, -0.0776],
          [-0.5007,  0.0328, -0.0776],
          [-0.5007,  0.0328, -0.0776]],

         [[-0.0036,  0.2021,  0.1711],
          [-0.0036,  0.2021,  0.1711],
          [-0.0036,  0.2021,  0.1711]]],


        [[[-0.1971,  0.1862, -0.2834],
          [-0.1971,  0.1862, -0.2834],
          [-0.1971,  0.1862, -0.2834]],

         [[-0.5007,  0.0328, -0.0776],
          [-0.5007,  0.0328, -0.0776],
          [-0.5007,  0.0328, -0.0776]],

         [[-0.0036,  0.2021,  0.1711],
          [-0.0036,  0.2021,  0.1711],
          [-0.0036,  0.2021,  0.1711]]],


        [[[-0.1971,  0.1862, -0.2834],
          [-0.1971,  0.1862, -0.2834],
          [-0.1971,  0.1862, -0.2834]],

         [[-0.5007,  0.0328, -0.0776],
          [-0.5007,  0.0328, -0.0776],
          [-0.5007,  0.0328, -0.0776]],

         [[-0.0036,  0.2021,  0.1711],
          [-0.0036,  0.2021,  0.1711],
          [-0.0036,  0.2021,  0.1711]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


v.shape: 

```python
torch.Size([4, 12, 49, 32])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>q = q * self.scale</div>


: 

```python
q = q * self.scale
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0382,  0.0722, -0.0099],
          [-0.0382,  0.0722, -0.0099],
          [-0.0382,  0.0722, -0.0099]],

         [[-0.1325,  0.0052,  0.0313],
          [-0.1325,  0.0052,  0.0313],
          [-0.1325,  0.0052,  0.0313]],

         [[ 0.0467, -0.0764, -0.0675],
          [ 0.0467, -0.0764, -0.0675],
          [ 0.0467, -0.0764, -0.0675]]],


        [[[-0.0382,  0.0722, -0.0099],
          [-0.0382,  0.0722, -0.0099],
          [-0.0382,  0.0722, -0.0099]],

         [[-0.1325,  0.0052,  0.0313],
          [-0.1325,  0.0052,  0.0313],
          [-0.1325,  0.0052,  0.0313]],

         [[ 0.0467, -0.0764, -0.0675],
          [ 0.0467, -0.0764, -0.0675],
          [ 0.0467, -0.0764, -0.0675]]],


        [[[-0.0382,  0.0722, -0.0099],
          [-0.0382,  0.0722, -0.0099],
          [-0.0382,  0.0722, -0.0099]],

         [[-0.1325,  0.0052,  0.0313],
          [-0.1325,  0.0052,  0.0313],
          [-0.1325,  0.0052,  0.0313]],

         [[ 0.0467, -0.0764, -0.0675],
          [ 0.0467, -0.0764, -0.0675],
          [ 0.0467, -0.0764, -0.0675]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.scale: 

```python
0.1767766952966369
```


q.shape: 

```python
torch.Size([4, 12, 49, 32])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>计算q和k相似性的得分</div>


: 

```python
attn = (q @ k.transpose(-2, -1))
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.1726, -0.1726, -0.1726],
          [-0.1726, -0.1726, -0.1726],
          [-0.1726, -0.1726, -0.1726]],

         [[-0.2509, -0.2509, -0.2509],
          [-0.2509, -0.2509, -0.2509],
          [-0.2509, -0.2509, -0.2509]],

         [[-0.0600, -0.0600, -0.0600],
          [-0.0600, -0.0600, -0.0600],
          [-0.0600, -0.0600, -0.0600]]],


        [[[-0.1726, -0.1726, -0.1726],
          [-0.1726, -0.1726, -0.1726],
          [-0.1726, -0.1726, -0.1726]],

         [[-0.2509, -0.2509, -0.2509],
          [-0.2509, -0.2509, -0.2509],
          [-0.2509, -0.2509, -0.2509]],

         [[-0.0600, -0.0600, -0.0600],
          [-0.0600, -0.0600, -0.0600],
          [-0.0600, -0.0600, -0.0600]]],


        [[[-0.1726, -0.1726, -0.1726],
          [-0.1726, -0.1726, -0.1726],
          [-0.1726, -0.1726, -0.1726]],

         [[-0.2509, -0.2509, -0.2509],
          [-0.2509, -0.2509, -0.2509],
          [-0.2509, -0.2509, -0.2509]],

         [[-0.0600, -0.0600, -0.0600],
          [-0.0600, -0.0600, -0.0600],
          [-0.0600, -0.0600, -0.0600]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>relative_position_bias</div>


: 

```python
relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
```


relative_position_bias[:3,:3,:3]: 

```python
tensor([[[-0.0308, -0.0290, -0.0258],
         [-0.0415, -0.0308, -0.0290],
         [-0.0068, -0.0415, -0.0308]],

        [[-0.0120,  0.0076,  0.0134],
         [-0.0128, -0.0120,  0.0076],
         [ 0.0483, -0.0128, -0.0120]],

        [[-0.0138,  0.0147, -0.0154],
         [ 0.0149, -0.0138,  0.0147],
         [ 0.0312,  0.0149, -0.0138]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


relative_position_bias.shape: 

```python
torch.Size([12, 49, 49])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>attn加上bias操作</div>


: 

```python
attn = attn + relative_position_bias.unsqueeze(0)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.2034, -0.2016, -0.1984],
          [-0.2141, -0.2034, -0.2016],
          [-0.1794, -0.2141, -0.2034]],

         [[-0.2629, -0.2433, -0.2375],
          [-0.2637, -0.2629, -0.2433],
          [-0.2026, -0.2637, -0.2629]],

         [[-0.0739, -0.0453, -0.0754],
          [-0.0452, -0.0739, -0.0453],
          [-0.0289, -0.0452, -0.0739]]],


        [[[-0.2034, -0.2016, -0.1984],
          [-0.2141, -0.2034, -0.2016],
          [-0.1794, -0.2141, -0.2034]],

         [[-0.2629, -0.2433, -0.2375],
          [-0.2637, -0.2629, -0.2433],
          [-0.2026, -0.2637, -0.2629]],

         [[-0.0739, -0.0453, -0.0754],
          [-0.0452, -0.0739, -0.0453],
          [-0.0289, -0.0452, -0.0739]]],


        [[[-0.2034, -0.2016, -0.1984],
          [-0.2141, -0.2034, -0.2016],
          [-0.1794, -0.2141, -0.2034]],

         [[-0.2629, -0.2433, -0.2375],
          [-0.2637, -0.2629, -0.2433],
          [-0.2026, -0.2637, -0.2629]],

         [[-0.0739, -0.0453, -0.0755],
          [-0.0452, -0.0739, -0.0453],
          [-0.0289, -0.0452, -0.0739]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>mask操作</div>


<div style='color:#3296ee;font-weight:800;font-size:23px;'>mask矩阵</div>


mask: 

```python
tensor([[[   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         ...,
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.],
         [   0.,    0.,    0.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         ...,
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         ...,
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.]],

        [[   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         ...,
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.]]], device='cuda:0')
```


mask.shape: 

```python
torch.Size([4, 49, 49])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>mask操作</div>


: 

```python
attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
```


attn[:3,:3,:3,:3,:3]: 

```python
tensor([[[[[-0.2034, -0.2016, -0.1984],
           [-0.2141, -0.2034, -0.2016],
           [-0.1794, -0.2141, -0.2034]],

          [[-0.2629, -0.2433, -0.2375],
           [-0.2637, -0.2629, -0.2433],
           [-0.2026, -0.2637, -0.2629]],

          [[-0.0739, -0.0453, -0.0754],
           [-0.0452, -0.0739, -0.0453],
           [-0.0289, -0.0452, -0.0739]]],


         [[[-0.2034, -0.2016, -0.1984],
           [-0.2141, -0.2034, -0.2016],
           [-0.1794, -0.2141, -0.2034]],

          [[-0.2629, -0.2433, -0.2375],
           [-0.2637, -0.2629, -0.2433],
           [-0.2026, -0.2637, -0.2629]],

          [[-0.0739, -0.0453, -0.0754],
           [-0.0452, -0.0739, -0.0453],
           [-0.0289, -0.0452, -0.0739]]],


         [[[-0.2034, -0.2016, -0.1984],
           [-0.2141, -0.2034, -0.2016],
           [-0.1794, -0.2141, -0.2034]],

          [[-0.2629, -0.2433, -0.2375],
           [-0.2637, -0.2629, -0.2433],
           [-0.2026, -0.2637, -0.2629]],

          [[-0.0739, -0.0453, -0.0755],
           [-0.0452, -0.0739, -0.0453],
           [-0.0289, -0.0452, -0.0739]]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([1, 4, 12, 49, 49])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>attn.view</div>


: 

```python
attn = attn.view(-1, self.num_heads, N, N)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.2034, -0.2016, -0.1984],
          [-0.2141, -0.2034, -0.2016],
          [-0.1794, -0.2141, -0.2034]],

         [[-0.2629, -0.2433, -0.2375],
          [-0.2637, -0.2629, -0.2433],
          [-0.2026, -0.2637, -0.2629]],

         [[-0.0739, -0.0453, -0.0754],
          [-0.0452, -0.0739, -0.0453],
          [-0.0289, -0.0452, -0.0739]]],


        [[[-0.2034, -0.2016, -0.1984],
          [-0.2141, -0.2034, -0.2016],
          [-0.1794, -0.2141, -0.2034]],

         [[-0.2629, -0.2433, -0.2375],
          [-0.2637, -0.2629, -0.2433],
          [-0.2026, -0.2637, -0.2629]],

         [[-0.0739, -0.0453, -0.0754],
          [-0.0452, -0.0739, -0.0453],
          [-0.0289, -0.0452, -0.0739]]],


        [[[-0.2034, -0.2016, -0.1984],
          [-0.2141, -0.2034, -0.2016],
          [-0.1794, -0.2141, -0.2034]],

         [[-0.2629, -0.2433, -0.2375],
          [-0.2637, -0.2629, -0.2433],
          [-0.2026, -0.2637, -0.2629]],

         [[-0.0739, -0.0453, -0.0755],
          [-0.0452, -0.0739, -0.0453],
          [-0.0289, -0.0452, -0.0739]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>计算相似性权重</div>


: 

```python
attn = self.softmax(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0198, 0.0199, 0.0199],
          [0.0196, 0.0198, 0.0198],
          [0.0203, 0.0196, 0.0198]],

         [[0.0201, 0.0204, 0.0206],
          [0.0201, 0.0201, 0.0205],
          [0.0213, 0.0200, 0.0201]],

         [[0.0202, 0.0207, 0.0201],
          [0.0207, 0.0201, 0.0207],
          [0.0210, 0.0207, 0.0201]]],


        [[[0.0348, 0.0349, 0.0350],
          [0.0345, 0.0348, 0.0349],
          [0.0356, 0.0344, 0.0347]],

         [[0.0350, 0.0357, 0.0359],
          [0.0350, 0.0350, 0.0357],
          [0.0372, 0.0350, 0.0350]],

         [[0.0353, 0.0363, 0.0352],
          [0.0360, 0.0350, 0.0360],
          [0.0366, 0.0360, 0.0350]]],


        [[[0.0346, 0.0346, 0.0348],
          [0.0343, 0.0347, 0.0347],
          [0.0355, 0.0343, 0.0346]],

         [[0.0351, 0.0358, 0.0360],
          [0.0352, 0.0352, 0.0359],
          [0.0373, 0.0351, 0.0351]],

         [[0.0353, 0.0363, 0.0352],
          [0.0362, 0.0352, 0.0362],
          [0.0367, 0.0361, 0.0351]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>drop操作</div>


: 

```python
attn = self.attn_drop(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0198, 0.0199, 0.0199],
          [0.0196, 0.0198, 0.0198],
          [0.0203, 0.0196, 0.0198]],

         [[0.0201, 0.0204, 0.0206],
          [0.0201, 0.0201, 0.0205],
          [0.0213, 0.0200, 0.0201]],

         [[0.0202, 0.0207, 0.0201],
          [0.0207, 0.0201, 0.0207],
          [0.0210, 0.0207, 0.0201]]],


        [[[0.0348, 0.0349, 0.0350],
          [0.0345, 0.0348, 0.0349],
          [0.0356, 0.0344, 0.0347]],

         [[0.0350, 0.0357, 0.0359],
          [0.0350, 0.0350, 0.0357],
          [0.0372, 0.0350, 0.0350]],

         [[0.0353, 0.0363, 0.0352],
          [0.0360, 0.0350, 0.0360],
          [0.0366, 0.0360, 0.0350]]],


        [[[0.0346, 0.0346, 0.0348],
          [0.0343, 0.0347, 0.0347],
          [0.0355, 0.0343, 0.0346]],

         [[0.0351, 0.0358, 0.0360],
          [0.0352, 0.0352, 0.0359],
          [0.0373, 0.0351, 0.0351]],

         [[0.0353, 0.0363, 0.0352],
          [0.0362, 0.0352, 0.0362],
          [0.0367, 0.0361, 0.0351]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([4, 12, 49, 49])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>相似性得分乘v加堆叠多头操作</div>


: 

```python
x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
```


x[:3,:3,:3]: 

```python
tensor([[[-0.1971,  0.1862, -0.2834],
         [-0.1971,  0.1862, -0.2834],
         [-0.1971,  0.1862, -0.2834]],

        [[-0.1971,  0.1862, -0.2834],
         [-0.1971,  0.1862, -0.2834],
         [-0.1971,  0.1862, -0.2834]],

        [[-0.1971,  0.1862, -0.2834],
         [-0.1971,  0.1862, -0.2834],
         [-0.1971,  0.1862, -0.2834]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>投影操作</div>


: 

```python
x = self.proj(x)
```


x[:3,:3,:3]: 

```python
tensor([[[-0.0495, -0.1220, -0.0277],
         [-0.0495, -0.1220, -0.0277],
         [-0.0495, -0.1220, -0.0277]],

        [[-0.0495, -0.1220, -0.0277],
         [-0.0495, -0.1220, -0.0277],
         [-0.0495, -0.1220, -0.0277]],

        [[-0.0495, -0.1220, -0.0277],
         [-0.0495, -0.1220, -0.0277],
         [-0.0495, -0.1220, -0.0277]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.proj: 

```python
Linear(in_features=384, out_features=384, bias=True)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>投影dropout操作</div>


: 

```python
x = self.proj(x)
```


self.proj_drop: 

```python
Dropout(p=0.0, inplace=False)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0495, -0.1220, -0.0277],
         [-0.0495, -0.1220, -0.0277],
         [-0.0495, -0.1220, -0.0277]],

        [[-0.0495, -0.1220, -0.0277],
         [-0.0495, -0.1220, -0.0277],
         [-0.0495, -0.1220, -0.0277]],

        [[-0.0495, -0.1220, -0.0277],
         [-0.0495, -0.1220, -0.0277],
         [-0.0495, -0.1220, -0.0277]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([4, 49, 384])
```


### WindowAttention 结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>W-MSA/SW-MSA</div>


: 

```python
attn_windows = self.attn(x_windows, mask=attn_mask)
```


attn_windows.shape: 

```python
torch.Size([4, 49, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>merge windows（window_reverse）</div>


: 

```python
attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
```


attn_windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


### window_reverse 开始


<div style='color:#fe1111;font-weight:800;font-size:23px;'>将一个个window还原成一个feature map</div>


<div style='color:#3296ee;font-weight:800;font-size:23px;'>输入</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277]],

         [[-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277]],

         [[-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277]]],


        [[[-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277]],

         [[-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277]],

         [[-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277]]],


        [[[-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277]],

         [[-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277]],

         [[-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([4, 7, 7, 384])
```


: 

```python
B = int(windows.shape[0] / (H * W / window_size / window_size))
```


B: 

```python
1
```


: 

```python
x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
```


x.shape: 

```python
torch.Size([1, 2, 2, 7, 7, 384])
```


: 

```python
x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277]],

         [[-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277]],

         [[-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277],
          [-0.0495, -0.1220, -0.0277]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


### window_reverse 结束


: 

```python
shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
```


shifted_x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>reverse cyclic shift</div>


: 

```python
x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


: 

```python
x = x.view(B, H * W, C)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>残差连接</div>


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0495, -0.1220, -0.0277],
         [-0.0495, -0.1220, -0.0277],
         [-0.0495, -0.1220, -0.0277]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0495, -0.1220, -0.0277],
         [-0.0495, -0.1220, -0.0277],
         [-0.0495, -0.1220, -0.0277]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[-0.0539, -0.1328, -0.0302,  ..., -0.1298,  0.1129, -0.4380],
         [-0.0539, -0.1328, -0.0302,  ..., -0.1298,  0.1129, -0.4380],
         [-0.0539, -0.1328, -0.0302,  ..., -0.1298,  0.1129, -0.4380],
         ...,
         [-0.0539, -0.1328, -0.0302,  ..., -0.1298,  0.1129, -0.4380],
         [-0.0539, -0.1328, -0.0302,  ..., -0.1298,  0.1129, -0.4380],
         [-0.0539, -0.1328, -0.0302,  ..., -0.1298,  0.1129, -0.4380]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[-0.0539, -0.1328, -0.0302,  ..., -0.1298,  0.1129, -0.4380],
         [-0.0539, -0.1328, -0.0302,  ..., -0.1298,  0.1129, -0.4380],
         [-0.0539, -0.1328, -0.0302,  ..., -0.1298,  0.1129, -0.4380],
         ...,
         [-0.0539, -0.1328, -0.0302,  ..., -0.1298,  0.1129, -0.4380],
         [-0.0539, -0.1328, -0.0302,  ..., -0.1298,  0.1129, -0.4380],
         [-0.0539, -0.1328, -0.0302,  ..., -0.1298,  0.1129, -0.4380]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 196, 384])
```


### DropPath 结束


: 

```python
x = shortcut + self.drop_path(x)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


### Mlp 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.4069, -0.1992,  1.1198],
         [ 0.4069, -0.1992,  1.1198],
         [ 0.4070, -0.1992,  1.1198]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


self.fc1: 

```python
Linear(in_features=384, out_features=1536, bias=True)
```


self.act: 

```python
GELU()
```


self.drop1: 

```python
Dropout(p=0.0, inplace=False)
```


self.fc2: 

```python
Linear(in_features=1536, out_features=384, bias=True)
```


self.drop2: 

```python
Dropout(p=0.0, inplace=False)
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0583, -0.1822, -0.1391],
         [-0.0583, -0.1822, -0.1391],
         [-0.0583, -0.1822, -0.1391]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


### Mlp 结束


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0583, -0.1822, -0.1391],
         [-0.0583, -0.1822, -0.1391],
         [-0.0583, -0.1822, -0.1391]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.0583, -0.1822, -0.1391],
         [-0.0583, -0.1822, -0.1391],
         [-0.0583, -0.1822, -0.1391]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[-0.0635, -0.1984, -0.1515,  ..., -0.1722, -0.0448,  0.0499],
         [-0.0635, -0.1984, -0.1515,  ..., -0.1722, -0.0448,  0.0499],
         [-0.0635, -0.1984, -0.1515,  ..., -0.1722, -0.0448,  0.0499],
         ...,
         [-0.0635, -0.1984, -0.1515,  ..., -0.1722, -0.0448,  0.0499],
         [-0.0635, -0.1984, -0.1515,  ..., -0.1722, -0.0448,  0.0499],
         [-0.0635, -0.1984, -0.1515,  ..., -0.1722, -0.0448,  0.0499]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 196, 384])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[-0.0635, -0.1984, -0.1515,  ..., -0.1722, -0.0448,  0.0499],
         [-0.0635, -0.1984, -0.1515,  ..., -0.1722, -0.0448,  0.0499],
         [-0.0635, -0.1984, -0.1515,  ..., -0.1722, -0.0448,  0.0499],
         ...,
         [-0.0635, -0.1984, -0.1515,  ..., -0.1722, -0.0448,  0.0499],
         [-0.0635, -0.1984, -0.1515,  ..., -0.1722, -0.0448,  0.0499],
         [-0.0635, -0.1984, -0.1515,  ..., -0.1722, -0.0448,  0.0499]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 196, 384])
```


### DropPath 结束


: 

```python
x = x + self.drop_path(self.mlp(self.norm2(x)))
```


self.norm2: 

```python
LayerNorm((384,), eps=1e-05, elementwise_affine=True)
```


self.mlp: 

```python
Mlp(
  (fc1): Linear(in_features=384, out_features=1536, bias=True)
  (act): GELU()
  (drop1): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=1536, out_features=384, bias=True)
  (drop2): Dropout(p=0.0, inplace=False)
)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.2342, -0.3815,  0.7117],
         [ 0.2342, -0.3815,  0.7117],
         [ 0.2342, -0.3815,  0.7117]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


## SwinTransformerBlock 结束


<div style='color:#fe618e;font-weight:800;font-size:23px;'>END: for blk in self.blocks:</div>


## downsample操作


## PatchMerging 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.2342, -0.3815,  0.7117],
         [ 0.2342, -0.3815,  0.7117],
         [ 0.2342, -0.3815,  0.7117]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 196, 384])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H, W, C)
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>pad操作</div>


: 

```python
x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
```


x.shape: 

```python
torch.Size([1, 14, 14, 384])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>PatchMerging取元素操作</div>


: 

```python
x0 = x[:, 0::2, 0::2, :]
```


: 

```python
x1 = x[:, 1::2, 0::2, :]
```


: 

```python
x2 = x[:, 0::2, 1::2, :]
```


: 

```python
x3 = x[:, 1::2, 1::2, :]
```


x0.shape: 

```python
torch.Size([1, 7, 7, 384])
```


x1.shape: 

```python
torch.Size([1, 7, 7, 384])
```


x2.shape: 

```python
torch.Size([1, 7, 7, 384])
```


x3.shape: 

```python
torch.Size([1, 7, 7, 384])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>cat操作</div>


: 

```python
x = torch.cat([x0, x1, x2, x3], -1)
```


x.shape: 

```python
torch.Size([1, 7, 7, 1536])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, -1, 4 * C) 
```


x.shape: 

```python
torch.Size([1, 49, 1536])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>norm操作</div>


: 

```python
x = self.norm(x)
```


self.norm: 

```python
LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
```


x.shape: 

```python
torch.Size([1, 49, 1536])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>reduction操作</div>


: 

```python
x = self.reduction(x)
```


self.reduction: 

```python
Linear(in_features=1536, out_features=768, bias=False)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0954, -0.4715,  2.1121],
         [ 0.0954, -0.4715,  2.1121],
         [ 0.0954, -0.4715,  2.1121]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


## PatchMerging 结束


<div style='color:#3296ee;font-weight:800;font-size:23px;'>downsample操作</div>


: 

```python
x = self.downsample(x, H, W)
```


self.downsample: 

```python
PatchMerging(
  (reduction): Linear(in_features=1536, out_features=768, bias=False)
  (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0954, -0.4715,  2.1121],
         [ 0.0954, -0.4715,  2.1121],
         [ 0.0954, -0.4715,  2.1121]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


H: 

```python
7
```


W: 

```python
7
```


# BasicLayer 结束


# BasicLayer 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>A basic Swin Transformer layer for one stage</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0954, -0.4715,  2.1121],
         [ 0.0954, -0.4715,  2.1121],
         [ 0.0954, -0.4715,  2.1121]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


H: 

```python
7
```


W: 

```python
7
```


## 创建mask：self.create_mask(x, H, W)


### window_partition 开始


<div style='color:#3296ee;font-weight:800;font-size:23px;'>将feature map按照window_size划分成一个个没有重叠的window</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[4.],
          [4.],
          [4.]],

         [[4.],
          [4.],
          [4.]],

         [[4.],
          [4.],
          [4.]]]], device='cuda:0')
```


x.shape: 

```python
torch.Size([1, 7, 7, 1])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
```


x.shape: 

```python
torch.Size([1, 1, 7, 1, 7, 1])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>permute操作</div>


: 

```python
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
```


windows.shape: 

```python
torch.Size([1, 7, 7, 1])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[4.],
          [4.],
          [4.]],

         [[4.],
          [4.],
          [4.]],

         [[4.],
          [4.],
          [4.]]]], device='cuda:0')
```


windows.shape: 

```python
torch.Size([1, 7, 7, 1])
```


### window_partition 结束


<div style='color:#fe618e;font-weight:800;font-size:23px;'>创建mask</div>


: 

```python
attn_mask = self.create_mask(x, H, W)
```


attn_mask.shape: 

```python
torch.Size([1, 49, 49])
```


## for blk in self.blocks:


<div style='color:#fe618e;font-weight:800;font-size:23px;'>for blk in self.blocks:</div>


核心代码: 

```python
x = blk(x, attn_mask)
```


## SwinTransformerBlock 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0954, -0.4715,  2.1121],
         [ 0.0954, -0.4715,  2.1121],
         [ 0.0954, -0.4715,  2.1121]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


attn_mask[:3,:3,:3]: 

```python
tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]], device='cuda:0')
```


attn_mask.shape: 

```python
torch.Size([1, 49, 49])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>保存输入便于做残差连接</div>


: 

```python
shortcut = x
```


shortcut.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>norm操作</div>


: 

```python
x = self.norm1(x)
```


self.norm1: 

```python
LayerNorm((768,), eps=1e-05, elementwise_affine=True)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H, W, C)
```


x.shape: 

```python
torch.Size([1, 7, 7, 768])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>把feature map给pad到window size的整数倍</div>


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>cyclic shift</div>


: 

```python
shifted_x = x
```


: 

```python
attn_mask = None
```


shifted_x.shape: 

```python
torch.Size([1, 7, 7, 768])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>partition windows</div>


### window_partition 开始


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>将feature map按照window_size划分成一个个没有重叠的window</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1558, -0.5714,  2.7428],
          [ 0.1558, -0.5714,  2.7428],
          [ 0.1558, -0.5714,  2.7428]],

         [[ 0.1558, -0.5714,  2.7428],
          [ 0.1558, -0.5714,  2.7428],
          [ 0.1558, -0.5714,  2.7428]],

         [[ 0.1558, -0.5714,  2.7428],
          [ 0.1558, -0.5714,  2.7428],
          [ 0.1558, -0.5714,  2.7428]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 7, 7, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
```


x.shape: 

```python
torch.Size([1, 1, 7, 1, 7, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>permute操作</div>


: 

```python
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
```


windows.shape: 

```python
torch.Size([1, 7, 7, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1558, -0.5714,  2.7428],
          [ 0.1558, -0.5714,  2.7428],
          [ 0.1558, -0.5714,  2.7428]],

         [[ 0.1558, -0.5714,  2.7428],
          [ 0.1558, -0.5714,  2.7428],
          [ 0.1558, -0.5714,  2.7428]],

         [[ 0.1558, -0.5714,  2.7428],
          [ 0.1558, -0.5714,  2.7428],
          [ 0.1558, -0.5714,  2.7428]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([1, 7, 7, 768])
```


### window_partition 结束


: 

```python
x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
```


x_windows.shape: 

```python
torch.Size([1, 49, 768])
```


### WindowAttention 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.1558, -0.5714,  2.7428],
         [ 0.1558, -0.5714,  2.7428],
         [ 0.1558, -0.5714,  2.7428]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
```


self.qkv: 

```python
Linear(in_features=768, out_features=2304, bias=True)
```


qkv.shape: 

```python
torch.Size([3, 1, 24, 49, 32])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
q, k, v = qkv.unbind(0)
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.4285, -0.2380,  0.1884],
          [ 0.4285, -0.2380,  0.1884],
          [ 0.4285, -0.2380,  0.1884]],

         [[-0.2465,  0.1055, -0.3619],
          [-0.2465,  0.1055, -0.3619],
          [-0.2465,  0.1055, -0.3619]],

         [[ 0.1225, -0.7147, -0.2896],
          [ 0.1225, -0.7147, -0.2896],
          [ 0.1225, -0.7147, -0.2896]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


q.shape: 

```python
torch.Size([1, 24, 49, 32])
```


k[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1299,  0.4245,  0.0549],
          [ 0.1299,  0.4245,  0.0549],
          [ 0.1299,  0.4245,  0.0549]],

         [[ 0.4984, -0.9131,  0.1954],
          [ 0.4984, -0.9131,  0.1954],
          [ 0.4984, -0.9131,  0.1954]],

         [[-1.1780,  0.1032, -0.4881],
          [-1.1780,  0.1032, -0.4881],
          [-1.1780,  0.1032, -0.4881]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


k.shape: 

```python
torch.Size([1, 24, 49, 32])
```


v[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.6425, -0.1660, -0.2513],
          [ 0.6425, -0.1660, -0.2513],
          [ 0.6425, -0.1660, -0.2513]],

         [[-0.0953, -1.2079, -0.9602],
          [-0.0953, -1.2079, -0.9602],
          [-0.0953, -1.2079, -0.9602]],

         [[-0.2483,  0.0182, -0.1611],
          [-0.2483,  0.0182, -0.1611],
          [-0.2483,  0.0182, -0.1611]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


v.shape: 

```python
torch.Size([1, 24, 49, 32])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>q = q * self.scale</div>


: 

```python
q = q * self.scale
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.0757, -0.0421,  0.0333],
          [ 0.0757, -0.0421,  0.0333],
          [ 0.0757, -0.0421,  0.0333]],

         [[-0.0436,  0.0187, -0.0640],
          [-0.0436,  0.0187, -0.0640],
          [-0.0436,  0.0187, -0.0640]],

         [[ 0.0216, -0.1263, -0.0512],
          [ 0.0216, -0.1263, -0.0512],
          [ 0.0216, -0.1263, -0.0512]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.scale: 

```python
0.1767766952966369
```


q.shape: 

```python
torch.Size([1, 24, 49, 32])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>计算q和k相似性的得分</div>


: 

```python
attn = (q @ k.transpose(-2, -1))
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0464, -0.0464, -0.0464],
          [-0.0464, -0.0464, -0.0464],
          [-0.0464, -0.0464, -0.0464]],

         [[-0.5982, -0.5982, -0.5982],
          [-0.5982, -0.5982, -0.5982],
          [-0.5982, -0.5982, -0.5982]],

         [[-0.0234, -0.0234, -0.0234],
          [-0.0234, -0.0234, -0.0234],
          [-0.0234, -0.0234, -0.0234]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([1, 24, 49, 49])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>relative_position_bias</div>


: 

```python
relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
```


relative_position_bias[:3,:3,:3]: 

```python
tensor([[[-0.0077, -0.0307, -0.0155],
         [ 0.0105, -0.0077, -0.0307],
         [-0.0386,  0.0105, -0.0077]],

        [[ 0.0134,  0.0025, -0.0240],
         [ 0.0292,  0.0134,  0.0025],
         [-0.0106,  0.0292,  0.0134]],

        [[ 0.0132, -0.0238,  0.0343],
         [ 0.0064,  0.0132, -0.0238],
         [-0.0167,  0.0064,  0.0132]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


relative_position_bias.shape: 

```python
torch.Size([24, 49, 49])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>attn加上bias操作</div>


: 

```python
attn = attn + relative_position_bias.unsqueeze(0)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0542, -0.0771, -0.0619],
          [-0.0359, -0.0542, -0.0771],
          [-0.0850, -0.0359, -0.0542]],

         [[-0.5847, -0.5957, -0.6221],
          [-0.5690, -0.5847, -0.5957],
          [-0.6087, -0.5690, -0.5847]],

         [[-0.0101, -0.0472,  0.0110],
          [-0.0169, -0.0101, -0.0472],
          [-0.0401, -0.0169, -0.0101]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([1, 24, 49, 49])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>mask操作</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>计算相似性权重</div>


: 

```python
attn = self.softmax(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0203, 0.0199, 0.0202],
          [0.0207, 0.0203, 0.0198],
          [0.0197, 0.0207, 0.0203]],

         [[0.0206, 0.0204, 0.0199],
          [0.0210, 0.0207, 0.0205],
          [0.0202, 0.0211, 0.0207]],

         [[0.0207, 0.0199, 0.0211],
          [0.0205, 0.0206, 0.0199],
          [0.0200, 0.0205, 0.0206]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([1, 24, 49, 49])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>drop操作</div>


: 

```python
attn = self.attn_drop(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0203, 0.0199, 0.0202],
          [0.0207, 0.0203, 0.0198],
          [0.0197, 0.0207, 0.0203]],

         [[0.0206, 0.0204, 0.0199],
          [0.0210, 0.0207, 0.0205],
          [0.0202, 0.0211, 0.0207]],

         [[0.0207, 0.0199, 0.0211],
          [0.0205, 0.0206, 0.0199],
          [0.0200, 0.0205, 0.0206]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([1, 24, 49, 49])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>相似性得分乘v加堆叠多头操作</div>


: 

```python
x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
```


x[:3,:3,:3]: 

```python
tensor([[[ 0.6425, -0.1660, -0.2513],
         [ 0.6425, -0.1660, -0.2513],
         [ 0.6425, -0.1660, -0.2513]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>投影操作</div>


: 

```python
x = self.proj(x)
```


x[:3,:3,:3]: 

```python
tensor([[[ 0.3416,  0.0875, -0.2126],
         [ 0.3416,  0.0875, -0.2126],
         [ 0.3416,  0.0875, -0.2126]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.proj: 

```python
Linear(in_features=768, out_features=768, bias=True)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>投影dropout操作</div>


: 

```python
x = self.proj(x)
```


self.proj_drop: 

```python
Dropout(p=0.0, inplace=False)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.3416,  0.0875, -0.2126],
         [ 0.3416,  0.0875, -0.2126],
         [ 0.3416,  0.0875, -0.2126]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


### WindowAttention 结束


<div style='color:#fe618e;font-weight:800;font-size:23px;'>W-MSA/SW-MSA</div>


: 

```python
attn_windows = self.attn(x_windows, mask=attn_mask)
```


attn_windows.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>merge windows（window_reverse）</div>


: 

```python
attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
```


attn_windows.shape: 

```python
torch.Size([1, 7, 7, 768])
```


### window_reverse 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>将一个个window还原成一个feature map</div>


<div style='color:#ff9702;font-weight:800;font-size:23px;'>输入</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.3416,  0.0875, -0.2126],
          [ 0.3416,  0.0875, -0.2126],
          [ 0.3416,  0.0875, -0.2126]],

         [[ 0.3416,  0.0875, -0.2126],
          [ 0.3416,  0.0875, -0.2126],
          [ 0.3416,  0.0875, -0.2126]],

         [[ 0.3416,  0.0875, -0.2126],
          [ 0.3416,  0.0875, -0.2126],
          [ 0.3416,  0.0875, -0.2126]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([1, 7, 7, 768])
```


: 

```python
B = int(windows.shape[0] / (H * W / window_size / window_size))
```


B: 

```python
1
```


: 

```python
x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
```


x.shape: 

```python
torch.Size([1, 1, 1, 7, 7, 768])
```


: 

```python
x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
```


x.shape: 

```python
torch.Size([1, 7, 7, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.3416,  0.0875, -0.2126],
          [ 0.3416,  0.0875, -0.2126],
          [ 0.3416,  0.0875, -0.2126]],

         [[ 0.3416,  0.0875, -0.2126],
          [ 0.3416,  0.0875, -0.2126],
          [ 0.3416,  0.0875, -0.2126]],

         [[ 0.3416,  0.0875, -0.2126],
          [ 0.3416,  0.0875, -0.2126],
          [ 0.3416,  0.0875, -0.2126]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 7, 7, 768])
```


### window_reverse 结束


: 

```python
shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
```


shifted_x.shape: 

```python
torch.Size([1, 7, 7, 768])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>reverse cyclic shift</div>


: 

```python
x = shifted_x
```


x.shape: 

```python
torch.Size([1, 7, 7, 768])
```


: 

```python
x = x.view(B, H * W, C)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>残差连接</div>


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.3416,  0.0875, -0.2126],
         [ 0.3416,  0.0875, -0.2126],
         [ 0.3416,  0.0875, -0.2126]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.3416,  0.0875, -0.2126],
         [ 0.3416,  0.0875, -0.2126],
         [ 0.3416,  0.0875, -0.2126]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[ 0.3757,  0.0962, -0.2339,  ...,  0.5627,  0.5907, -0.2442],
         [ 0.3757,  0.0962, -0.2339,  ...,  0.5627,  0.5907, -0.2442],
         [ 0.3757,  0.0962, -0.2339,  ...,  0.5627,  0.5907, -0.2442],
         ...,
         [ 0.3757,  0.0962, -0.2339,  ...,  0.5627,  0.5907, -0.2442],
         [ 0.3757,  0.0962, -0.2339,  ...,  0.5627,  0.5907, -0.2442],
         [ 0.3757,  0.0962, -0.2339,  ...,  0.5627,  0.5907, -0.2442]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 49, 768])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[ 0.3757,  0.0962, -0.2339,  ...,  0.5627,  0.5907, -0.2442],
         [ 0.3757,  0.0962, -0.2339,  ...,  0.5627,  0.5907, -0.2442],
         [ 0.3757,  0.0962, -0.2339,  ...,  0.5627,  0.5907, -0.2442],
         ...,
         [ 0.3757,  0.0962, -0.2339,  ...,  0.5627,  0.5907, -0.2442],
         [ 0.3757,  0.0962, -0.2339,  ...,  0.5627,  0.5907, -0.2442],
         [ 0.3757,  0.0962, -0.2339,  ...,  0.5627,  0.5907, -0.2442]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 49, 768])
```


### DropPath 结束


: 

```python
x = shortcut + self.drop_path(x)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


### Mlp 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.6053, -0.4085,  2.2908],
         [ 0.6053, -0.4085,  2.2908],
         [ 0.6053, -0.4085,  2.2908]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


self.fc1: 

```python
Linear(in_features=768, out_features=3072, bias=True)
```


self.act: 

```python
GELU()
```


self.drop1: 

```python
Dropout(p=0.0, inplace=False)
```


self.fc2: 

```python
Linear(in_features=3072, out_features=768, bias=True)
```


self.drop2: 

```python
Dropout(p=0.0, inplace=False)
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.9176, -0.1333,  0.0272],
         [-0.9176, -0.1333,  0.0272],
         [-0.9176, -0.1333,  0.0272]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


### Mlp 结束


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.9176, -0.1333,  0.0272],
         [-0.9176, -0.1333,  0.0272],
         [-0.9176, -0.1333,  0.0272]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.9176, -0.1333,  0.0272],
         [-0.9176, -0.1333,  0.0272],
         [-0.9176, -0.1333,  0.0272]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[-1.0093, -0.1466,  0.0299,  ...,  0.0191,  0.2125, -0.6985],
         [-1.0093, -0.1466,  0.0299,  ...,  0.0191,  0.2125, -0.6985],
         [-1.0093, -0.1466,  0.0299,  ...,  0.0191,  0.2125, -0.6985],
         ...,
         [-1.0093, -0.1466,  0.0299,  ...,  0.0191,  0.2125, -0.6985],
         [-1.0093, -0.1466,  0.0299,  ...,  0.0191,  0.2125, -0.6985],
         [-1.0093, -0.1466,  0.0299,  ...,  0.0191,  0.2125, -0.6985]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 49, 768])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[-1.0093, -0.1466,  0.0299,  ...,  0.0191,  0.2125, -0.6985],
         [-1.0093, -0.1466,  0.0299,  ...,  0.0191,  0.2125, -0.6985],
         [-1.0093, -0.1466,  0.0299,  ...,  0.0191,  0.2125, -0.6985],
         ...,
         [-1.0093, -0.1466,  0.0299,  ...,  0.0191,  0.2125, -0.6985],
         [-1.0093, -0.1466,  0.0299,  ...,  0.0191,  0.2125, -0.6985],
         [-1.0093, -0.1466,  0.0299,  ...,  0.0191,  0.2125, -0.6985]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 49, 768])
```


### DropPath 结束


: 

```python
x = x + self.drop_path(self.mlp(self.norm2(x)))
```


self.norm2: 

```python
LayerNorm((768,), eps=1e-05, elementwise_affine=True)
```


self.mlp: 

```python
Mlp(
  (fc1): Linear(in_features=768, out_features=3072, bias=True)
  (act): GELU()
  (drop1): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=3072, out_features=768, bias=True)
  (drop2): Dropout(p=0.0, inplace=False)
)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.5382, -0.5219,  1.9081],
         [-0.5382, -0.5219,  1.9081],
         [-0.5382, -0.5219,  1.9081]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


## SwinTransformerBlock 结束


## SwinTransformerBlock 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.5382, -0.5219,  1.9081],
         [-0.5382, -0.5219,  1.9081],
         [-0.5382, -0.5219,  1.9081]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


attn_mask[:3,:3,:3]: 

```python
tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]], device='cuda:0')
```


attn_mask.shape: 

```python
torch.Size([1, 49, 49])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>保存输入便于做残差连接</div>


: 

```python
shortcut = x
```


shortcut.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>norm操作</div>


: 

```python
x = self.norm1(x)
```


self.norm1: 

```python
LayerNorm((768,), eps=1e-05, elementwise_affine=True)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H, W, C)
```


x.shape: 

```python
torch.Size([1, 7, 7, 768])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>把feature map给pad到window size的整数倍</div>


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>cyclic shift</div>


: 

```python
shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
```


self.shift_size: 

```python
3
```


shifted_x.shape: 

```python
torch.Size([1, 7, 7, 768])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>partition windows</div>


### window_partition 开始


<div style='color:#3296ee;font-weight:800;font-size:23px;'>将feature map按照window_size划分成一个个没有重叠的window</div>


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[-0.5267, -0.5089,  2.1306],
          [-0.5267, -0.5089,  2.1306],
          [-0.5267, -0.5089,  2.1306]],

         [[-0.5267, -0.5089,  2.1306],
          [-0.5267, -0.5089,  2.1306],
          [-0.5267, -0.5089,  2.1306]],

         [[-0.5267, -0.5089,  2.1306],
          [-0.5267, -0.5089,  2.1306],
          [-0.5267, -0.5089,  2.1306]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 7, 7, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>view操作</div>


: 

```python
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
```


x.shape: 

```python
torch.Size([1, 1, 7, 1, 7, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>permute操作</div>


: 

```python
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
```


windows.shape: 

```python
torch.Size([1, 7, 7, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[-0.5267, -0.5089,  2.1306],
          [-0.5267, -0.5089,  2.1306],
          [-0.5267, -0.5089,  2.1306]],

         [[-0.5267, -0.5089,  2.1306],
          [-0.5267, -0.5089,  2.1306],
          [-0.5267, -0.5089,  2.1306]],

         [[-0.5267, -0.5089,  2.1306],
          [-0.5267, -0.5089,  2.1306],
          [-0.5267, -0.5089,  2.1306]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([1, 7, 7, 768])
```


### window_partition 结束


: 

```python
x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
```


x_windows.shape: 

```python
torch.Size([1, 49, 768])
```


### WindowAttention 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.5267, -0.5089,  2.1306],
         [-0.5267, -0.5089,  2.1306],
         [-0.5267, -0.5089,  2.1306]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
```


self.qkv: 

```python
Linear(in_features=768, out_features=2304, bias=True)
```


qkv.shape: 

```python
torch.Size([3, 1, 24, 49, 32])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>qkv操作</div>


: 

```python
q, k, v = qkv.unbind(0)
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[-0.4483, -0.1345, -0.8345],
          [-0.4483, -0.1345, -0.8345],
          [-0.4483, -0.1345, -0.8345]],

         [[ 0.2226, -0.1262, -0.0359],
          [ 0.2226, -0.1262, -0.0359],
          [ 0.2226, -0.1262, -0.0359]],

         [[ 0.7216, -0.7346,  0.5829],
          [ 0.7216, -0.7346,  0.5829],
          [ 0.7216, -0.7346,  0.5829]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


q.shape: 

```python
torch.Size([1, 24, 49, 32])
```


k[:3,:3,:3,:3]: 

```python
tensor([[[[-0.2697, -0.0127, -1.1953],
          [-0.2697, -0.0127, -1.1953],
          [-0.2697, -0.0127, -1.1953]],

         [[ 0.3313,  0.5806, -0.9143],
          [ 0.3313,  0.5806, -0.9143],
          [ 0.3313,  0.5806, -0.9143]],

         [[ 0.0522,  0.2945,  0.0316],
          [ 0.0522,  0.2945,  0.0316],
          [ 0.0522,  0.2945,  0.0316]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


k.shape: 

```python
torch.Size([1, 24, 49, 32])
```


v[:3,:3,:3,:3]: 

```python
tensor([[[[-0.1017,  0.0727, -0.0369],
          [-0.1017,  0.0727, -0.0369],
          [-0.1017,  0.0727, -0.0369]],

         [[ 0.1971,  0.4739,  0.2031],
          [ 0.1971,  0.4739,  0.2031],
          [ 0.1971,  0.4739,  0.2031]],

         [[-0.1750, -0.2763, -1.0292],
          [-0.1750, -0.2763, -1.0292],
          [-0.1750, -0.2763, -1.0292]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


v.shape: 

```python
torch.Size([1, 24, 49, 32])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>q = q * self.scale</div>


: 

```python
q = q * self.scale
```


q[:3,:3,:3,:3]: 

```python
tensor([[[[-0.0792, -0.0238, -0.1475],
          [-0.0792, -0.0238, -0.1475],
          [-0.0792, -0.0238, -0.1475]],

         [[ 0.0393, -0.0223, -0.0063],
          [ 0.0393, -0.0223, -0.0063],
          [ 0.0393, -0.0223, -0.0063]],

         [[ 0.1276, -0.1299,  0.1030],
          [ 0.1276, -0.1299,  0.1030],
          [ 0.1276, -0.1299,  0.1030]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.scale: 

```python
0.1767766952966369
```


q.shape: 

```python
torch.Size([1, 24, 49, 32])
```


<div style='color:#6f67e0;font-weight:800;font-size:23px;'>计算q和k相似性的得分</div>


: 

```python
attn = (q @ k.transpose(-2, -1))
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1681,  0.1681,  0.1681],
          [ 0.1681,  0.1681,  0.1681],
          [ 0.1681,  0.1681,  0.1681]],

         [[ 0.2782,  0.2782,  0.2782],
          [ 0.2782,  0.2782,  0.2782],
          [ 0.2782,  0.2782,  0.2782]],

         [[-0.2434, -0.2434, -0.2434],
          [-0.2434, -0.2434, -0.2434],
          [-0.2434, -0.2434, -0.2434]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([1, 24, 49, 49])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>relative_position_bias</div>


: 

```python
relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
```


relative_position_bias[:3,:3,:3]: 

```python
tensor([[[ 0.0076,  0.0223, -0.0140],
         [ 0.0072,  0.0076,  0.0223],
         [ 0.0156,  0.0072,  0.0076]],

        [[-0.0114,  0.0087,  0.0055],
         [ 0.0082, -0.0114,  0.0087],
         [ 0.0012,  0.0082, -0.0114]],

        [[-0.0125, -0.0334, -0.0139],
         [ 0.0417, -0.0125, -0.0334],
         [-0.0275,  0.0417, -0.0125]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


relative_position_bias.shape: 

```python
torch.Size([24, 49, 49])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>attn加上bias操作</div>


: 

```python
attn = attn + relative_position_bias.unsqueeze(0)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1758,  0.1904,  0.1541],
          [ 0.1753,  0.1758,  0.1904],
          [ 0.1837,  0.1753,  0.1758]],

         [[ 0.2668,  0.2869,  0.2837],
          [ 0.2864,  0.2668,  0.2869],
          [ 0.2794,  0.2864,  0.2668]],

         [[-0.2559, -0.2768, -0.2573],
          [-0.2018, -0.2559, -0.2768],
          [-0.2709, -0.2018, -0.2559]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([1, 24, 49, 49])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>mask操作</div>


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>mask矩阵</div>


mask: 

```python
tensor([[[   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         [   0.,    0.,    0.,  ..., -100., -100., -100.],
         ...,
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.],
         [-100., -100., -100.,  ...,    0.,    0.,    0.]]], device='cuda:0')
```


mask.shape: 

```python
torch.Size([1, 49, 49])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>mask操作</div>


: 

```python
attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
```


attn[:3,:3,:3,:3,:3]: 

```python
tensor([[[[[ 0.1758,  0.1904,  0.1541],
           [ 0.1753,  0.1758,  0.1904],
           [ 0.1837,  0.1753,  0.1758]],

          [[ 0.2668,  0.2869,  0.2837],
           [ 0.2864,  0.2668,  0.2869],
           [ 0.2794,  0.2864,  0.2668]],

          [[-0.2559, -0.2768, -0.2573],
           [-0.2018, -0.2559, -0.2768],
           [-0.2709, -0.2018, -0.2559]]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([1, 1, 24, 49, 49])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>attn.view</div>


: 

```python
attn = attn.view(-1, self.num_heads, N, N)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.1758,  0.1904,  0.1541],
          [ 0.1753,  0.1758,  0.1904],
          [ 0.1837,  0.1753,  0.1758]],

         [[ 0.2668,  0.2869,  0.2837],
          [ 0.2864,  0.2668,  0.2869],
          [ 0.2794,  0.2864,  0.2668]],

         [[-0.2559, -0.2768, -0.2573],
          [-0.2018, -0.2559, -0.2768],
          [-0.2709, -0.2018, -0.2559]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([1, 24, 49, 49])
```


<div style='color:#fe618e;font-weight:800;font-size:23px;'>计算相似性权重</div>


: 

```python
attn = self.softmax(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0624, 0.0633, 0.0610],
          [0.0625, 0.0625, 0.0634],
          [0.0629, 0.0624, 0.0624]],

         [[0.0618, 0.0631, 0.0629],
          [0.0632, 0.0620, 0.0633],
          [0.0628, 0.0632, 0.0620]],

         [[0.0623, 0.0610, 0.0622],
          [0.0656, 0.0622, 0.0609],
          [0.0610, 0.0654, 0.0620]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([1, 24, 49, 49])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>drop操作</div>


: 

```python
attn = self.attn_drop(attn)
```


attn[:3,:3,:3,:3]: 

```python
tensor([[[[0.0624, 0.0633, 0.0610],
          [0.0625, 0.0625, 0.0634],
          [0.0629, 0.0624, 0.0624]],

         [[0.0618, 0.0631, 0.0629],
          [0.0632, 0.0620, 0.0633],
          [0.0628, 0.0632, 0.0620]],

         [[0.0623, 0.0610, 0.0622],
          [0.0656, 0.0622, 0.0609],
          [0.0610, 0.0654, 0.0620]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


attn.shape: 

```python
torch.Size([1, 24, 49, 49])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>相似性得分乘v加堆叠多头操作</div>


: 

```python
x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
```


x[:3,:3,:3]: 

```python
tensor([[[-0.1017,  0.0727, -0.0369],
         [-0.1017,  0.0727, -0.0369],
         [-0.1017,  0.0727, -0.0369]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>投影操作</div>


: 

```python
x = self.proj(x)
```


x[:3,:3,:3]: 

```python
tensor([[[ 0.2232, -0.2850, -0.2125],
         [ 0.2232, -0.2850, -0.2125],
         [ 0.2232, -0.2850, -0.2125]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


self.proj: 

```python
Linear(in_features=768, out_features=768, bias=True)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>投影dropout操作</div>


: 

```python
x = self.proj(x)
```


self.proj_drop: 

```python
Dropout(p=0.0, inplace=False)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.2232, -0.2850, -0.2125],
         [ 0.2232, -0.2850, -0.2125],
         [ 0.2232, -0.2850, -0.2125]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


### WindowAttention 结束


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>W-MSA/SW-MSA</div>


: 

```python
attn_windows = self.attn(x_windows, mask=attn_mask)
```


attn_windows.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>merge windows（window_reverse）</div>


: 

```python
attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
```


attn_windows.shape: 

```python
torch.Size([1, 7, 7, 768])
```


### window_reverse 开始


<div style='color:#fe1111;font-weight:800;font-size:23px;'>将一个个window还原成一个feature map</div>


<div style='color:#ff9702;font-weight:800;font-size:23px;'>输入</div>


windows[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.2232, -0.2850, -0.2125],
          [ 0.2232, -0.2850, -0.2125],
          [ 0.2232, -0.2850, -0.2125]],

         [[ 0.2232, -0.2850, -0.2125],
          [ 0.2232, -0.2850, -0.2125],
          [ 0.2232, -0.2850, -0.2125]],

         [[ 0.2232, -0.2850, -0.2125],
          [ 0.2232, -0.2850, -0.2125],
          [ 0.2232, -0.2850, -0.2125]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


windows.shape: 

```python
torch.Size([1, 7, 7, 768])
```


: 

```python
B = int(windows.shape[0] / (H * W / window_size / window_size))
```


B: 

```python
1
```


: 

```python
x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
```


x.shape: 

```python
torch.Size([1, 1, 1, 7, 7, 768])
```


: 

```python
x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
```


x.shape: 

```python
torch.Size([1, 7, 7, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3,:3]: 

```python
tensor([[[[ 0.2232, -0.2850, -0.2125],
          [ 0.2232, -0.2850, -0.2125],
          [ 0.2232, -0.2850, -0.2125]],

         [[ 0.2232, -0.2850, -0.2125],
          [ 0.2232, -0.2850, -0.2125],
          [ 0.2232, -0.2850, -0.2125]],

         [[ 0.2232, -0.2850, -0.2125],
          [ 0.2232, -0.2850, -0.2125],
          [ 0.2232, -0.2850, -0.2125]]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 7, 7, 768])
```


### window_reverse 结束


: 

```python
shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
```


shifted_x.shape: 

```python
torch.Size([1, 7, 7, 768])
```


<div style='color:#fe1111;font-weight:800;font-size:23px;'>reverse cyclic shift</div>


: 

```python
x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
```


x.shape: 

```python
torch.Size([1, 7, 7, 768])
```


: 

```python
x = x.view(B, H * W, C)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>残差连接</div>


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.2232, -0.2850, -0.2125],
         [ 0.2232, -0.2850, -0.2125],
         [ 0.2232, -0.2850, -0.2125]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.2232, -0.2850, -0.2125],
         [ 0.2232, -0.2850, -0.2125],
         [ 0.2232, -0.2850, -0.2125]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[ 0.2480, -0.3166, -0.2361,  ...,  0.0578, -0.1915, -0.5290],
         [ 0.2480, -0.3166, -0.2361,  ...,  0.0578, -0.1915, -0.5290],
         [ 0.2480, -0.3166, -0.2361,  ...,  0.0578, -0.1915, -0.5290],
         ...,
         [ 0.2480, -0.3166, -0.2361,  ...,  0.0578, -0.1915, -0.5290],
         [ 0.2480, -0.3166, -0.2361,  ...,  0.0578, -0.1915, -0.5290],
         [ 0.2480, -0.3166, -0.2361,  ...,  0.0578, -0.1915, -0.5290]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 49, 768])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[ 0.2480, -0.3166, -0.2361,  ...,  0.0578, -0.1915, -0.5290],
         [ 0.2480, -0.3166, -0.2361,  ...,  0.0578, -0.1915, -0.5290],
         [ 0.2480, -0.3166, -0.2361,  ...,  0.0578, -0.1915, -0.5290],
         ...,
         [ 0.2480, -0.3166, -0.2361,  ...,  0.0578, -0.1915, -0.5290],
         [ 0.2480, -0.3166, -0.2361,  ...,  0.0578, -0.1915, -0.5290],
         [ 0.2480, -0.3166, -0.2361,  ...,  0.0578, -0.1915, -0.5290]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 49, 768])
```


### DropPath 结束


: 

```python
x = shortcut + self.drop_path(x)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


### Mlp 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.2395, -0.7925,  1.7391],
         [-0.2395, -0.7925,  1.7391],
         [-0.2395, -0.7925,  1.7391]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


self.fc1: 

```python
Linear(in_features=768, out_features=3072, bias=True)
```


self.act: 

```python
GELU()
```


self.drop1: 

```python
Dropout(p=0.0, inplace=False)
```


self.fc2: 

```python
Linear(in_features=3072, out_features=768, bias=True)
```


self.drop2: 

```python
Dropout(p=0.0, inplace=False)
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0154, -0.3309, -0.1171],
         [ 0.0154, -0.3309, -0.1171],
         [ 0.0154, -0.3309, -0.1171]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


### Mlp 结束


### DropPath 开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0154, -0.3309, -0.1171],
         [ 0.0154, -0.3309, -0.1171],
         [ 0.0154, -0.3309, -0.1171]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


#### drop_path_f开始


<div style='color:#fe618e;font-weight:800;font-size:23px;'>输入</div>


x[:3,:3,:3]: 

```python
tensor([[[ 0.0154, -0.3309, -0.1171],
         [ 0.0154, -0.3309, -0.1171],
         [ 0.0154, -0.3309, -0.1171]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


output: 

```python
tensor([[[ 0.0171, -0.3677, -0.1301,  ...,  0.3523,  0.4005,  0.4140],
         [ 0.0171, -0.3677, -0.1301,  ...,  0.3523,  0.4005,  0.4140],
         [ 0.0171, -0.3677, -0.1301,  ...,  0.3523,  0.4005,  0.4140],
         ...,
         [ 0.0171, -0.3677, -0.1301,  ...,  0.3523,  0.4005,  0.4140],
         [ 0.0171, -0.3677, -0.1301,  ...,  0.3523,  0.4005,  0.4140],
         [ 0.0171, -0.3677, -0.1301,  ...,  0.3523,  0.4005,  0.4140]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


output.shape: 

```python
torch.Size([1, 49, 768])
```


#### drop_path_f结束


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


ans: 

```python
tensor([[[ 0.0171, -0.3677, -0.1301,  ...,  0.3523,  0.4005,  0.4140],
         [ 0.0171, -0.3677, -0.1301,  ...,  0.3523,  0.4005,  0.4140],
         [ 0.0171, -0.3677, -0.1301,  ...,  0.3523,  0.4005,  0.4140],
         ...,
         [ 0.0171, -0.3677, -0.1301,  ...,  0.3523,  0.4005,  0.4140],
         [ 0.0171, -0.3677, -0.1301,  ...,  0.3523,  0.4005,  0.4140],
         [ 0.0171, -0.3677, -0.1301,  ...,  0.3523,  0.4005,  0.4140]]],
       device='cuda:0', grad_fn=<MulBackward0>)
```


ans.shape: 

```python
torch.Size([1, 49, 768])
```


### DropPath 结束


: 

```python
x = x + self.drop_path(self.mlp(self.norm2(x)))
```


self.norm2: 

```python
LayerNorm((768,), eps=1e-05, elementwise_affine=True)
```


self.mlp: 

```python
Mlp(
  (fc1): Linear(in_features=768, out_features=3072, bias=True)
  (act): GELU()
  (drop1): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=3072, out_features=768, bias=True)
  (drop2): Dropout(p=0.0, inplace=False)
)
```


self.drop_path: 

```python
DropPath()
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#ff9702;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.2731, -1.2062,  1.5420],
         [-0.2731, -1.2062,  1.5420],
         [-0.2731, -1.2062,  1.5420]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


## SwinTransformerBlock 结束


<div style='color:#fe618e;font-weight:800;font-size:23px;'>END: for blk in self.blocks:</div>


## downsample操作


<div style='color:#3296ee;font-weight:800;font-size:23px;'>downsample操作</div>


: 

```python
x = self.downsample(x, H, W)
```


self.downsample: 

```python
None
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3,:3]: 

```python
tensor([[[-0.2731, -1.2062,  1.5420],
         [-0.2731, -1.2062,  1.5420],
         [-0.2731, -1.2062,  1.5420]]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


H: 

```python
7
```


W: 

```python
7
```


# BasicLayer 结束


# 分类头操作


<div style='color:#00c2ea;font-weight:800;font-size:23px;'>层归一化</div>


: 

```python
x = self.norm(x)
```


x.shape: 

```python
torch.Size([1, 49, 768])
```


<div style='color:#3296ee;font-weight:800;font-size:23px;'>平均池化</div>


: 

```python
x = self.avgpool(x.transpose(1, 2))
```


x.shape: 

```python
torch.Size([1, 768, 1])
```


<div style='color:#7fd02b;font-weight:800;font-size:23px;'>打平</div>


: 

```python
x = torch.flatten(x, 1)
```


x.shape: 

```python
torch.Size([1, 768])
```


<div style='color:#19ce8b;font-weight:800;font-size:23px;'>分类头</div>


: 

```python
x = self.head(x)
```


self.head(x): 

```python
Linear(in_features=768, out_features=1000, bias=True)
```


x.shape: 

```python
torch.Size([1, 1000])
```


<div style='color:#fd7949;font-weight:800;font-size:23px;'>输出</div>


x[:3,:3]: 

```python
tensor([[ 0.0646, -0.4972, -0.7431]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```


x.shape: 

```python
torch.Size([1, 1000])
```


# SwinTransformer 结束

torch.Size([1, 1000])
