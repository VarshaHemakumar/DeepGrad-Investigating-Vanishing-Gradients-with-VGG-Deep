# DeepGrad: Investigating Vanishing Gradients with VGG-Deep

This project investigates the vanishing gradient phenomenon in deep convolutional neural networks by extending the VGG-16 architecture and comparing it with ResNet-18. Using PyTorch's gradient hooks, we track and visualize how gradient norms behave across layers during training. The project also explores architectural design experiments to deepen understanding of how CNNs learn over depth.

---

## Objective

- Experimentally demonstrate vanishing gradients in deep CNNs.
- Track gradient flow using PyTorch hooks across convolutional layers.
- Compare gradient behaviors in:
  - Shallow VGG-16
  - Deepened VGG-16 ("VGG-Deep")
  - Residual Network (ResNet-18)
- Explore the impact of:
  - Kernel size
  - Pooling strategies
  - Activation functions
  - 1×1 convolutions
  - Computational cost & learned filters

---

##  Architectures

###  VGG-16 (Baseline)
- Standard convolutional stack with max pooling and fully connected layers.
- Includes dropout and regularization techniques.

###  VGG-Deep
- VGG-16 with 4 additional convolutional layers (no regularization).
- Trained with SGD to highlight raw gradient behavior.
- ReLU activation; no batch norm, no dropout.

###  ResNet-18
- Implemented from scratch with identity residual connections.
- Compared directly with VGG-16 and VGG-Deep under identical settings.

---

##  Gradient Norm Analysis

Used `register_full_backward_hook()` on each convolutional layer to monitor gradient magnitudes during backpropagation.

###  Observations:
- VGG-Deep shows significantly smaller gradient norms in early layers.
- ResNet-18 maintains stronger gradients due to skip connections.
- Gradient norms in VGG-Deep’s deepest layers can be ~10x lower than shallow ones.

---
##  Visual Results

###  Gradient Norms Over Time

![Gradient Norms Over Time](https://github.com/user-attachments/assets/3c3b5274-f5ab-45db-8790-223373887c4b)

> The gradient norms for deeper layers progressively decrease throughout training, indicating the presence of **vanishing gradients**. This limits deeper layers' learning effectiveness, leading to slower convergence and lower final accuracy (~74.76%).

>  *Each line represents the L2 norm of gradients for a different convolutional layer, tracked over batches across epochs.*

---

###  Gradient Norms of Selected Layers (Depth Comparison)

![Gradient Norms of Selected Layers](https://github.com/user-attachments/assets/d29b8899-7130-4d3a-b8a9-86e4dc5c2ca3)

> This plot isolates the **2nd, 5th, 8th**, and **deepest** convolutional layers. The deeper the layer (e.g., red line), the more significant the gradient degradation — demonstrating the classic vanishing gradient problem.

---

###  VGG-Deep vs VGG-16 vs ResNet-18 — Model Comparison

![Comparison Plot for VGG-Deep, VGG-16, ResNet-18](https://github.com/user-attachments/assets/b78810f3-46f4-4e6c-bbe1-49edc9989939)

> **ResNet-18** outperforms both VGG variants, achieving **93.29% test accuracy** and lowest test loss (0.1843). Residual connections enable stable gradient flow and rapid convergence.  
> - **VGG-16** shows good accuracy but fluctuates across epochs.  
> - **VGG-Deep** struggles due to gradient vanishing, with slower convergence and higher loss.

---

###  Visualizing Learned Filters (First Conv Layer)

####  VGG-16 Filters

![VGG Filters](https://github.com/user-attachments/assets/cc2c3297-d515-49b5-91ad-3ce0411f3ee6)

####  ResNet-18 Filters

![ResNet Filters](https://github.com/user-attachments/assets/3c3601a0-fb44-450b-91ee-69cfdf0574c8)

> The visualized filters from the **first convolutional layers** of VGG and ResNet reveal how each model begins learning feature detectors. While both show structured patterns, ResNet filters often appear smoother, possibly due to stabilized gradient flow.



---

##  Additional Experiments

###  Kernel Size Exploration
- Compared 3×3, 5×5, and 7×7 kernels in shallow VGG-like models.
- Larger kernels increase receptive field but may overfit small datasets.

###  Pooling Strategy
- Compared Max Pooling vs Average Pooling.
- Max Pooling showed better feature retention in this task.

###  Activation Functions
- Evaluated ReLU, Leaky ReLU, ELU, GELU
- GELU improved stability but was slower to converge.

###  1×1 Convolutions
- Used for dimensionality reduction in bottleneck layers.
- Reduced parameter count while maintaining performance.

###  Computational Cost
- Compared parameter counts and memory requirements using `torchsummary`.
- ResNet-18 had fewer parameters but higher representational capacity.

---


##  Sample Model Comparison

| Model       | Optimizer | Train Acc | Test Acc | Notable Behavior               |
|-------------|-----------|-----------|----------|--------------------------------|
| VGG-16      | Adam      | 94%       | 80%      | Strong performance, slight overfit |
| VGG-Deep    | SGD       | 88%       | 70%      | Severe vanishing gradients     |
| ResNet-18   | AdamW     | 96%       | 85%      | Stable gradient flow & generalization |

---

##  Tools & Libraries

- PyTorch (nn, hooks, optim)
- Torchsummary
- Matplotlib, Seaborn
- Sklearn (metrics)

---

##  References

- [He et al. (2015) - Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [Simonyan & Zisserman (2014) - VGG](https://arxiv.org/abs/1409.1556)
- [PyTorch gradient hook docs](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_full_backward_hook.html)




