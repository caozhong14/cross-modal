## <a href = "https://arxiv.org/pdf/1603.08511.pdf"> Colorful Image Colorization  </a>

**Abstract.**
Given a grayscale photograph as input, this paper attacks the problem of hallucinating a plausible color version of the photograph.This problem is clearly underconstrained, so previous approaches have either relied on significant user interaction or resulted in desaturated col- orizations.
We propose a fully automatic approach that produces vibrant and realistic colorizations. We embrace the underlying uncertainty of the problem by posing it as a classification task and use class-rebalancing at training time to increase the diversity of colors in the result. The sys- tem is implemented as a feed-forward pass in a CNN at test time and is trained on over a million color images. 
We evaluate our algorithm using a “colorization Turing test,” asking human participants to choose between a generated and ground truth color image. Our method successfully fools humans on 32% of the trials, significantly higher than previous methods. Moreover, we show that colorization can be a powerful pretext task for self-supervised feature learning, acting as a cross-channel encoder.This approach results in state-of-the-art performance on several feature learn- ing benchmarks.

**Keywords:** Colorization, Vision for Graphics, CNNs, Self-supervised learning

**dataset** Colorization Results on ImageNet

**Conclusion**
While image colorization is a boutique computer graphics task, it is also an in- stance of a difficult pixel prediction problem in computer vision. Here we have shown that colorization with a deep CNN and a well-chosen objective function can come closer to producing results indistinguishable from real color photos. Our method not only provides a useful graphics output, but can also be viewed as a pretext task for representation learning. Although only trained to color, our network learns a representation that is surprisingly useful for object clas- sification, detection, and segmentation, performing strongly compared to other self-supervised pre-training methods.


## [Generative Image Modeling using Style and Structure Adversarial Networks](https://arxiv.org/pdf/1603.05631.pdf)
