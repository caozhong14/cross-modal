## <a href = "https://arxiv.org/pdf/1603.08511.pdf"> Colorful Image Colorization  </a>

**Abstract.**
Given a grayscale photograph as input, this paper attacks the problem of hallucinating a plausible color version of the photograph.This problem is clearly underconstrained, so previous approaches have either relied on significant user interaction or resulted in desaturated col- orizations.
We propose a fully automatic approach that produces vibrant and realistic colorizations. We embrace the underlying uncertainty of the problem by posing it as a classification task and use class-rebalancing at training time to increase the diversity of colors in the result. The sys- tem is implemented as a feed-forward pass in a CNN at test time and is trained on over a million color images. 
We evaluate our algorithm using a “colorization Turing test,” asking human participants to choose between a generated and ground truth color image. Our method successfully fools humans on 32% of the trials, significantly higher than previous methods. Moreover, we show that colorization can be a powerful pretext task for self-supervised feature learning, acting as a cross-channel encoder.This approach results in state-of-the-art performance on several feature learn- ing benchmarks.

**Keywords:** Colorization, Vision for Graphics, CNNs, Self-supervised learning

**Relative** 
More recent works have explored feature learning via data imputation, where a held-out subset of the complete data is predicted (e.g., [7,8,9,10,11,_12_,13])
Does our method produce realistic enough colorizations to be interpretable to an off-the-shelf ob- ject classifier? We tested this by feeding our fake colorized images to a VGG network that was trained to predict ImageNet classes from real color photos. If the classifier performs well, that means the colorizations are accurate enough to be informative about object class. Using an off-the-shelf classifier to assess the realism of synthesized data has been previously suggested by _[12]_

**Dataset** Colorization Results on ImageNet

**Conclusion**
While image colorization is a boutique computer graphics task, it is also an in- stance of a difficult pixel prediction problem in computer vision. Here we have shown that colorization with a deep CNN and a well-chosen objective function can come closer to producing results indistinguishable from real color photos. Our method not only provides a useful graphics output, but can also be viewed as a pretext task for representation learning. Although only trained to color, our network learns a representation that is surprisingly useful for object clas- sification, detection, and segmentation, performing strongly compared to other self-supervised pre-training methods.


## [Generative Image Modeling using Style and Structure Adversarial Networks](https://arxiv.org/pdf/1603.05631.pdf)

**Abstract.**
Current generative frameworks use end-to-end learning and generate images by sampling from uniform noise distribution. However, these approaches ignore the most basic principle of image formation: im- ages are product of: (a) Structure: the underlying 3D model; (b) Style: the texture mapped onto structure. In this paper, we factorize the image generation process and propose Style and Structure Generative Adversar- ial Network (S2-GAN). Our S2-GAN has two components: the Structure- GAN generates a surface normal map; the Style-GAN takes the surface normal map as input and generates the 2D image. Apart from a real vs. generated loss function, we use an additional loss with computed surface normals from generated images. The two GANs are first trained inde- pendently, and then merged together via joint learning. We show our S2-GAN model is interpretable, generates more realistic images and can be used to learn unsupervised RGBD representations.

**Relative** There are two primary approaches to unsupervised learning. The first is the discriminative approach where we use auxiliary tasks such that ground truth can be generated without labeling. Some examples of these auxiliary tasks include predicting: the relative location of two patches [2], ego-motion in videos [15,16], physical signals [_17_,18,19].

**Dataset** scene classification on SUN RGB-D dataset, object detection on NYUv2 dataset

**Conclusion**
We present a novel Style and Structure GAN which factorizes the image gen- eration process. We show our model is more interpretable and generates more realistic images compared to the baselines. We also show that our method can learn RGBD representations in an unsupervised manner.


## [Learning Aligned Cross-Modal Representations from Weakly Aligned Data](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Castrejon_Learning_Aligned_Cross-Modal_CVPR_2016_paper.pdf)

**Abstract**
People can recognize scenes across many different modalities beyond natural images. In this paper, we investigate how to learn cross-modal scene representations that transfer across modalities. To study this problem, we introduce a new cross-modal scene dataset. While convolutional neural networks can categorize cross-modal scenes well, they also learn an intermediate representation not aligned across modalities, which is undesirable for cross-modal transfer applications. We present methods to regularize cross-modal convolutional neural networks so that they have a shared representation that is agnostic of the modality. Our experiments suggest that our scene representation can help transfer representations across modalities for retrieval. Moreover, our visualizations suggest that units emerge in the shared representation that tend to activate on consistent concepts independently of the modality.

**Relative** Our method learns a joint embedding for many different modalities, including different visual domains and text. Another group of works incorporate sound as another modality [28, _30_].

**Dataset** [Cross-Modal Places Dataset](http://projects.csail.mit.edu/cmplaces/download.html) based on Places Dataset

**Result**
Our goal in this paper is to learn a representation that is aligned across modalities. We show three main results that evaluate how well our methods address this problem. First, we perform cross-modal retrieval of semantically-related content. Secondly, we show visualizations of the learned representations that give a qualitative measure of how this alignment is achieved. Finally, we show we can reconstruct natural images from other modalities using the features in the aligned representation as a qualitative measure of which semantics are preserved in our cross-modal representation.


## [Shuffle and Learn: Unsupervised Learning using Temporal Order Verification](https://arxiv.org/pdf/1603.08561.pdf)

**Abstract.** In this paper, we present an approach for learning a visual representation from the raw spatiotemporal signals in videos. Our representation is learned without supervision from semantic labels. We formulate our method as an unsupervised sequential verification task, i.e., we determine whether a sequence of frames from a video is in the correct temporal order. With this simple task and no semantic labels, we learn a powerful visual representation using a Convolutional Neural Network (CNN). The representation contains complementary information to that learned from supervised image datasets like ImageNet. Qualitative results show that our method captures information that is temporally varying, such as human pose. When used as pre-training for action recognition, our method gives significant gains over learning without external data on benchmark datasets like UCF101 and HMDB51. To demonstrate its sensitivity to human pose, we show results for pose estimation on the FLIC and MPII datasets that are competitive, or better than approaches using significantly more supervision. Our method can be combined with supervised representations to provide an additional boost in accuracy.

**Keywords:** Unsupervised learning; Videos; Sequence Verification; Ac- tion Recognition; Pose Estimation; Convolutional Neural Networks

**Relativ**
Several recent papers [36, 48, 50] use egomotion constraints from video to further constrain the learning. Jayaraman et al. [36] show how they can learn equivariant transforms from such constraints. Similar to our work, they use full video frames for learning with little pre-processing. Owens et al. [_51_] use audio signals from videos to learn visual representations. Another line of work [52] uses video data to mine patches which belong to the same object to learn representations useful for distinguishing objects. Typically, these approaches require significant pre-processing to create this task. While our work also uses videos, we explore them in the spirit of sequence verification for action recognition which learns from the raw video with very little pre-processing.

**Dataset** We report all our results using split 1 of the benchmark UCF101 [12]. We demonstrate the effectiveness of our unsupervised method on benchmark action recognition datasets UCF101 [12] and  dataset HMDB51 [13], and the FLIC [14] and MPII [15] pose estimation datasets.

**Discussion**
In this paper, we studied unsupervised learning from the raw spatiotemporal signal in videos. Our proposed method outperforms other existing unsupervised methods and is competitive with supervised methods. A next step to our work is to explore different types of videos and use other ‘free’ signals such as optical flow. Another direction is to use a combination of CNNs and RNNs, and to extend our tuple verification task to much longer sequences. We believe combining this with semi-supervised methods [70, 71] is a promising future direction.


