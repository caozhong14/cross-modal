# Visually Indicated Sounds 
## 被引用的原因主要在于以下几点，
1. 利用多模态的一种数据__生成__另一种数据（图像、声音、视频、框架图、颜色等) 
   我想到，能不能借鉴统计中**EM算法**对于**混合模型**或者**缺失数据处理**的思想，设计出可以处理缺失标签数据的模型或者算法。
2. 利用__无监督学习__的方式用于预训练或者分类
   用一些特征在分布上的相似性去训练另一些特征在分布上的相似性，我想这是这个方法的根本来源。其实这跟 GAN 的思路也很像，下面的 Aytar等人的Cross-Modal Scene Networks 两篇里面着重讲了共享特征的设计。只不过其中共享特征都是基于图像的模型中**highlevel**的特征，是有明显的语义信息的。  
3. 这个数据集只有他们这个组的人在用，发了两篇文章。一篇是 [Visually Indicated Sounds](https://arxiv.org/pdf/1512.08512.pdf)，另一篇就是 [Ambient Sound Provides Supervision for Visual Learning](https://arxiv.org/pdf/1608.07017.pdf), 很明显这个团队还在继续这项工作，所以我们要做的话，要找一个好的切入点，要不然可能会被涵盖到他们的工作中。

## <a href = "https://arxiv.org/pdf/1603.08511.pdf"> Colorful Image Colorization  </a>

**Abstract.**
Given a grayscale photograph as input, this paper attacks the problem of hallucinating a plausible color version of the photograph.This problem is clearly underconstrained, so previous approaches have either relied on significant user interaction or resulted in desaturated col- orizations.
We propose a fully automatic approach that produces vibrant and realistic colorizations. We embrace the underlying uncertainty of the problem by posing it as a classification task and use class-rebalancing at training time to increase the diversity of colors in the result. The sys- tem is implemented as a feed-forward pass in a CNN at test time and is trained on over a million color images. 
We evaluate our algorithm using a “colorization Turing test,” asking human participants to choose between a generated and ground truth color image. Our method successfully fools humans on 32% of the trials, significantly higher than previous methods. Moreover, we show that colorization can be a powerful pretext task for self-supervised feature learning, acting as a cross-channel encoder.This approach results in state-of-the-art performance on several feature learn- ing benchmarks.

**Keywords:** Colorization, Vision for Graphics, CNNs, Self-supervised learning

**Relative** 
More recent works have explored feature learning via data imputation, where a held-out subset of the complete data is predicted (e.g., [7,8,9,10,11,_12_,13])
Does our method produce realistic enough colorizations to be interpretable to an off-the-shelf object classifier? We tested this by feeding our fake colorized images to a VGG network that was trained to predict ImageNet classes from real color photos. If the classifier performs well, that means the colorizations are accurate enough to be informative about object class. Using an off-the-shelf classifier to assess the realism of synthesized data has been previously suggested by _[12]_

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

## @@@[Cross-Modal Scene Networks](https://arxiv.org/pdf/1610.09003.pdf)
### 以下这两个，讲到了怎么去做 high-level 的共享特征
>>Yusuf Aytar*, Lluis Castrejon*, Carl Vondrick, Hamed Pirsiavash, Antonio Torralba
>>Y Aytar, C Vondrick, A Torralba are with Massachusetts Institute of Technol- ogy, 77 Massachusetts Ave, Cambridge, MA 02139 USA.
>>L Castrejon is with the Department of Computer Science, University of Toronto, Ontario, Canada.H Pirsiavash is with the University of Maryland Baltimore County, 1000 Hilltop Cir, ITE 342, Baltimore, MD 21250 USA Manuscript submitted October 14, 2016

**Abstract**
People can recognize scenes across many different modalities beyond natural images. In this paper, we investigate how to learn cross-modal scene representations that transfer across modalities. To study this problem, we introduce a new cross-modal scene dataset. While convolutional neural networks can categorize scenes well, they also learn an intermediate representation not aligned across modalities, which is undesirable for cross-modal transfer applications. We present methods to regularize crossmodal convolutional neural networks so that they have a shared representation that is agnostic of the modality. Our experiments suggest that our scene representation can help transfer representations across modalities for retrieval. Moreover, our visualizations suggest that units emerge in the shared representation that tend to activate on consistent concepts independently of the modality.

**Conclusion**
Humans are able to leverage knowledge and experiences independently of the modality they perceive it in, and a similar capability in machines would enable several important applications in retrieval and recognition. In this paper, we proposed an approach to learn aligned cross-modal representations without paired data. Interestingly, our experiments suggest that our approach encourages alignment to emerge in the representation automatically across modalities, even when the training data is unaligned.

## @@@[Learning Aligned Cross-Modal Representations from Weakly Aligned Data](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Castrejon_Learning_Aligned_Cross-Modal_CVPR_2016_paper.pdf)

>>Llu ́ıs Castrejo ́n∗ University of Toronto, Yusuf Aytar∗ MIT CSAIL, Carl Vondrick MIT CSAIL
>>Hamed Pirsiavash University of Maryland - Baltimore County, Antonio Torralba MIT CSAIL

**Abstract**
People can recognize scenes across many different modalities beyond natural images. In this paper, we investigate how to learn cross-modal scene representations that transfer across modalities. To study this problem, we introduce a new cross-modal scene dataset. While convolutional neural networks can categorize cross-modal scenes well, they also learn an intermediate representation not aligned across modalities, which is undesirable for cross-modal transfer applications. We present methods to regularize cross-modal convolutional neural networks so that they have a shared representation that is agnostic of the modality. Our experiments suggest that our scene representation can help transfer representations across modalities for retrieval. Moreover, our visualizations suggest that units emerge in the shared representation that tend to activate on consistent concepts independently of the modality.

**Relative** Our method learns a joint embedding for many different modalities, including different visual domains and text. Another group of works incorporate sound as another modality [28, _30_].

**Dataset** [Cross-Modal Places Dataset](http://projects.csail.mit.edu/cmplaces/download.html) based on Places Dataset

**Result**
Our goal in this paper is to learn a representation that is aligned across modalities. We show three main results that evaluate how well our methods address this problem. First, we perform cross-modal retrieval of semantically-related content. Secondly, we show visualizations of the learned representations that give a qualitative measure of how this alignment is achieved. Finally, we show we can reconstruct natural images from other modalities using the features in the aligned representation as a qualitative measure of which semantics are preserved in our cross-modal representation.


## [Shuffle and Learn: Unsupervised Learning using Temporal Order Verification](https://arxiv.org/pdf/1603.08561.pdf)

**Abstract.** In this paper, we present an approach for learning a visual representation from the raw spatiotemporal signals in videos. Our representation is learned without supervision from semantic labels. We formulate our method as an unsupervised sequential verification task, i.e., we determine whether a sequence of frames from a video is in the correct temporal order. With this simple task and no semantic labels, we learn a powerful visual representation using a Convolutional Neural Network (CNN). The representation contains complementary information to that learned from supervised image datasets like ImageNet. Qualitative results show that our method captures information that is temporally varying, such as human pose. When used as pre-training for action recognition, our method gives significant gains over learning without external data on benchmark datasets like UCF101 and HMDB51. To demonstrate its sensitivity to human pose, we show results for pose estimation on the FLIC and MPII datasets that are competitive, or better than approaches using significantly more supervision. Our method can be combined with supervised representations to provide an additional boost in accuracy.

**Keywords:** Unsupervised learning; Videos; Sequence Verification; Action Recognition; Pose Estimation; Convolutional Neural Networks

**Relative**
Several recent papers [36, 48, 50] use egomotion constraints from video to further constrain the learning. Jayaraman et al. [36] show how they can learn equivariant transforms from such constraints. Similar to our work, they use full video frames for learning with little pre-processing. Owens et al. [_51_] use audio signals from videos to learn visual representations. Another line of work [52] uses video data to mine patches which belong to the same object to learn representations useful for distinguishing objects. Typically, these approaches require significant pre-processing to create this task. While our work also uses videos, we explore them in the spirit of sequence verification for action recognition which learns from the raw video with very little pre-processing.

**Dataset** We report all our results using split 1 of the benchmark UCF101 [12]. We demonstrate the effectiveness of our unsupervised method on benchmark action recognition datasets UCF101 [12] and  dataset HMDB51 [13], and the FLIC [14] and MPII [15] pose estimation datasets.

**Discussion**
In this paper, we studied unsupervised learning from the raw spatiotemporal signal in videos. Our proposed method outperforms other existing unsupervised methods and is competitive with supervised methods. A next step to our work is to explore different types of videos and use other ‘free’ signals such as optical flow. Another direction is to use a combination of CNNs and RNNs, and to extend our tuple verification task to much longer sequences. We believe combining this with semi-supervised methods [70, 71] is a promising future direction.


## @@@[Ambient Sound Provides Supervision for Visual Learning](https://arxiv.org/pdf/1608.07017.pdf)
### 这是数据集作者接着做的下一个工作
>>Andrew Owens1, Jiajun Wu1, Josh H. McDermott1, William T. Freeman1,2, and Antonio Torralba1
>>1 Massachusetts Institute of Technology
>>2 Google Research

**Abstract.**
The sound of crashing waves, the roar of fast-moving cars – sound conveys important information about the objects in our surround- ings. In this work, we show that ambient sounds can be used as a super- visory signal for learning visual models. To demonstrate this, we train a convolutional neural network to predict a statistical summary of the sound associated with a video frame. We show that, through this pro- cess, the network learns a representation that conveys information about objects and scenes. We evaluate this representation on several recogni- tion tasks, finding that its performance is comparable to that of other state-of-the-art unsupervised learning methods. Finally, we show through visualizations that the network learns units that are selective to objects that are often associated with characteristic sounds.

**Keywords:** Sound, convolutional networks, unsupervised learning.

**Relative**
Recently, researchers have proposed many unsupervised learning methods that learn visual representations by solving prediction tasks (sometimes known as pretext tasks) for which the held-out prediction target is derived from a natural signal in the world, rather than from human annotations. This style of learning has been called “self supervision” [4] or “natural supervision” [_30_].
Our approach is closely related to recent audio-visual work [_30_] that predicts soundtracks for videos that show a person striking objects with a drumstick. 
A natural question, then, is how our model should represent sound. Perhaps the first approach that comes to mind would be to estimate a frequency spectrum at the moment in which the picture was taken, similar to [_30_]. However, this is potentially suboptimal because in natural scenes it is difficult to predict the precise timing of a sound from visual information.

**Dataset** Flickr video dataset, Places dataset, object recognition on the PASCAL VOC 2007 dataset, scene recognition task using the SUN dataset, object-selective per category when evaluating the model on the SUN and ImageNet datasets,

**Discussion**
Sound has many properties that make it useful as a supervisory training signal: it is abundantly available without human annotations, and it is known to convey information about objects and scenes. It is also complementary to visual information, and may therefore convey information not easily obtainable from unlabeled image analysis.
In this work, we proposed using ambient sound to learn visual representations. We introduced a model, based on convolutional neural networks, that predicts a statistical sound summary from a video frame. We then showed, with visualizations and experiments on recognition tasks, that the resulting image representation contains information about objects and scenes.
Here we considered one audio representation, based on sound textures, but it is natural to ask whether other audio representations would lead the model to learn about additional types of objects. To help answer this question, we would like to more systematically study the situations when sound does (and does not) tell us about objects in the visual world. Ultimately, we would like to know what object and scene structures are detectable through sound-based training, and we see our work as a step in this direction.


## [SoundNet: Learning Sound Representations from Unlabeled Video](http://papers.nips.cc/paper/6146-soundnet-learning-sound-representations-from-unlabeled-video.pdf)

**Abstract**
We learn rich natural sound representations by capitalizing on large amounts of unlabeled sound data collected in the wild. We leverage the natural synchronization between vision and sound to learn an acoustic representation using two-million unlabeled videos. Unlabeled video has the advantage that it can be economically acquired at massive scales, yet contains useful signals about natural sound. We propose a student-teacher training procedure which transfers discriminative visual knowledge from well established visual recognition models into the sound modality using unlabeled video as a bridge. Our sound representation yields significant performance improvements over the state-of-the-art results on standard benchmarks for acoustic scene/object classification. Visualizations suggest some high-level semantics automatically emerge in the sound network, even though it is trained without ground truth labels.

**Relative** 
Sound Recognition:[_26_] also investigates the relation between vision and sound modalities, but focuses on producing sound from image sequences.
Cross-Modal Learning and Unlabeled Video: Our approach is broadly inspired by efforts to model cross-modal relations [24, 14, 7, _26_] and works that leverage large amounts of unlabeled video [25, 41, 8, 40, 39].

**dataset** variety of sources available on the web (e.g., YouTube, Flickr) We evaluate classification accuracy on the DCASE dataset\ESC datasets

**Conclusion**
We propose to train deep sound networks (SoundNet) by transferring knowledge from established vision networks and large amounts of unlabeled video. The synchronous nature of videos (sound + vision) allow us to perform such a transfer which resulted in semantically rich audio representations for natural sounds. Our results show that transfer with unlabeled video is a powerful paradigm for learning sound representations. All of our experiments suggest that one may obtain better performance simply by downloading more videos, creating deeper networks, and leveraging richer vision models.


## [What makes ImageNet good for transfer learning?](https://arxiv.org/pdf/1608.08614.pdf)

**Abastract**
The tremendous success of ImageNet-trained deep features on a wide range of transfer tasks raises the question: what is it about the ImageNet dataset that makes the learnt features as good as they are? This work provides an empirical investigation into the various facets of this question, such as, looking at the importance of the amount of examples, number of classes, balance between images-per-class and classes, and the role of fine and coarse grained recognition. We pre-train CNN features on various subsets of the ImageNet dataset and evaluate transfer performance on a variety of standard vision tasks. Our overall findings suggest that most changes in the choice of pre-training data long thought to be critical, do not significantly affect transfer performance.

**Relative**
To try and force better feature generalization, more recent “self-supervised” methods use more difficult data prediction auxiliary tasks in an effort to make the CNNs “work harder”. Attempted self-supervised tasks include predictions of ego-motion [1, 16], spatial context [8, 31, 28], temporal context [41], and even color [45, 23] and sound [_30_]. While features learned using these methods often come close to ImageNet performance, to date, none have been able to beat it.

**Dataset**
object detection on PASCAL- VOC 2007 dataset (PASCAL-DET), action classification on PASCAL-VOC 2012 dataset (PASCAL-ACT-CLS) and scene classification on the SUN dataset (SUN-CLS)

**Conclusion**
In this work we analyzed factors that affect the quality of ImageNet pre-trained features for transfer learning. Our goal was not to consider alternative neural network archi- tectures, but rather to establish facts about which aspects of the training data are important for feature learning.


## [LEARNING TO PERFORM PHYSICS EXPERIMENTS VIA DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/1611.01843.pdf)

**Abstract**
When encountering novel objects, humans are able to infer a wide range of physical properties such as mass, friction and deformability by interacting with them in a goal driven way. This process of active interaction is in the same spirit as a scientist performing experiments to discover hidden facts. Recent advances in artificial intelligence have yielded machines that can achieve superhuman performance in Go, Atari, natural language processing, and complex control problems; however, it is not clear that these systems can rival the scientific intuition of even a young child. In this work we introduce a basic set of tasks that require agents to estimate properties such as mass and cohesion of objects in an interactive simulated envi- ronment where they can manipulate the objects and observe the consequences. We found that state of art deep reinforcement learning methods can learn to perform the experiments necessary to discover such hidden properties. By systematically manipulating the problem difficulty and the cost incurred by the agent for per-forming experiments, we found that agents learn different strategies that balance the cost of gathering information against the cost of making mistakes in different situations.

**Relative**
Researchers have looked at cross modal learning, for example synthesizing sounds from visual images (Owens et al., 2015), using summary statistics of audio to learn features for object recognition (Owens et al., 2016) or image colorization (Zhang et al., 2016).

**Dataset** Towers environment

**Conclusion**
Despite recent advances in artificial intelligence, machines still lack a common sense understanding of our physical world. There has been impressive progress in recognizing objects, segmenting object boundaries and even describing visual scenes with natural language. However, these tasks are not enough for machines to infer physical properties of objects such as mass, friction or deformability.
We introduce a deep reinforcement learning agent that actively interacts with physical objects to infer their hidden properties. Our approach is inspired by findings from the developmental psychology literature indicating that infants spend a lot of their early time experimenting with objects through random exploration (Smith & Gasser, 2005; Gopnik, 2012; Spelke & Kinzler, 2007). By letting our agents conduct physical experiments in an interactive simulated environment, they learn to manipulate objects and observe the consequences to infer hidden object properties. We demonstrate the efficacy of our approach on two important physical understanding tasks—inferring mass and counting the number of objects under strong visual ambiguities. Our empirical findings suggest that our agents learn different strategies for these tasks that balance the cost of gathering information against the cost of making mistakes in different situations.


## [VID2SPEECH: SPEECH RECONSTRUCTION FROM SILENT VIDEO](https://arxiv.org/pdf/1701.00495.pdf)

**Abstract**
Speechreading is a notoriously difficult task for humans to perform. In this paper we present an end-to-end model based on a convolutional neural network (CNN) for generating an intelligible acoustic speech signal from silent video frames of a speaking person. The proposed CNN generates sound features for each frame based on its neighboring frames. Waveforms are then synthesized from the learned speech features to produce intelligible speech. We show that by leveraging the automatic feature learning capabilities of a CNN, we can obtain state-of-the-art word intelligibility on the GRID dataset, and show promising results for learning out-of-vocabulary (OOV) words.

**Relative**
A major advantage of this model of learning is its nondependency on a particular segmentation of the input data into words or sub-words. It does not either need to have explicit manually-annotated labels, but rather uses “natural supervision” [_11_], in which the prediction target is derived from a natural signal in the world. A regression-based model is also vocabulary-agnostic.

**Dataset**
on the GRID dataset, and show promising results for learning out-of-vocabulary (OOV) words.

**Conclusion**
This work has proven the feasibility of reconstructing an intelligible audio speech signal from silent videos frames. OOV(out-of- vocabulary) word reconstruction was also shown to hold promise by modeling automatic speechreading as a regression problem, and using a CNN to automatically learn relevant visual features.
The work described in this paper can serve as a basis for several directions of further research. These include using a less constrained video dataset to show real-world reconstruction viability and generalizing to speaker-independent and multiple speaker reconstruction.
