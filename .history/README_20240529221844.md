# Decom-CAM
![Python 3.6.5](https://img.shields.io/badge/python-3.6.5-green.svg?style=plastic)
<a href="https://drive.google.com/file/d/1eLsuVQfLIldFXFCBB5rUqExychOoof7a/view?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)


🎉🎉🎉 Our work [DecomCAM: Advancing Beyond Saliency Maps through Decomposition and Integration]() has been accepted by Neurocomputing journal, feel free to cite us [here](#Citations).

## Features
Instead of simply highlighting the salient regions of the query target, DecomCAM's innovation is additionally providing the orthogonal sub-salient maps.

<p align="center">
<img src=".\.img/cover_00.jpg" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> The illustration of Decom-CAM's orthogonal sub-saliency maps.
</p>


## Framework
Interpreting complex deep networks, notably pre-trained vision-language models (VLMs), is a formidable challenge. Current Class Activation Map (CAM) methods highlight regions revealing the model's decision-making basis but lack clear saliency maps and detailed interpretability. To bridge this gap, we propose DecomCAM, a novel decomposition-and-integration method that distills shared patterns from channel activation maps. Utilizing singular value decomposition, DecomCAM decomposes class-discriminative activation maps into orthogonal sub-saliency maps (OSSMs), which are then integrated together based on their contribution to the target concept. Extensive experiments on six benchmarks reveal that DecomCAM not only excels in locating accuracy but also achieves an optimizing balance between interpretability and computational efficiency. Further analysis unveils that OSSMs correlate with discernible object components, facilitating a granular understanding of the model's reasoning. This positions DecomCAM as a potential tool for fine-grained interpretation of advanced deep learning models.

<p align="center">
<img src=".\.img/overview_00.jpg" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Overall framework.
</p>


<p align="center">
<img src=".\.img/vis_cams2_00.jpg" height = "400" alt="" align=center />
<br><br>
<b>Figure 3.</b> Comparative experiment results of different CAMs.
</p>

## Reproducibility

 <a href="https://drive.google.com/file/d/1eLsuVQfLIldFXFCBB5rUqExychOoof7a/view?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a> 
 
 We have released our demo (interpreting CLIP model) on colab. For source code, we provide the code based on a popular deep learning interpretation repository [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam).

Please feel free to run the following command to try our DecomCAM:
```bash
#as for CLIP, you can simply assign the query target by a simple STRING in the file.
python cnn_decmo.py
python clip_decmo.py
```
