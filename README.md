Class-Incremental Learning for Action Recognition in Videos
============================================================================================
Overview
--------

This repository contains implementation for the paper [__"Class-Incremental Learning for Action Recognition in Videos"__](https://openaccess.thecvf.com/content/ICCV2021/papers/Park_Class-Incremental_Learning_for_Action_Recognition_in_Videos_ICCV_2021_paper.pdf) by Jaeyoo Park, Minsoo Kang and Bohyung Han, __ICCV 2021__.

> We tackle catastrophic forgetting problem in the context of class-incremental learning for video recognition, which has not been explored actively despite the popularity of continual learning.
> Our framework addresses this challenging task by introducing time-channel importance maps and exploiting the importance maps for learning the representations of incoming examples via knowledge distillation.
> We also incorporate a regularization scheme in our objective function, which encourages individual features obtained from different time steps in a video to be uncorrelated and eventually improves accuracy by alleviating catastrophic forgetting.
> We evaluate the proposed approach on brand-new splits of class-incremental action recognition benchmarks constructed upon the UCF101, HMDB51, and Something-Something V2 datasets, and demonstrate the effectiveness of our algorithm in comparison to the existing continual learning methods that are originally designed for image data.

0.Installation
--------------

    git clone https://github.com/bellos1203/TCD.git
    cd TCD
    pip install -r requirements.txt


1.Preprocessing for the videos
---------------
* We follow [TSM codebase](https://github.com/mit-han-lab/temporal-shift-module) for the preprocessing.

2.Run the Model
-----------------
* To reproduce the experimental results in the paper for UCF-101, please refer to the attached script files in `scripts`.
* Other experiments can be easily reproduced with slight modifications on `.sh` files in `scripts` or other options in `opts.py`. 

3.Evaluation
-----------------
* To obtain the numbers in the paper, compute average results from all 3 runs from 3 different seeds (1000, 1993, 2021).
* Note that we only run on single seed (1000) for Something-Something V2 datasets due to the computational cost. 

Citation
--------
If you use this code for your research, please cite our paper :
  ```shell
  @inproceedings{park2021class,
  title={Class-Incremental Learning for Action Recognition in Videos},
  author={Park, Jaeyoo and Kang, Minsoo and Han, Bohyung},
  booktitle={ICCV},
  pages={13698--13707},
  year={2021}
  }
  ```


Comments
--------

If you have any questions or comments, please contact me. <bellos1203@snu.ac.kr>

Acknowledgements
--------

This code repository is basically developed based on [TSM](https://github.com/mit-han-lab/temporal-shift-module).
Some parts related to incremental learning are borrowed from [UCIR](https://github.com/hshustc/CVPR19_Incremental_Learning) and [PODNet](https://github.com/arthurdouillard/incremental_learning.pytorch).
