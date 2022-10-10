<h1 align="center">
  MLDSO
</h1>
<p align="center">
<a href="https://github.com/Littleor/MLDSO/blob/master/LICENSE" target="blank">
<img src="https://img.shields.io/github/license/Littleor/MLDSO?style=flat-square" alt="github-profile-readme-generator license" />
</a>
<a href="https://github.com/Littleor/MLDSO/fork" target="blank">
<img src="https://img.shields.io/github/forks/Littleor/MLDSO?style=flat-square" alt="github-profile-readme-generator forks"/>
</a>
<a href="https://github.com/Littleor/MLDSO/stargazers" target="blank">
<img src="https://img.shields.io/github/stars/Littleor/MLDSO?style=flat-square" alt="github-profile-readme-generator stars"/>
</a>
<a href="https://github.com/Littleor/MLDSO/issues" target="blank">
<img src="https://img.shields.io/github/issues/Littleor/MLDSO?style=flat-square" alt="github-profile-readme-generator issues"/>
</a>
<a href="https://github.com/Littleor/MLDSO/pulls" target="blank">
<img src="https://img.shields.io/github/issues-pr/Littleor/MLDSO?style=flat-square" alt="github-profile-readme-generator pull-requests"/>
</a>
</p>

> This is the condensed source codes for the paper [Few-shot bearing fault diagnosis based on meta-learning with discriminant space optimization](https://doi.org/10.1088/1361-6501/ac8303)
> which published in _Measurement Science and Technology_.
>
> **Thanks for the Case Western University Bearing Data Center for providing the experimental data.**

The repository works with Python 3.8.5 and PyTorch 1.8.1.

## Citation

If you want to use the code, please cite the following paper:

```
@article{Zhang_2022,
	doi = {10.1088/1361-6501/ac8303},
	url = {https://doi.org/10.1088/1361-6501/ac8303},
	year = 2022,
	month = {aug},
	publisher = {{IOP} Publishing},
	volume = {33},
	number = {11},
	pages = {115024},
	author = {Dengming Zhang and Kai Zheng and Yin Bai and Dengke Yao and Dewei Yang and Shaowang Wang},
	title = {Few-shot bearing fault diagnosis based on meta-learning with discriminant space optimization},
	journal = {Measurement Science and Technology},
	abstract = {In practical industrial applications, the collected fault data are usually insufficient due to the sudden occurrence of faults. However, the current deep-learning-based fault diagnosis methods often rely on a large number of samples to achieve satisfactory performance. Moreover, the heavy background noise and the variability of working conditions also degrade the performance of existing fault diagnostic approaches. To address these challenges, a new fault diagnosis method for few-shot bearing fault diagnosis based on meta-learning with discriminant space optimization (MLDSO) is proposed in this research. First, the fault feature of the rolling bearing is extracted through the tailored networks. Then, the feature extractor is optimized by the discriminant space loss proposed in this paper, to promote the clustering of the extracted fault features of the same category and to distinguish between different types of fault features. Next, the feature extractor and discriminant space optimizer are constructed to optimize the feature discriminant space; thus, a high fault-tolerant discriminant space is obtained for meta-learning. Eventually, the faults in the new task can be accurately classified with the assistance of previously learned meta-knowledge and a few known samples when dealing with new tasks under different working conditions. The effectiveness and superiority of the proposed MLDSO method are verified via the datasets collected from our self-designed experimental platform and the Case Western Reserve University test platform. The experimental results show superior performance over the advanced methods. This indicates that the proposed method is a promising approach under small sample situations, heavy noise, and variable working conditions.}
}
```

## Usage

You can use the following command to run the code:

```bash
# Clone the repository
git clone git@github.com:Littleor/MLDSO.git
# Change the directory
cd MLDSO
# Install the requirements
pip install -r requirements.txt
# Run the test
python train.py
```

If you want change the parameters, you can modify the `config.py` file.