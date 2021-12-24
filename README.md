# Warning: the implementation is unfinished!! 
### PARTS model and segmentation training on Clevrer dataset in pytorch
(On-going) My implementation of [PARTS: Unsupervised segmentation with slots, attention and independence maximization](https://openaccess.thecvf.com/content/ICCV2021/papers/Zoran_PARTS_Unsupervised_Segmentation_With_Slots_Attention_and_Independence_Maximization_ICCV_2021_paper.pdf).
<br/><br/>
I'm still debugging the model (help is welcomed of course). \
The model is currently largely downscaled and tested on the Sprites-MOT dataset from [here](https://github.com/ecker-lab/object-centric-representation-benchmark)
Training is very unstable: sometimes representations will be disentangled, sometimes not; sometimes colors and shapes can be learned, sometimes not; sometimes the model will suddenly collapse to nonsense outputs. It is also very sensitive to beta.

<div>Ground truth<br/><img width="150" alt="" src="/images/gt1.gif"></div> <div>Reconstruction<br/><img width="150" alt="" src="/images/recon1.gif"></div><br/><br/>


<img width="150" alt="" src="/images/0_1.gif"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img width="150" alt="" src="/images/1_1.gif"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img width="150" alt="" src="/images/2_1.gif"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img width="150" alt="" src="/images/3_1.gif">
