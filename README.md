# Warning: the implementation is unfinished!! 
### PARTS model and segmentation training in pytorch
(On-going) My implementation of [PARTS: Unsupervised segmentation with slots, attention and independence maximization](https://openaccess.thecvf.com/content/ICCV2021/papers/Zoran_PARTS_Unsupervised_Segmentation_With_Slots_Attention_and_Independence_Maximization_ICCV_2021_paper.pdf).
<br/><br/>
I'm still debugging the model (help is welcomed of course). \
The model is currently significantly downscaled (model with the original setting is very large) and trained on the Sprites-MOT dataset from [here](https://github.com/ecker-lab/object-centric-representation-benchmark) which is an easier dataset than Clevrer.

Training can be unstable and inconsistent: sometimes representations will be disentangled, sometimes not; sometimes colors and shapes can be learned, sometimes not; sometimes the model will suddenly collapse to nonsense outputs. It is also very sensitive to beta. Below is trained from beta=0.3. 

<div>Ground truth<br/><img width="150" alt="" src="/images/gt1.gif"></div>  <nobr/> <div>Reconstruction<br/><img width="150" alt="" src="/images/recon1.gif"></div><br/><br/>


<img width="150" alt="" src="/images/0_1.gif"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img width="150" alt="" src="/images/1_1.gif"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img width="150" alt="" src="/images/2_1.gif"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img width="150" alt="" src="/images/3_1.gif">
