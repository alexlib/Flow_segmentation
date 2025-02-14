# Flow_segmentation

This is the official repository for "Segment Anything in Flow Experiments".

## News

- 2024.04.15: First release 
<a href="https://colab.research.google.com/github/AliRKhojasteh/Flow_segmentation/blob/main/Notebooks/Flow_segmentation.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" width="150"/>
</a>
<img src="Flow_segmentation.png" width="800">

## Installation

1. Create a virtual environment: `conda create -n flowsam python=3.10 -y` and activate it: `conda activate flowsam`.
2. Clone the repository: `git clone https://github.com/AliRKhojasteh/Flow_segmentation`.
3. Enter the Flow_segmentation folder: `cd Notebooks`.

## Get Started

Follow the instructions inside the Notebook folder. Open the `Flow_segmentation.ipynb`. Within the notebook, you will automatically install and clone required packages, input your image and the text prompt. The model checkpoints are downloaded and available.

1. Click on "Open in Colab" if you would like to **run on your browser** without the need for further installations.  
2. Read dependencies and install them
3. Loading the Input Image "Fingers.png"
4. text_prompt = 'Fingers and a hand'
5. Compute masks

<img src="Fingers_masks.png" width="800">
*2D PIV experiment of a moving hand, textual input = fingers + hand*


## Sample Usage Permissions 
All examples available in the 'demo' directory are permitted for demonstration purposes only. For additional usage permissions, please contact the corresponding authors listed in the references. 

## References
This project makes use of the following repositories:

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Lang Segment Anything](https://github.com/luca-medeiros/lang-segment-anything)
- [Lightning SAM](https://github.com/luca-medeiros/lightning-sam)
- [Supervision](https://github.com/roboflow/supervision)


## Citation

```bibtex
@article{khojasteh2024practical,
  title={Practical Object and Flow Structure Segmentation using Artificial Intelligence},
  author={Khojasteh, Ali Rahimi and van de Water, Willem and Westerweel, Jerry},
  note={Submitted to Experiments in Fluids},
  year={2024}
}


