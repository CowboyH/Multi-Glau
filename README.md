# Multi-Glau
This is the source code for the article 'An AI system with multiple functions to advance health equity of glaucoma healthcare in China '.<br>
<div align=center>
	<img src=readme/workflow.png>
</div> <bar>
## File description
* SHAP.py : Visualizing the impact of features on the model, specifically visualizing the relative importance of features. Here is the link to the SHAP library: [SHAP Library](https://shap.readthedocs.io/en/latest/index.html#).
* decision.py : The decision curve is a graphical representation that allows for the assessment of clinical strategies by evaluating their net benefit across different Pthresholds. The x-axis represents the range of possible Pthresholds, while the y-axis indicates the net benefit.
* plot.py: It encompasses the code for generating violin plots, pie charts, and kernel density estimation diagrams. These codes can be utilized for exploratory data analysis. Figure 2 in the manuscript is based on this code and is created by loading the data for visualization. Figure 2 is displayed below.
* **The code related to the model files will be published after the paper is accepted. However, you can click the following link to use [Multi-Glau](http://multi-glau.online/).**
## Data availability
Due to healthcare data management policies, the data used in this study cannot be made publicly accessible. However, the Multi-Glau discussed in this paper are generic and can be used as long as the input consists of medical images and structured numerical data.



