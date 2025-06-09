# Multi-Glau
This is the source code for the article 'A three-tier AI solution for equitable glaucoma diagnosis across China’s hierarchical healthcare system '.<br>
![](readme/workflow.png)
## File description
* SHAP.py : Visualizing the impact of features on the model, specifically visualizing the relative importance of features. Here is the link to the SHAP library: [SHAP Library](https://shap.readthedocs.io/en/latest/index.html#).
* decision.py : The decision curve is a graphical representation that allows for the assessment of clinical strategies by evaluating their net benefit across different Pthresholds. The x-axis represents the range of possible Pthresholds, while the y-axis indicates the net benefit.

## Contents

- **SHAP.py**  
  Visualizes the impact of features on the model, specifically highlighting the relative importance of each feature.  
  Reference: [SHAP Library](https://shap.readthedocs.io/en/latest/index.html#)

- **decision.py**  
  Implements the decision curve analysis, a graphical method used to evaluate clinical strategies by assessing net benefit across different probability thresholds (*P<sub>threshold</sub>*).  
  The x-axis represents the range of possible thresholds, while the y-axis indicates the corresponding net benefit.

> ⚙️ Code for the screening model is currently being organized and annotated. It will be uploaded by **June 15, 2025 (Beijing Time)**.


## Data availability
Due to healthcare data management policies, the data used in this study cannot be made publicly accessible. However, the Multi-Glau discussed in this paper are generic and can be used as long as the input consists of medical images and structured numerical data.



