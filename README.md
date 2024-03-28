# Multi-Glau
This is the source code for the article 'A multitask system based on machine learning to advance health equity of glaucoma healthcare in China'.
## File description
* SHAP.py : Visualizing the impact of features on the model, specifically visualizing the relative importance of features. Here is the link to the SHAP library: [SHAP Library](https://shap.readthedocs.io/en/latest/index.html#).
* decision.py : The decision curve is a graphical representation that allows for the assessment of clinical strategies by evaluating their net benefit across different Pthresholds. The x-axis represents the range of possible Pthresholds, while the y-axis indicates the net benefit. 
## Data availability
Due to healthcare data management policies, the data used in this study cannot be made publicly accessible. However, the Multi-Glau discussed in this paper are generic and can be used as long as the input consists of medical images and structured numerical data.
## Deployment of Multi-Glau on the web platform
We have deployed Multi-Glau using Gradio on Hugging Face. You can click the link below to try using our developed Multi-Glau.  

[Multi-Glau](https://huggingface.co/spaces/Aohanah/Window)


