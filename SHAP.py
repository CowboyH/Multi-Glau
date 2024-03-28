import matplotlib.pyplot as plt
import shap
import joblib
import pandas as pd

# load model
xgb_model = joblib.load('')
# load dataset
plot_data = pd.read_excel('')


# shap
fig ,ax5 = plt.subplots()
explainer=shap.TreeExplainer(xgb_model)
shap_values= explainer.shap_values(plot_data.iloc[:,:-1]) 
feature_name=['Gender','Age','BCVA','CDR','IOP']
shap.summary_plot(shap_values,plot_data.iloc[:,:-1],feature_names=feature_name,plot_type="bar",show=False)
plt.savefig('',dpi=1000,bbox_inches='tight')
plt.show()


# 

fig ,ax6 = plt.subplots()
shap.summary_plot(shap_values, plot_data.iloc[:,:-1],feature_names=feature_name,show=False)
plt.savefig('',dpi=1000,bbox_inches='tight')
plt.show()