import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
#Model was saved, so no need to rebuild RandomForestRegressor
#from sklearn.ensemble import RandomForestRegressor
import pickle

st.write("""
# Boston Houses Price Prediction App
This app predicts the **Boston Housing Prices** and shows the significance of each variable!
""")
st.write('---')

# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = st.sidebar.slider('CRIM', float(X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('ZN', float(X.ZN.min()), float(X.ZN.max()), float(X.ZN.mean()))
    INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()), float(X.INDUS.max()), float(X.INDUS.mean()))
    CHAS = st.sidebar.slider('CHAS', float(X.CHAS.min()), float(X.CHAS.max()), float(X.CHAS.mean()))
    NOX = st.sidebar.slider('NOX', float(X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
    RM = st.sidebar.slider('RM', float(X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
    AGE = st.sidebar.slider('AGE', float(X.AGE.min()), float(X.AGE.max()), float(X.AGE.mean()))
    DIS = st.sidebar.slider('DIS', float(X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
    RAD = st.sidebar.slider('RAD', float(X.RAD.min()), float(X.RAD.max()), float(X.RAD.mean()))
    TAX = st.sidebar.slider('TAX', float(X.TAX.min()), float(X.TAX.max()), float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider('PTRATIO', float(X.PTRATIO.min()), float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
    B = st.sidebar.slider('B', float(X.B.min()), float(X.B.max()), float(X.B.mean()))
    LSTAT = st.sidebar.slider('LSTAT', float(X.LSTAT.min()), float(X.LSTAT.max()), float(X.LSTAT.mean()))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
#model = RandomForestRegressor()
#model.fit(X, Y)
# Apply Model to Make Prediction
# The model was saved in a pickle file order to make it load faster.
filename = 'weight_pred_model'
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.predict(df)

st.header('Prediction of MEDV (Median value of owner-occupied homes in $1000’s)')
st.write(loaded_model.predict(df))
st.write('---')

if st.button('Attribute Information'):
    st.write("""
    | Attribute      | Description |
    | :---        |    :----:   |
    | CRIM | per capita crime rate by town |
    | ZN | proportion of residential land zoned for lots over 25,000 sq.ft. |
    | INDUS | proportion of non-retail business acres per town       |
    | CHAS | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) |
    | NOX | nitric oxides concentration (parts per 10 million) |
    | RM | average number of rooms per dwelling |
    | AGE | proportion of owner-occupied units built prior to 1940 |
    | DIS | weighted distances to five Boston employment centres |
    | RAD | index of accessibility to radial highways |
    | TAX | full-value property-tax rate per $10,000 |
    | PTRATIO | pupil-teacher ratio by town |
    | B | is the proportion of black people by town |
    | LSTAT | % lower status of the population |
    | MEDV | Median value of owner-occupied homes in $1000’s |
    
    """)
    st.write('---')


# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(loaded_model)
shap_values = explainer.shap_values(X)

# In order to hard fix streamlit update bug
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
## Feature Importance
* This next bar table does tell you how ***strong*** is each parameter contributing.
* This next bar table doesn't tell you ***how*** is each parameter contributing.
* As in what aspect, is it a **Positive** or **Negative** impact.
""")
plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')

st.write("""
## Feature Importance
* This next table does show you the **Positive** or **Negative** impact.
""")
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')