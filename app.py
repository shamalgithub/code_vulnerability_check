import streamlit as st
from pygments import highlight
from pygments.lexers import CLexer
from pygments.formatters import HtmlFormatter
import joblib 
import pandas as pd 
import numpy as np
from model import model


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.title("Buffer Overflow Detection")
uploaded_file = st.file_uploader("Upload a .c file", type=["c"] )

@st.cache_resource
def load_model():
    model = joblib.load("random_forrest_model.joblib")
    return model


def prediction():
    line_list =[]
    if uploaded_file is not None:
        
        for lines in uploaded_file:
            line_list.append(lines.decode('utf-8'))
        
        line_number = [i+1 for i in range(len(line_list))]
        
        df = pd.DataFrame({"code": line_list ,"line_number":line_number})
        df['code'] = df['code'].str.strip()
        df['code'].replace('', np.nan, inplace=True)
        df.dropna(inplace=True)
        
        model = load_model()
        
        
        mask = model.predict(df['code'])
        mask_prob = model.predict_proba(df['code'])
        
        
        indexes_of_ones = [i for i, x in enumerate(mask) if x == 1]
        probabiliy_list = [round((mask_prob[i][1]) , 3) for i   in indexes_of_ones]
        mask = mask.astype(bool)
        df_filtered = df[mask]
        df_filtered.loc[:, 'probability'] = probabiliy_list
        df_filtered['probability'] = df_filtered['probability'].apply(lambda x: '{:.3f}%'.format(x * 100))
        return df_filtered
    
prediction_bttn = st.button("Run Prediction")

if prediction_bttn:
    result = prediction()
    result = result[['line_number' , 'code' ,'probability']]
    st.code(result.to_string(index=False, justify='right',header=False) , language="c")
    

