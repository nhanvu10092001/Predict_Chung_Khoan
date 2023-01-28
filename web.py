import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datetime import date
import streamlit.components.v1 as components
import get_data
import sklearn.externals
import joblib
from sklearn.preprocessing import MinMaxScaler
import os







add_selectbox = st.sidebar.selectbox(
    ("Option"),
    ("Import Data", "Visualize", "Predict")
)

if add_selectbox == 'Import Data':
    
    st.title("Do u wanna import data to predict them? ")
    file = st.file_uploader("upload file (*.CSV)")

    if file  is not None:
        st.write(file, "\n")
        data = pd.read_csv(file).drop('Unnamed: 0', axis = 1)
        st.write(data.head())
        data.to_csv('user_data.csv')
    else:
        st.warning("Data Is NULL")
    st.title('data today \n')
    st.write(get_data.get_data())
elif add_selectbox ==   'Visualize':
    st.header('Visualize of data train model')

    HtmlFile = open("Visual.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height = 1500, width = 1500)
else :
    class Regression(nn.Module):
        def __init__(self, input_feature):
            super(Regression, self).__init__()

            self.layer_0 = nn.Linear(input_feature, 100)
            self.layer_1 = nn.Linear(100, 80)
            self.layer_2 = nn.Linear(80, 60)
            self.layer_3 = nn.Linear(60, 20)
            self.layer_4 = nn.Linear(20, 5)
        def forward(self, x):
            
            output = self.layer_0(x)
            output = self.layer_1(output)
            output = self.layer_2(output)
            output = self.layer_3(output)
            output = self.layer_4(output)
            return output


    class FPTDataset(Dataset):
        def __init__(self, array, label):
            super(FPTDataset, self).__init__()

            self.data = array.astype(np.float32)
            self.label = label.astype(np.float32)
        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, index):
            return self.data[index, :], self.label[index, :]
            
        

    model = Regression(180)
    model.load_state_dict(torch.load('model.pth'))
        
    st.text(model.eval())

    
    if os.path.exists('user_data.csv') :
        data = pd.read_csv('user_data.csv').drop('Unnamed: 0', axis = 1)
        
        data = torch.tensor(data, dtype = torch.float)
        result = model(data)
        result = result.cpu().detach().numpy().flatten()
        dict_day = {
        "Day 1": result[0],
        "Day 2": result[1],
        "Day 3": result[2],
        "Day 4": result[3],
        "Day 5": result[4]

        }
        st.text('Predict result of user: ')
        st.write(dict_day)
        os.remove('user_data.csv')

    data_today  = get_data.get_data()
    data_today = torch.tensor(data_today.to_numpy(), dtype = torch.float)
    result_today  = model(data_today)
    st.text('Predict for next 5 day: \n')
    result_today = result_today.cpu().detach().numpy().flatten()
    dict_day = {
        "Day 1": result_today[0],
        "Day 2": result_today[1],
        "Day 3": result_today[2],
        "Day 4": result_today[3],
        "Day 5": result_today[4]

    }
    
    st.write(dict_day)