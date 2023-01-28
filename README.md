# Predict_Chung_Khoan
Dựa vào giá trị của trứng khoán của 30 ngày trước để dự đoán giá trị trứng khoán của 5 ngày sau đó.
Build bằng machine learning và deep learning.
Tạo python web cơ bản để visualize nhũng thứ đã làm được

# Intallation
install package from requirements.txt
```bash
pip install -r requirements.txt
# install vnquant package
git clone https://github.com/phamdinhkhanh/vnquant
cd vnquant
python setup.py install
```

# Train model

run file [model.ipynb](model.ipynb) to download data and train model
trained model will be saved to model.pth

# Web
user can upload data 30 day to predict 5 next day from csv file

1. user can upload data in task `import data`


upload data: ![image](https://github.com/nhanvu10092001/Predict_Chung_Khoan/blob/main/img/1.png)

2. user can get predict result from file uploaded in `predict`


predict data: ![image](https://github.com/nhanvu10092001/Predict_Chung_Khoan/blob/main/img/2.png)