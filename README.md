# Pavement Temperature Prediction

The repository deals with the prediction of the pavement temperature in Canada. The repository not only carries the code that is being develop to cater the issue but also contain the results of each model along with their optimal weights. 

## Motivation
Canada is a country that experience high volume of snowfall hence, snow removal of roads holds a pivotal position to ensure saftey of the drivers. The removal is carried out via salt application on the roads and rate of application is dependent on the pavement temperature. The higher the temperature the greater would be the application. Therefore, optimizing this rate is imperative to have optimal resource allocation. The optimization can be carried out by predicting the pavement temperature. 

## Models
A total of 5 models are being developed to tackle the issue. Attention layers are also used for each of the model. The models deployed are the following:

* LSTM
* CNN-LSTM
* ConvLSTM 
* Seq2Seq
* WaveNet

# Tools Used
The following packages are used to carry out the task:

* Numpy
* Pandas
* TensorFlow
* Seaborn
* Matplotlib
* Scikit-Learn

## Results
The results of each combination can be seen in the Results folder in `Dashboard.xlsx`. The file is an interactive file which allows the user to navigate through various variables and getting the desired results. The screenshot below is an illustration of the file. 

![Dashboard](Images/Dashboard.PNG)

## Structure
The repoistory has the following directory tree.

```
Pavement-Temperature-Prediction
 ┣ Code
 ┃ ┣ Jupyter Notebook
 ┃ ┃ ┣ 01 - Exploratory Data Analysis.ipynb
 ┃ ┃ ┗ 02 - Forecasting.ipynb
 ┃ ┣ Python Scripts
 ┃ ┃ ┣ Combined_plot.py
 ┃ ┃ ┣ Ensembler.py
 ┃ ┃ ┣ main.py
 ┃ ┃ ┣ Models.py
 ┃ ┃ ┣ Train.py
 ┃ ┃ ┗ Utils.py
 ┣ Data
 ┃ ┣ Pave_data_1.xlsx
 ┃ ┗ Pave_data_cleaned.xlsx
 ┣ Results
 ┃ ┣ Six Hours Lag
 ┃ ┃ ┣ CNN-LSTM
 ┃ ┃ ┣ ConvLSTM
 ┃ ┃ ┣ LSTM
 ┃ ┃ ┣ Seq2Seq
 ┃ ┃ ┗ Wavenet
 ┃ ┣ Twenty Four Hours Lag
 ┃ ┃ ┣ CNN-LSTM
 ┃ ┃ ┣ ConvLSTM
 ┃ ┃ ┣ LSTM
 ┃ ┃ ┣ Seq2Seq
 ┃ ┃ ┗ Wavenet
 ┃ ┗ Dashboard.xlsx
```


```
Pavement-Temperature-Prediction
├─ .vscode
│  └─ settings.json
├─ Code
│  ├─ .ipynb_checkpoints
│  │  ├─ 01 - Exploratory Data Analysis-checkpoint.ipynb
│  │  ├─ 02 - Forecasting-checkpoint.ipynb
│  │  └─ Main-checkpoint.ipynb
│  ├─ Jupyter Notebook
│  │  ├─ 01 - Exploratory Data Analysis.ipynb
│  │  └─ 02 - Forecasting.ipynb
│  ├─ Results
│  ├─ Source
│  │  ├─ .vscode
│  │  │  └─ settings.json
│  │  ├─ forecasting.ipynb
│  │  ├─ main.py
│  │  ├─ model.py
│  │  ├─ utils.py
│  │  └─ __pycache__
│  │     ├─ model.cpython-38.pyc
│  │     └─ utils.cpython-38.pyc
│  ├─ Testing
│  │  ├─ Combined_plots.ipynb
│  │  ├─ Main.ipynb
│  │  └─ Python Scripts
│  │     ├─ Combined_plot.py
│  │     ├─ Ensembler.py
│  │     ├─ main.py
│  │     ├─ Models.py
│  │     ├─ Train.py
│  │     └─ Utils.py
│  ├─ Weights
│  └─ __pycache__
│     ├─ Ensembler.cpython-37.pyc
│     ├─ Models.cpython-37.pyc
│     ├─ Train.cpython-37.pyc
│     └─ Utils.cpython-37.pyc
├─ Data
│  ├─ Pave_data_1.xlsx
│  └─ Pave_data_cleaned.xlsx
├─ Images
│  └─ Dashboard.PNG
├─ README.md
├─ requirements.txt
└─ Results
   ├─ Dashboard.xlsx
   ├─ Six Hours Lag
   │  ├─ CNN-LSTM
   │  │  ├─ attention_cnnlstm.hdf5
   │  │  ├─ CNN-LStM Results.xls
   │  │  └─ simple_cnnlstm.hdf5
   │  ├─ ConvLSTM
   │  │  ├─ attention_convlstm.hdf5
   │  │  ├─ ConvLSTM Results.xls
   │  │  └─ simple_convlstm.hdf5
   │  ├─ LSTM
   │  │  ├─ attention_lstm.hdf5
   │  │  ├─ LSTM Result.xls
   │  │  ├─ simple_lstm.hdf5
   │  │  └─ simple_lstm_l2.hdf5
   │  ├─ LSTM1
   │  │  ├─ attention_lstm_bn.hdf5
   │  │  ├─ attention_lstm_l2.hdf5
   │  │  ├─ result.xls
   │  │  ├─ result.xlsx
   │  │  └─ simple_lstm.hdf5
   │  ├─ Seq2Seq
   │  │  ├─ attention_seq2seq.hdf5
   │  │  ├─ Seq2Seq Results.xls
   │  │  └─ simple_seq2seq.hdf5
   │  └─ Wavenet
   │     ├─ attention_wavenet.hdf5
   │     ├─ simple_wavenet.hdf5
   │     └─ Wavenet Results.xls
   └─ Twenty Four Hours Lag
      ├─ CNN-LSTM
      │  ├─ attention_cnnlstm.hdf5
      │  ├─ CNN-LStM Results.xls
      │  └─ simple_cnnlstm.hdf5
      ├─ ConvLSTM
      │  ├─ attention_convlstm.hdf5
      │  ├─ ConvLSTM Results.xls
      │  └─ simple_convlstm.hdf5
      ├─ LSTM
      │  ├─ attention_lstm.hdf5
      │  ├─ LSTM Result.xls
      │  └─ simple_lstm.hdf5
      ├─ Seq2Seq
      │  ├─ attention_seq2seq.hdf5
      │  ├─ Seq2Seq Results.xls
      │  ├─ simple_seq2seq.hdf5
      │  └─ simple_seq2seq1.hdf5
      └─ Wavenet
         ├─ attention_wavenet.hdf5
         ├─ simple_wavenet.hdf5
         └─ Wavenet Results.xls

```