import utils as u
import model as m


# Specifying the source path
src = r'D:\Pavement-Temperature-Prediction\Data\Pave_data_cleaned.xlsx'

# Reading the file
df = u.read_file(src)

# Creating the training and testing data
x_train, x_test, y_train, y_test, scaler = u.splitter(
    df[['Temp', 'Pavement']], ['Pavement'], 6, 6, 0.8)
print(f'The shape of x_train is {x_train.shape} and x_test is {x_test.shape}')
print(f'The shape of y_train is {y_train.shape} and y_test is {y_test.shape}')

functs = [m.buildLSTM,m.buildCNNLSTM,m.buildConvLSTM,m.buildSeq2Seq,m.buildWavenet]

for f in functs:
    for attention in [True,False]:
        mode = f(x_train,y_train,scaler,atten=attention)
        mode.fit()
        mode.predictions(x_test,y_test)
