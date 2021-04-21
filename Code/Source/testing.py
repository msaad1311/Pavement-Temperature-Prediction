import utils as u
import model as m

print('Libraries Loaded')

## Specifying the source path
src = r'D:\Pavement-Temperature-Prediction\Data\Pave_data_cleaned.xlsx'

## Reading the file
df = u.read_file(src)

## Creating the training and testing data
x_train,x_test,y_train,y_test,scaler = u.splitter(df[['Temp','Pavement']],['Pavement'],6,6,0.8)
print(f'The shape of x_train is {x_train.shape} and x_test is {x_test.shape}')
print(f'The shape of y_train is {y_train.shape} and y_test is {y_test.shape}')


model1 = m.buildCNNLSTM(x_train,y_train,scaler,True)
model1.fit()
model1.predictions(x_test,y_test)