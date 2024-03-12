import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

data=pd.read_csv("D:\\mobile_price\\cleaned_mobile_data.csv")
data.head()
features=data.drop(columns=['price_range'])
targets=data[['price_range']]

scaler=StandardScaler()
scaled_data=scaler.fit_transform(features)
x_train,x_test,y_train,y_test=train_test_split(scaled_data,targets,test_size=0.2,random_state=42)

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(x_train,y_train)
ypred=model.predict(x_test)
print(classification_report(y_test,ypred))

print(confusion_matrix(y_test,ypred))
pickle.dump({'model': model, 'scaler': scaler}, open('mp.pkl', 'wb'))
model=pickle.load(open('mp.pkl','rb'))
