import pandas as pd  
 
penguin_df = pd.read_csv('penguins.csv') 
penguin_df.dropna(inplace=True) 
output = penguin_df['species'] 
features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']] 
features = pd.get_dummies(features) 
output, uniques = pd.factorize(output)
print('Here is what our unique output variables represent') 
print(uniques)
print('Here are our feature variables') 
print(features.head())

from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=.8) 
rfc = RandomForestClassifier(random_state=15) 
rfc.fit(x_train, y_train) 
y_pred = rfc.predict(x_test) 
score = round(accuracy_score(y_pred, y_test), 2)
print(score)