import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import Parser
import os

Parser.ParseCar1()
input_file = 'CarParsed.csv'
data = pd.read_csv(input_file)

X = data.drop(columns=['Class'])
y = data['Class']

### TASK 1 ###
output_file = 'kvalue.csv'
# ITERATE SCALARA TYPES AND K VAULES
import os
if os.path.exists(output_file):
    os.remove(output_file)

for i in range(2):
    if i == 0:
        scaler = MinMaxScaler()
        scaler_type = 'MinMaxScaler'
    else:
        scaler = StandardScaler()
        scaler_type = 'StandardScaler'

    for kval in range(1, 100):  # Increment K value by 1
        X_scaled = scaler.fit_transform(X)
        X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(X_scaled, y, test_size=0.2)
        knntestval = KNeighborsClassifier(n_neighbors=kval)  # K must be an integer
        knntestval.fit(X_train_set, y_train_set)
        y_pred_set = knntestval.predict(X_test_set)
        accuracy = accuracy_score(y_test_set, y_pred_set)

        output_data = pd.DataFrame({
            'K Value': [kval],
            'Accuracy': [accuracy * 100],  # Convert to percentage
            'Scaler': [scaler_type]
        })

        file_exists = os.path.exists(output_file)
        output_data.to_csv(output_file, index=False, mode='a', header=not file_exists)


### TASK 2 ###
# READ THE K VALUES DATA CSV
data = pd.read_csv(output_file)

minmax_data = data[data['Scaler'] == 'MinMaxScaler']
standard_data = data[data['Scaler'] == 'StandardScaler']
plt.figure(figsize=(10, 6))
plt.plot(minmax_data['K Value'], minmax_data['Accuracy'], linestyle='-', label='MinMaxScaler')
plt.plot(standard_data['K Value'], standard_data['Accuracy'], linestyle='-', label='StandardScaler')
plt.title('K Value vs. Accuracy by Scaler Type', fontsize=16)
plt.xlabel('K Value', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

### TASK 3 ###
# USE K VALUE 10 WITH MINMAXSCALAR
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
kval = 10
final_model = KNeighborsClassifier(n_neighbors=kval)

final_model.fit(X_train_set, y_train_set)
print(f"\n Kvalue {kval}, {scaler}")

#predictions
y_train_pred = final_model.predict(X_train_set)
y_test_pred = final_model.predict(X_test_set)

#confusion
conf_matrix = confusion_matrix(y_test_set, y_test_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

#accuracy
train_accuracy = accuracy_score(y_train_set, y_train_pred) * 100
test_accuracy = accuracy_score(y_test_set, y_test_pred) * 100

print(f"\nModel accuracy in the training set = {train_accuracy:.2f}%")
print(f"Model accuracy in the test set = {test_accuracy:.2f}%")


### TASK 4 ###
new_instance = [[65.0, 50.0, 1200, 0.75, 3.0, 0.25, 3.5]]
new_instance_scaled = scaler.transform(new_instance)
predicted_class = final_model.predict(new_instance_scaled)

print(f"\nPredicted Class Data: {new_instance}")
print(f"\nPredicted Class for the New Instance: {predicted_class[0]}")
