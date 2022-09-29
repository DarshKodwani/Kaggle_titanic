import os
import pandas as pd
import tensorflow as tf
import numpy as np

inputs_folder = os.path.join(os.environ["REPO_ROOT"], "input_data")
train_raw = pd.read_csv(os.path.join(inputs_folder, "train.csv"))
test_raw = pd.read_csv(os.path.join(inputs_folder, "test.csv"))
features = ['SibSp', 'Pclass', 'Sex', 'Parch']

y_train = train_raw['Survived']
X_train = train_raw[features]
X_test = test_raw[features]

X_train_one_hot = pd.get_dummies(X_train)
X_test_one_hot = pd.get_dummies(X_test)

model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics='accuracy')
model.fit(x=X_train_one_hot, y=y_train, epochs=200)

y_train_preds = np.round(model.predict(X_train_one_hot)).squeeze()
y_test_preds = np.round(model.predict(X_test_one_hot)).squeeze()