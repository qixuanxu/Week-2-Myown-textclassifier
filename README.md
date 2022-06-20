# Week-8-Final Project
I Build a image classifier Classify and detect pictures
from a Concrete surface sample images for Surface Crack Detection dataset
And this is the link to the dataset:
https://www.kaggle.com/datasets/arunrk7/surface-crack-detection

This Notebook is built on the base of Coding-3-Week-2 CNNImageClassifier Notebook.
And on the process of building modle ,I referred to a tutorial notebook that uses the Dataframe method to load large amounts of data.
nd this is the link to the tutorial:
https://www.kaggle.com/code/gcdatkin/concrete-crack-image-detection/notebook

## Here is the code:
### Import Libraries
```
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report
```
### Import Dataset from my Google.Colab Drive
```
from google.colab import drive
drive.mount('/content/drive')
!ls "/content/drive/MyDrive/concretedataset"
positive_dir = Path('/content/drive/MyDrive/concretedataset/Positive')
negative_dir = Path('/content/drive/MyDrive/concretedataset/Negative')
```
### Cteate Images Dataframe
（refered to the tutorial notebook）
```
def generate_df(image_dir, label):
    filepaths = pd.Series(list(image_dir.glob(r'*.jpg')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=filepaths.index)
    df = pd.concat([filepaths, labels], axis=1)
    return df

positive_df = generate_df(positive_dir, label="POSITIVE")
negative_df = generate_df(negative_dir, label="NEGATIVE")

all_df = pd.concat([positive_df, negative_df], axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)
all_df

train_df, test_df = train_test_split(
    all_df.sample(3000, random_state=1),
    train_size=0.6,# 0.7
    shuffle=True,
    random_state=1
)
```
### Loading Images by Dataframe
（refered to the tutorial notebook）
```
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255 # RGB image data
) 

# Solve memory problems for training large data sets
```
```
train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_data = train_gen.flow_from_dataframe(
    test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=False,
    seed=42
)
```
### Create the neural network
(By changing the week-2 Notebook)
```
inputs = tf.keras.Input(shape=(120, 120, 3))
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```
### Compile and train the model
(By changing the week-2 Notebook)
```
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

```
```
history = model.fit(train_data,
                    epochs=10, 
                    validation_data=val_data,
                    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
                    )
```
#### RESULT:
Epoch 10/10
45/45 [==============================] - 4s 100ms/step - loss: 0.3933 - accuracy: 0.8889 - val_loss: 0.3782 - val_accuracy: 0.8917

### Plot how accuracy changes over time on the training set and the test ("validation") set
(By changing the week-2 Notebook)
```
plt.plot(history.history['accuracy'], label='training set accuracy')
plt.plot(history.history['val_accuracy'], label = 'test set accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

```
#### RESULT:
![下载](https://user-images.githubusercontent.com/91987208/174646414-dfcdd3dd-e21e-46c2-8b98-94a7fd68e3bb.png)

### Use code to test a trained model
(By changing the week-2 Notebook)
```
test_loss, test_acc = model.evaluate(train_data, verbose=2)
print("Test accuracy after final epoch is ", test_acc*100)

```

#### RESULT:
45/45 - 3s - loss: 0.3590 - accuracy: 0.9111 - 3s/epoch - 74ms/step
Test accuracy after final epoch is  91.11111164093018

### Evaluate modle
（refered to the tutorial notebook）

```
def evaluate_model(model, test_data):
    
    results = model.evaluate(test_data, verbose=0)
    loss = results[0]
    acc = results[1]
    
    print("    Test Loss: {:.5f}".format(loss))
    print("Test Accuracy: {:.2f}%".format(acc * 100))
    
    y_pred = np.squeeze((model.predict(test_data) >= 0.5).astype(np.int))
    cm = confusion_matrix(test_data.labels, y_pred)
    clr = classification_report(test_data.labels, y_pred, target_names=["NEGATIVE", "POSITIVE"])
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
    plt.xticks(ticks=np.arange(2) + 0.5, labels=["NEGATIVE", "POSITIVE"])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=["NEGATIVE", "POSITIVE"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    print("Classification Report:\n----------------------\n", clr)
    
```
    #### RESULT:
    Test Loss: 0.35784
    Test Accuracy: 90.67%
![下载 (1)](https://user-images.githubusercontent.com/91987208/174646781-80688203-f4eb-4eae-aaa9-2b583ca2be52.png)

```

Classification Report:
               precision    recall  f1-score   support

    NEGATIVE       0.90      0.92      0.91       592
    POSITIVE       0.92      0.90      0.91       608

    accuracy                           0.91      1200
   macro avg       0.91      0.91      0.91      1200
weighted avg       0.91      0.91      0.91      1200
```

