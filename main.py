import cv2
import os
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
from collections import Counter

model = tf.keras.applications.MobileNetV2(weights='imagenet')
results = []
PATH = '/Users/loulo/OneDrive/Objectif_Journalier'

def preprocess_image(image_path, size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, size)
    image = image / 255.0
    return image

def get_dominant_colors(image, k=5):
    image = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)
    colors = kmeans.cluster_centers_
    return colors

def plot_colors(colors):
    fig, ax = plt.subplots(1, figsize=(12, 2),
                           subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
    ax.imshow([colors], aspect='auto')
    plt.show()

def classify_image(image):
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    predictions = model.predict(np.expand_dims(image, axis=0))
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)
    preds = [decoded_predictions[0][0][1], decoded_predictions[0][1][1], decoded_predictions[0][2][1]]
    return preds

def images_to_dataframe():
    folder_paths = os.listdir(PATH)
    for folder_path in folder_paths:
        path = PATH+'/'+folder_path
        list_images = os.listdir(path)
        for image_path in list_images:
            img_path = PATH+'/'+folder_path+'/'+image_path
            image = preprocess_image(img_path)
            colors = get_dominant_colors(image)
            predictions = classify_image(image)

            result = {
                'image_path': image_path,
                'colors': colors,
                'predictions': predictions
            }
            results.append(result)

    df = pd.DataFrame(results)
    df.to_csv('analysis_results.csv', index=False)
    print("Analysis completed and saved to analysis_results.csv")

def csv_to_dataframe(csv_path):
    return pd.read_csv(csv_path)

def main():
    #images_to_dataframe()

    df = csv_to_dataframe('analysis_results.csv')
    # Aplatir les prédictions et compter les occurrences
    object_counts = df['predictions'].explode().value_counts()

    # Convertir en DataFrame pour Seaborn
    df = object_counts.reset_index()
    df.columns = ['Object', 'Count']

    sns.barplot(x='Object', y='Count', data=df)
    plt.xticks(rotation=90)
    plt.show()

if __name__ == "__main__":
    main()