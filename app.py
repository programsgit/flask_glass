from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as mat
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

app = Flask(__name__)

# Load the dataset
url = "glass.data.csv"
column_names = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
df = pd.read_csv(url, header=None, names=column_names)
df = df.drop(columns='Id')

# Train the model
X = df.drop(columns='Type')
y = df['Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
pickle.dump(model, open('glass.pkl', 'wb'))

@app.route('/')
def index():
    return render_template('index.html', tables=[df.head().to_html(classes='data')], titles=df.columns.values)

@app.route('/eda')
def eda():
    # Generate plots for EDA
    return render_template('eda.html')

@app.route('/train')
def train():
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=df['Type'].unique())
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=df['Type'].unique(), yticklabels=df['Type'].unique(), ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    # Save the figure
    confusion_matrix_path = os.path.join('static', 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close(fig)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_report = classification_report(y_test, y_pred)

    return render_template('train.html', accuracy=accuracy, confusion_matrix_path=confusion_matrix_path, accuracy_report=accuracy_report)

if __name__ == '__main__':
    app.run(debug=True)
