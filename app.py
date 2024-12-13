from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as mat
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc

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
    
    # Save the confusion matrix figure
    confusion_matrix_path = os.path.join('static', 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close(fig)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_report = classification_report(y_test, y_pred)

    # Calculate ROC curve and AUC
    y_prob = model.predict_proba(X_test)  # Get predicted probabilities
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1], pos_label=1)  # Assuming binary classification for the second class
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    fig_roc, ax_roc = plt.subplots(figsize=(10, 7))
    ax_roc.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
    ax_roc.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax_roc.legend(loc='lower right')

    # Save the ROC curve figure
    roc_curve_path = os.path.join('static', 'roc_curve.png')
    plt.savefig(roc_curve_path)
    plt.close(fig_roc)

    return render_template('train.html', accuracy=accuracy, confusion_matrix_path=confusion_matrix_path, 
                           accuracy_report=accuracy_report, roc_curve_path=roc_curve_path, roc_auc=roc_auc)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Get user input from the form
        input_data = {
            'RI': float(request.form['RI']),
            'Na': float(request.form['Na']),
            'Mg': float(request.form['Mg']),
            'Al': float(request.form['Al']),
            'Si': float(request.form['Si']),
            'K': float(request.form['K']),
            'Ca': float(request.form['Ca']),
            'Ba': float(request.form['Ba']),
            'Fe': float(request.form['Fe'])
        }
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model2.predict(input_df)[0]
    
    return render_template('predict.html', prediction=prediction, glass_types=glass_types)

# if __name__ == '__main__':
#    app.run(debug=True)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

