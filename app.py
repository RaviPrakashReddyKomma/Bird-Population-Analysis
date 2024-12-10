import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template, jsonify, send_from_directory
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='shap')

# Initialize Flask app
app = Flask(__name__)

# Load dataset
bird_data = pd.read_csv(r"bird_population_climate_data.csv")
label_encoders = {}
for column in ['species', 'population_trend']:
    label_encoders[column] = LabelEncoder()
    bird_data[column] = label_encoders[column].fit_transform(bird_data[column])

X = bird_data.drop(columns=['population_trend', 'year'])
y = bird_data['population_trend']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoders['population_trend'].classes_)

# Feature importance plot
feature_importances = rf_model.feature_importances_
features = X.columns

# Create folder for storing images if it doesn't exist
if not os.path.exists('static/images'):
    os.makedirs('static/images')

# Save feature importance plot
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances)
plt.title('Feature Importance: Impact of Climate Change on Birds')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.savefig('static/images/feature_importance.png')
plt.close()

# Generate scatter plots for temperature, precipitation, longitude, latitude vs population trend
def save_scatter_plot():
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 2, 1)
    plt.scatter(bird_data['temperature'], bird_data['population_trend'], alpha=0.7, c='blue')
    plt.title('Temperature vs Bird Population Trend')
    plt.xlabel('Temperature')
    plt.ylabel('Population Trend')

    plt.subplot(3, 2, 2)
    plt.scatter(bird_data['precipitation'], bird_data['population_trend'], alpha=0.7, c='green')
    plt.title('Precipitation vs Bird Population Trend')
    plt.xlabel('Precipitation')
    plt.ylabel('Population Trend')

    plt.subplot(3, 2, 3)
    plt.scatter(bird_data['longitude'], bird_data['population_trend'], alpha=0.7, c='orange')
    plt.title('Longitude vs Bird Population Trend')
    plt.xlabel('Longitude')
    plt.ylabel('Population Trend')

    plt.subplot(3, 2, 4)
    plt.scatter(bird_data['latitude'], bird_data['population_trend'], alpha=0.7, c='red')
    plt.title('Latitude vs Bird Population Trend')
    plt.xlabel('Latitude')
    plt.ylabel('Population Trend')

    plt.subplot(3, 2, 5)
    plt.scatter(bird_data['year'], bird_data['population_trend'], alpha=0.7, c='purple')
    plt.title('Year vs Bird Population Trend')
    plt.xlabel('Year')
    plt.ylabel('Population Trend')

    plt.tight_layout()
    plt.savefig('static/images/scatter_plots.png')
    plt.close()

save_scatter_plot()

# SHAP plots
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train)

# Save SHAP summary plot
shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
plt.savefig('static/images/shap_summary_bar.png')
plt.close()

shap.summary_plot(shap_values, X_train, show=False)
plt.savefig('static/images/shap_summary.png')
plt.close()

@app.route('/')
def index():
    return render_template('index.html', accuracy=accuracy, classification_rep=classification_rep)

@app.route('/next/<int:graph_id>')
def next_graph(graph_id):
    graphs = [
        "feature_importance.png",
        "scatter_plots.png",
        "shap_summary_bar.png",
        "shap_summary.png"
    ]
    # Ensure we have 5 graphs, extend the list to include additional graphs if necessary.
    if graph_id < len(graphs):
        return send_from_directory('static/images', graphs[graph_id])
    else:
        return "No more graphs", 404

@app.route('/data')
def data():
    return jsonify({
        "accuracy": accuracy,
        "classification_report": classification_rep
    })

if __name__ == '__main__':
    app.run(debug=True)
