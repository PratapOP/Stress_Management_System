import matplotlib.pyplot as plt

# Models
models = ['Logistic Regression', 'Random Forest', 'SVM', 'Decision Tree', 'KNN']

# Metrics
accuracy = [0.882, 0.891, 0.877, 0.855, 0.850]
precision = [0.882, 0.893, 0.877, 0.855, 0.852]
recall = [0.882, 0.891, 0.877, 0.855, 0.850]
f1_score = [0.882, 0.891, 0.877, 0.855, 0.850]

# Create plot
plt.figure()

plt.plot(models, accuracy, marker='o', label='Accuracy')
plt.plot(models, precision, marker='o', label='Precision')
plt.plot(models, recall, marker='o', label='Recall')
plt.plot(models, f1_score, marker='o', label='F1 Score')

# Labels and title
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Performance Comparison (Line Graph)')

plt.xticks(rotation=20)
plt.legend()

plt.tight_layout()

# Save image
plt.savefig('model_comparison_line.png', dpi=300)

plt.show()