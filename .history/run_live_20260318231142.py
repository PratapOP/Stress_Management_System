import matplotlib.pyplot as plt
import numpy as np

# Models
models = ['Logistic Regression', 'Random Forest', 'SVM', 'Decision Tree', 'KNN']

# Metrics
accuracy = [0.882, 0.891, 0.877, 0.855, 0.850]
precision = [0.882, 0.893, 0.877, 0.855, 0.852]
recall = [0.882, 0.891, 0.877, 0.855, 0.850]
f1_score = [0.882, 0.891, 0.877, 0.855, 0.850]

x = np.arange(len(models))
width = 0.2

# Create plot
plt.figure()

plt.bar(x - 1.5*width, accuracy, width, label='Accuracy')
plt.bar(x - 0.5*width, precision, width, label='Precision')
plt.bar(x + 0.5*width, recall, width, label='Recall')
plt.bar(x + 1.5*width, f1_score, width, label='F1 Score')

# Labels and title
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Performance Comparison')

plt.xticks(x, models, rotation=20)
plt.legend()

plt.tight_layout()

# Save image for PPT
plt.savefig('model_comparison.png', dpi=300)

plt.show()