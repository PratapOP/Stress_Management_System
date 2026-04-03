import matplotlib.pyplot as plt

models = ['Logistic Regression', 'Random Forest', 'SVM', 'Decision Tree', 'KNN']

accuracy = [0.882, 0.891, 0.877, 0.855, 0.850]
precision = [0.882, 0.893, 0.877, 0.855, 0.852]
recall = [0.882, 0.891, 0.877, 0.855, 0.850]
f1_score = [0.882, 0.891, 0.877, 0.855, 0.850]

# 1. Accuracy Graph
plt.figure()
plt.plot(models, accuracy, marker='o')
plt.title('Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('accuracy.png', dpi=300)
plt.show()

# 2. Precision Graph
plt.figure()
plt.plot(models, precision, marker='o')
plt.title('Precision Comparison')
plt.xlabel('Models')
plt.ylabel('Precision')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('precision.png', dpi=300)
plt.show()

# 3. Recall Graph
plt.figure()
plt.plot(models, recall, marker='o')
plt.title('Recall Comparison')
plt.xlabel('Models')
plt.ylabel('Recall')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('recall.png', dpi=300)
plt.show()

# 4. F1 Score Graph
plt.figure()
plt.plot(models, f1_score, marker='o')
plt.title('F1 Score Comparison')
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('f1_score.png', dpi=300)
plt.show()