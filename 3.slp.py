import numpy as np
import pandas as pd

print("Start: Training Perceptron...")

df = pd.read_csv('credit_data_1m.csv')
X = df.drop('Default', axis=1).values
y = df['Default'].values

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1.0 
X = (X - X_mean) / X_std

X_train, y_train = X[:70000], y[:70000]
X_val, y_val = X[70000:85000], y[70000:85000]
X_test, y_test = X[85000:100000], y[85000:100000]

def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-1 * z))  

def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


np.random.seed(42)
w = np.random.randn(X.shape[1]) * np.sqrt(2 / X.shape[1])
b = 0.0
alpha = 0.1
epochs = 64
best_val_accuracy = 0
best_w, best_b = None, None

for epoch in range(epochs):

    # Накопление градиентов (как в файле)
    dw_accum = np.zeros_like(w)
    db_accum = 0.0
    total_loss = 0.0
    
    for i in range(len(X_train)):
        x_i = X_train[i]
        y_true_i = y_train[i]
        
        z_i = np.dot(x_i, w) + b
        y_pred_i = sigmoid(z_i)
        
        error = y_pred_i - y_true_i
        dw_accum += error * x_i
        db_accum += error
        
        total_loss += compute_loss(np.array([y_true_i]), np.array([y_pred_i]))
    
    m = len(X_train)
    dw = dw_accum / m
    db = db_accum / m

    w -= alpha * dw
    b -= alpha * db
    
    z_val = np.dot(X_val, w) + b
    y_pred_val = sigmoid(z_val)
    val_accuracy = np.mean((y_pred_val > 0.5) == y_val)
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_w, best_b = w.copy(), b.copy()

    print(f"Epoch {epoch + 1}:")
    print(f"  Train Loss: {total_loss / m:.6f}")
    print(f"  Val Accuracy: {val_accuracy:.4f} (Best: {best_val_accuracy:.4f})")

w, b = best_w, best_b
z_test = np.dot(X_test, w) + b
y_pred_test = (sigmoid(z_test) > 0.5).astype(int)


true_positives = np.sum((y_pred_test == 1) & (y_test == 1))
false_positives = np.sum((y_pred_test == 1) & (y_test == 0))
false_negatives = np.sum((y_pred_test == 0) & (y_test == 1))

test_accuracy = np.mean(y_pred_test == y_test)
precision = true_positives / (true_positives + false_positives + 1e-10)
recall = true_positives / (true_positives + false_negatives + 1e-10)

print(f"\nTest Metrics:")
print(f"  Accuracy: {test_accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")

with open('w&b.txt', 'w') as f:
    f.write("Weights (w):\n")
    f.write(' '.join(map(str, w)) + '\n')
    f.write(f"Bias (b): {b}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Precision: {precision:.4f}\n")
    f.write(f"Test Recall: {recall:.4f}\n")

print("Training completed. Metrics saved to 'w&b.txt'")