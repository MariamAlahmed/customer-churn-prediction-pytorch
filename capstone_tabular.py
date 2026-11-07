# ============================================================
# CAPSTONE PROJECT: TELCO CUSTOMER CHURN PREDICTION
# ============================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================
print("Downloading dataset...")
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Drop customerID (it's useless for training)
df = df.drop("customerID", axis=1)

# Clean 'TotalCharges' (convert empty strings to NaN, then fill with 0)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# Convert target 'Churn' to 0 and 1
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Identify categorical columns and apply One-Hot Encoding
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Split into Features (X) and Target (y)
X = df.drop("Churn", axis=1).values.astype(np.float32)
y = df["Churn"].values.astype(np.float32)

# Split into Train (70%), Val (15%), Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Scale features using StandardScaler (fit on train ONLY)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create DataLoaders
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

print(f"Training Samples:   {len(X_train)}")
print(f"Validation Samples: {len(X_val)}")
print(f"Test Samples:       {len(X_test)}")
print(f"Input Features:     {X_train.shape[1]}")

# ============================================================
# 2. MODEL ARCHITECTURE
# ============================================================
class ChurnMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Second hidden layer
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output layer (1 logit for binary classification)
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

model = ChurnMLP(input_size=X_train.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ============================================================
# 3. TRAINING LOOP
# ============================================================
num_epochs = 50
best_val_loss = float('inf')

print("\n--- STARTING TRAINING ---")
for epoch in range(num_epochs):
    
    # --- TRAINING PHASE ---
    model.train()
    train_loss = 0.0
    train_correct = 0
    total_train = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(y_batch)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        train_correct += (preds == y_batch).sum().item()
        total_train += len(y_batch)

    train_loss /= total_train
    train_acc = train_correct / total_train

    # --- VALIDATION PHASE ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    total_val = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            val_loss += loss.item() * len(y_batch)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            val_correct += (preds == y_batch).sum().item()
            total_val += len(y_batch)

    val_loss /= total_val
    val_acc = val_correct / total_val

    # --- SAVE BEST MODEL ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_churn_model.pt")
        saved_msg = " (Saved Best!)"
    else:
        saved_msg = ""

    # Print every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}{saved_msg}")

# ============================================================
# 4. FINAL TEST EVALUATION
# ============================================================
print("\n--- FINAL TEST EVALUATION ---")
# Load the BEST saved weights
model.load_state_dict(torch.load("best_churn_model.pt"))
model.eval()

test_loss = 0.0
test_correct = 0
total_test = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        test_loss += loss.item() * len(y_batch)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        test_correct += (preds == y_batch).sum().item()
        total_test += len(y_batch)

test_loss /= total_test
test_acc = test_correct / total_test

print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2%}")

# ============================================================
# 5. INFERENCE (Predicting a single new customer)
# ============================================================
print("\n--- INFERENCE ON A NEW CUSTOMER ---")

# Take one random customer from the test set to act as our "new" customer
new_customer_features = X_test[0].unsqueeze(0) # Shape: (1, 30)
actual_label = y_test[0].item()

model.eval()
with torch.no_grad():
    logit = model(new_customer_features)
    prob = torch.sigmoid(logit).item()
    prediction = "Churn" if prob >= 0.5 else "Stay"

print(f"Actual Status:           {'Churn' if actual_label == 1 else 'Stay'}")
print(f"Model Prediction:        {prediction}")
print(f"Probability of Churning: {prob:.2%}")