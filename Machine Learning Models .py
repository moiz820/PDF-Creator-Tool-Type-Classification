from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming features and labels are already loaded as numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create SVM classifier
svm_classifier = SVC(kernel='linear')  # You can change the kernel type based on your requirement

# Train the classifier
svm_classifier.fit(X_train_scaled, y_train)

# Predict on the test data
y_pred = svm_classifier.predict(X_test_scaled)

# Generate classification report
report = classification_report(y_test, y_pred)
print(report)

#################################################################################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define the label mapping explicitly
label_map = {
    0: 'ACOMP exe',
    1: 'Acrobat PDFMaker for Word',
    2: 'Adobe_InDesign',
    3: 'Microsoft®_Word',
    4: 'Pages',
    5: 'PScript5.dll',
    6: 'Tex',
    7: 'Writer'
}

# Assuming features and labels are already loaded as numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # Adjust number of trees as needed

# Train the classifier
random_forest_classifier.fit(X_train_scaled, y_train)

# Predict on the test data
y_pred = random_forest_classifier.predict(X_test_scaled)

# Generate classification report with explicit label names
target_names = [label_map[i] for i in sorted(label_map.keys())]  # Ensure correct ordering
report = classification_report(y_test, y_pred, target_names=target_names)
print(report)

###################################################################################################################
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Define the label mapping explicitly
label_map = {
    0: 'ACOMP exe',
    1: 'Acrobat PDFMaker for Word',
    2: 'Adobe_InDesign',
    3: 'Microsoft®_Word',
    4: 'Pages',
    5: 'PScript5.dll',
    6: 'Tex',
    7: 'Writer'
}

# Assuming features and labels are already loaded as numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardizing the features (Recommended for distance-based models like KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)A

# Predict and evaluate the model
y_pred = knn_classifier.predict(X_test_scaled)

# Generate classification report with explicit label names
target_names = [label_map[i] for i in sorted(label_map.keys())]  # Ensure correct ordering
report = classification_report(y_test, y_pred, target_names=target_names)

print("Classification Report:\n", report)
#######################################################################################################################
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Define the label mapping explicitly
label_map = {
    0: 'ACOMP exe',
    1: 'Acrobat PDFMaker for Word',
    2: 'Adobe_InDesign',
    3: 'Microsoft®_Word',
    4: 'Pages',
    5: 'PScript5.dll',
    6: 'Tex',
    7: 'Writer'
}

# Assuming features and labels are already loaded as numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardizing the features (Recommended for distance-based models like KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)A

# Predict and evaluate the model
y_pred = knn_classifier.predict(X_test_scaled)

# Generate classification report with explicit label names
target_names = [label_map[i] for i in sorted(label_map.keys())]  # Ensure correct ordering
report = classification_report(y_test, y_pred, target_names=target_names)

print("Classification Report:\n", report)
#############################################################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Define the label mapping explicitly
label_map = {
    0: 'ACOMP exe',
    1: 'Acrobat PDFMaker for Word',
    2: 'Adobe_InDesign_16.1_(Macintosh)',
    3: 'Microsoft®_Word',
    4: 'Pages',
    5: 'PScript5.dll',
    6: 'Tex',
    7: 'Writer'
}

# Assuming features and labels are already loaded as numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardizing the features (not always needed for Decision Trees but ensures consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(X_train_scaled, y_train)

# Predict and evaluate the model
y_pred = decision_tree_classifier.predict(X_test_scaled)

# Generate classification report with explicit label names
target_names = [label_map[i] for i in sorted(label_map.keys())]  # Ensure correct ordering
report = classification_report(y_test, y_pred, target_names=target_names)

print("Classification Report:\n", report)

# Generate and plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Decision Tree Classifier')
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.show()

###################################################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Define the label mapping explicitly
label_map = {
    0: 'ACOMP exe',
    1: 'Acrobat PDFMaker for Word',
    2: 'Adobe_InDesign_',
    3: 'Microsoft®_Word',
    4: 'Pages',
    5: 'PScript5.dll',
    6: 'Tex',
    7: 'Writer'
}

# Assuming features and labels are already loaded as numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardizing the features (Recommended for gradient boosting models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Gradient Boosting classifier
gbm_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm_classifier.fit(X_train_scaled, y_train)

# Predict and evaluate the model
y_pred = gbm_classifier.predict(X_test_scaled)

# Generate classification report with explicit label names
target_names = [label_map[i] for i in sorted(label_map.keys())]  # Ensure correct ordering
report = classification_report(y_test, y_pred, target_names=target_names)

print("Classification Report:\n", report)

# Generate and plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Gradient Boosting Classifier')
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.show()


#######################################################################################################################################################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input

def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Define model parameters
input_shape = (257, 1)  # 257 features for each sample
num_classes = 6  # You have six unique labels as per your label map

# Build and compile the model
model = build_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model's architecture
model.summary()
from sklearn.model_selection import train_test_split

# Assuming 'features' and 'labels' are your data variables and already defined
features = features.reshape(-1, 257, 1)  # Reshape for CNN input, ensuring each feature vector is the correct shape for a 1D CNN
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Now let's ensure the model is built correctly. Assuming the `build_model` function is defined as provided:
input_shape = (257, 1)  # Features are 257 long, with one dimension for each feature
num_classes = len(np.unique(labels))  # Calculate the number of unique labels dynamically

model = build_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model's architecture
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_test, y_test))

# After training, evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)

#  Save the confusion matrix as a PNG file
confusion_matrix_path = "confusion_matrix.png"
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
plt.show()

#  Print the classification report with correct labels
print("Classification Report:\n", classification_report(y_test, predicted_classes, target_names=class_labels))

# Return the file for download
import shutil
shutil.move(confusion_matrix_path, r"\Users\moizz\Downloads\confmatrix8.png")
###############################################################################################################################################
















