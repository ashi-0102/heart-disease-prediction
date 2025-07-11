import numpy as np
import joblib
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QMessageBox, QVBoxLayout, QHBoxLayout, QWidget

class HeartDiseasePredictor(QMainWindow):
    def __init__(self, model_file, scaler_file):
        super(HeartDiseasePredictor, self).__init__()
        self.setWindowTitle("Heart Disease Predictor")
        self.setGeometry(100, 100, 600, 600)
        
        # Load SVM model and scaler
        self.svm_model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Title
        title_label = QLabel("Heart Disease Predictor", self)
        title_label.setFont(QtGui.QFont("Arial", 24, QtGui.QFont.Bold))
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create input fields for each feature
        self.entries = {}
        self.feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        form_layout = QVBoxLayout()
        
        for feature in self.feature_names:
            form_row = QHBoxLayout()
            
            label = QLabel(feature)
            label.setFont(QtGui.QFont("Arial", 12))
            label.setFixedWidth(100)
            form_row.addWidget(label)
            
            entry = QLineEdit(self)
            entry.setFont(QtGui.QFont("Arial", 12))
            entry.setFixedHeight(30)
            form_row.addWidget(entry)
            
            self.entries[feature] = entry
            form_layout.addLayout(form_row)
        
        main_layout.addLayout(form_layout)
        
        # Create predict button
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        self.predict_button.setFixedHeight(40)
        self.predict_button.clicked.connect(self.predict)
        
        main_layout.addWidget(self.predict_button)
        
        self.setCentralWidget(main_widget)

    def predict(self):
        # Get input data from entries
        input_data = []
        try:
            for feature in self.entries:
                value = float(self.entries[feature].text())
                input_data.append(value)
            
            # Convert input data to numpy array and scale it
            input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
            scaled_input_data = self.scaler.transform(input_data_as_numpy_array)
            
            # Make prediction using SVM model
            prediction = self.svm_model.predict(scaled_input_data)
            
            # Display prediction result
            if prediction[0] == 0:
                self.show_message("Result", "The person does not have heart disease.")
            else:
                self.show_message("Result", "The person has heart disease.")
        
        except ValueError:
            self.show_message("Error", "Please enter valid numerical values for all fields.")

    def show_message(self, title, message):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()

def main():
    app = QApplication([])
    model_file = 'svm_model.pkl'  # Path to the saved SVM model
    scaler_file = 'scaler.pkl'    # Path to the saved scaler
    window = HeartDiseasePredictor(model_file, scaler_file)
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
