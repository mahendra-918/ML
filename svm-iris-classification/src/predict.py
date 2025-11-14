import numpy as np
import joblib
from sklearn import datasets

print("=" * 60)
print("IRIS FLOWER SPECIES PREDICTOR")
print("=" * 60)

# Load the best model and scaler
print("\n[Loading Model...]")
model = joblib.load('../models/best_svm.pkl')
scaler = joblib.load('../models/scaler.pkl')
print("✓ Model loaded successfully!")

# Load iris data for reference
iris = datasets.load_iris()
species_names = iris.target_names

print("\n" + "=" * 60)
print("IRIS DATASET REFERENCE")
print("=" * 60)
print("\nFeature Ranges (in cm):")
print("  - Sepal Length: 4.3 - 7.9")
print("  - Sepal Width:  2.0 - 4.4")
print("  - Petal Length: 1.0 - 6.9")
print("  - Petal Width:  0.1 - 2.5")

print("\nSpecies:")
print("  0: Setosa")
print("  1: Versicolor")
print("  2: Virginica")

def predict_species():
    print("\n" + "=" * 60)
    print("ENTER FLOWER MEASUREMENTS")
    print("=" * 60)
    try:
        # Get user input
        sepal_length = float(input("\nSepal Length (cm): "))
        sepal_width = float(input("Sepal Width (cm): "))
        petal_length = float(input("Petal Length (cm): "))
        petal_width = float(input("Petal Width (cm): "))
        
        # Create feature array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        # Scale features
        features_scaled = scaler.transform(features)

        #Make prediction
        prediction = model.predict(features_scaled)[0]
        species = species_names[prediction]

        # Get decisions function scroes (confidence)
        decision_scores = model.decision_function(features_scaled)[0]

        # Display results
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"\n🌸 Predicted Species: {species.upper()}")
        print(f"   Class: {prediction}")

        print("\nDecision Function Scores:")
        for i, score in enumerate(decision_scores):
            print(f"  {species_names[i]}: {score:.4f}")
        
        print("\n" + "=" * 60)

    except ValueError as e:
        print(f"\n❌ Error: Please enter valid numeric values!")
        return
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return


# Main loop
while True:
    predict_species()
    
    print("\nWould you like to make another prediction?")
    choice = input("Enter 'y' for yes, any other key to exit: ").strip().lower()
    
    if choice != 'y':
        print("\n" + "=" * 60)
        print("Thank you for using Iris Species Predictor!")
        print("=" * 60)
        break
    




