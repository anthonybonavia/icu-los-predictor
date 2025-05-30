
import sys
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("best_model.joblib")
scaler = joblib.load("scaler.pkl")

def predict_icu_los_risk(bun, creatinine, ventilated, sofa):
    input_data = np.array([[bun, creatinine, ventilated, sofa]])
    standardized = scaler.transform(input_data)
    prob = model.predict(standardized)[0][0]

    print(f"\nInput:")
    print(f"BUN: {bun} | Creatinine: {creatinine} | Ventilated: {ventilated} | SOFA: {sofa}")
    print(f"Standardized Input: {standardized.flatten()}")
    print(f"Predicted Probability of ICU LOS > 7 days: {prob:.3f}")

    # Threshold flags
    sensitivity_flag = "HIGH" if prob >= 0.26 else "LOW"
    specificity_flag = "HIGH" if prob >= 0.63 else "LOW"
    print(f"\nInterpretation:")
    print(f"Sensitivity-Optimized Flag (>0.26): {sensitivity_flag}")
    print(f"Specificity-Optimized Flag (>0.63): {specificity_flag}")
    print(f"\nNotes:")
    print(f" - Use the sensitivity-optimized threshold to avoid missing patients with long ICU stays.")
    print(f" - Use the specificity-optimized threshold when false positives carry high cost.")

# Main
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python icu_los_predict_cli.py <bun> <creatinine> <ventilated> <sofa>")
        print("Example: python icu_los_predict_cli.py 22.5 1.8 1 10")
        sys.exit(1)

    bun = float(sys.argv[1])
    creatinine = float(sys.argv[2])
    ventilated = int(sys.argv[3])
    sofa = int(sys.argv[4])

    predict_icu_los_risk(bun, creatinine, ventilated, sofa)
