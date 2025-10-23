import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import os
import webbrowser

# -------------------------------
# Load ML Models
# -------------------------------
ml_bundle = joblib.load("rf_ml_only_cleaned.pkl")
hybrid_bundle = joblib.load("rf_hybrid_cleaned.pkl")
ml_model = ml_bundle["model"]
hybrid_model = hybrid_bundle["model"]

# -------------------------------
# Feature Setup
# -------------------------------
all_features = [
    'Target', 'DockingScorekcalmol', 'ADMETHighlights1ADMETpasses0ADMETnotpasses',
    'MWgmol', 'LogP', 'TPSA', 'HBD', 'HBA',
    'RotB', 'AromaticRings', 'MR', 'PlantSource'
]
excluded = ['Target', 'PlantSource']
input_features = [f for f in all_features if f not in excluded]

# -------------------------------
# Main Window Configuration
# -------------------------------
root = tk.Tk()
root.title("Phytochemical Anti-Angiogenic Predictor")
root.minsize(700, 500)
root.configure(bg="#f7f9fc")

# Make layout responsive
root.columnconfigure(0, weight=1)
root.rowconfigure(1, weight=1)

# -------------------------------
# Styles
# -------------------------------
style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background="#f7f9fc", font=("Segoe UI", 10))
style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
style.configure("TEntry", padding=5)
style.map("TButton",
          background=[("active", "#2ecc71")],
          foreground=[("active", "white")])

# -------------------------------
# Header
# -------------------------------
header = tk.Frame(root, bg="#2c3e50", pady=20)
header.grid(row=0, column=0, sticky="ew")

title_label = tk.Label(
    header,
    text="🧪 Phytochemical Activity Prediction",
    bg="#2c3e50",
    fg="white",
    font=("Segoe UI", 18, "bold")
)
title_label.pack()

subtitle_label = tk.Label(
    header,
    text="Predict anti-angiogenic or pro-angiogenic potential using trained ML models",
    bg="#2c3e50",
    fg="white",
    font=("Segoe UI", 11)
)
subtitle_label.pack()

# -------------------------------
# Scrollable Frame for Inputs
# -------------------------------
container = tk.Frame(root, bg="#f7f9fc")
container.grid(row=1, column=0, sticky="nsew", padx=15, pady=15)
container.columnconfigure(0, weight=1)
container.columnconfigure(1, weight=1)

canvas = tk.Canvas(container, bg="#f7f9fc", highlightthickness=0)
scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.grid(row=0, column=0, sticky="nsew")
scrollbar.grid(row=0, column=1, sticky="ns")

entries = {}

# -------------------------------
# Input Fields
# -------------------------------
def make_placeholder(entry, text):
    entry.insert(0, text)
    entry.config(foreground="#888")
    def on_focus_in(event):
        if entry.get() == text:
            entry.delete(0, tk.END)
            entry.config(foreground="#000")
    def on_focus_out(event):
        if not entry.get():
            entry.insert(0, text)
            entry.config(foreground="#888")
    entry.bind("<FocusIn>", on_focus_in)
    entry.bind("<FocusOut>", on_focus_out)

for i, feature in enumerate(input_features):
    lbl = ttk.Label(scrollable_frame, text=f"{feature}:")
    lbl.grid(row=i, column=0, padx=10, pady=5, sticky="w")
    entry = ttk.Entry(scrollable_frame, width=25)
    make_placeholder(entry, f"Enter {feature}")
    entry.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
    entries[feature] = entry
scrollable_frame.columnconfigure(1, weight=1)

# -------------------------------
# Core Functions
# -------------------------------
def predict():
    try:
        user_input = {}
        for feature in input_features:
            val = entries[feature].get().strip()
            if val == "" or val.startswith("Enter "):
                messagebox.showwarning("Missing Input", f"Please enter a value for {feature}")
                return
            try:
                user_input[feature] = float(val)
            except ValueError:
                user_input[feature] = val

        # Autofill fields
        user_input['Target'] = -2
        user_input['PlantSource'] = "Unknown"
        user_input['Phytochemical'] = "Unknown"

        input_df = pd.DataFrame([user_input])

        # Predictions
        ml_prob = ml_model.predict_proba(input_df)[0][1]
        hybrid_prob = hybrid_model.predict_proba(input_df)[0][1]

        result_text = (
            f"Machine Learning Model Probability:  {ml_prob:.4f}\n"
            f"Hybrid Model Probability:             {hybrid_prob:.4f}\n\n"
            f"📝 Interpretation:\n"
            f"  • Probability close to 1.0 → Anti-angiogenic\n"
            f"  • Probability close to 0.0 → Pro-angiogenic"
        )

        messagebox.showinfo("Prediction Result", result_text)

        # Save to CSV
        result_row = input_df.copy()
        result_row['ML_Prob'] = ml_prob
        result_row['Hybrid_Prob'] = hybrid_prob
        file_exists = os.path.isfile("GUIResults.csv")
        result_row.to_csv("GUIResults.csv", mode='a', header=not file_exists, index=False)

        status_var.set("✅ Prediction saved successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")
        status_var.set("❌ Prediction failed.")

def clear_inputs():
    for feature, entry in entries.items():
        entry.delete(0, tk.END)
        make_placeholder(entry, f"Enter {feature}")
    status_var.set("🧹 Cleared all fields.")

def open_csv():
    filepath = os.path.abspath("GUIResults.csv")
    if os.path.exists(filepath):
        webbrowser.open(filepath)
    else:
        messagebox.showinfo("No File", "No results file found yet.")

# -------------------------------
# Action Buttons
# -------------------------------
button_frame = tk.Frame(root, bg="#f7f9fc", pady=10)
button_frame.grid(row=2, column=0, sticky="ew")

predict_btn = ttk.Button(button_frame, text="Run Prediction", command=predict)
predict_btn.grid(row=0, column=0, padx=10)

clear_btn = ttk.Button(button_frame, text="Clear Inputs", command=clear_inputs)
clear_btn.grid(row=0, column=1, padx=10)

open_btn = ttk.Button(button_frame, text="Open Results", command=open_csv)
open_btn.grid(row=0, column=2, padx=10)

button_frame.columnconfigure((0, 1, 2), weight=1)

# -------------------------------
# Status Bar
# -------------------------------
status_var = tk.StringVar(value="Ready")
status_bar = tk.Label(root, textvariable=status_var, anchor="w", bg="#ecf0f1", relief="sunken")
status_bar.grid(row=3, column=0, sticky="ew")

# Make rows responsive
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=0)
root.rowconfigure(3, weight=0)
root.columnconfigure(0, weight=1)

root.mainloop()
