import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import os

# ============================
# Load ML Models
# ============================
ml_bundle = joblib.load("rf_ml_only_cleaned.pkl")
hybrid_bundle = joblib.load("rf_hybrid_cleaned.pkl")
ml_model = ml_bundle["model"]
hybrid_model = hybrid_bundle["model"]

# ============================
# Feature Setup & Descriptions
# ============================
all_features = [
    'Target', 'DockingScorekcalmol', 'ADMETHighlights1ADMETpasses0ADMETnotpasses',
    'MWgmol', 'LogP', 'TPSA', 'HBD', 'HBA',
    'RotB', 'AromaticRings', 'MR', 'PlantSource'
]
excluded = ['Target', 'PlantSource']
input_features = [f for f in all_features if f not in excluded]

feature_descriptions = {
    'DockingScorekcalmol': "Binding affinity (kcal/mol). Lower = stronger binding.",
    'ADMETHighlights1ADMETpasses0ADMETnotpasses': "ADMET filter (1=Pass, 0=Fail).",
    'MWgmol': "Molecular weight (g/mol). Typical drug-like range < 500.",
    'LogP': "Lipophilicity (LogP). Optimal range ~0–5.",
    'TPSA': "Topological Polar Surface Area (Å²). Affects absorption.",
    'HBD': "Hydrogen bond donors.",
    'HBA': "Hydrogen bond acceptors.",
    'RotB': "Rotatable bonds.",
    'AromaticRings': "Number of aromatic rings.",
    'MR': "Molar refractivity."
}

# ============================
# Main Window
# ============================
root = tk.Tk()
root.title("Phytochemical Anti-Angiogenic Predictor")
root.geometry("800x550")
root.minsize(650, 400)
root.configure(bg="#f7f9fc")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# ============================
# Styles
# ============================
style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Segoe UI", 12, "bold"), padding=8)
style.configure("TLabel", background="#f7f9fc", font=("Segoe UI", 13))
style.configure("TProgressbar", thickness=20)

# ============================
# Tooltip
# ============================
tooltip = None
def show_tooltip(event, text):
    global tooltip
    if tooltip:
        tooltip.destroy()
    tooltip = tk.Toplevel(root)
    tooltip.wm_overrideredirect(True)
    tooltip.configure(bg="#2c3e50")
    label = tk.Label(tooltip, text=text, bg="#2c3e50", fg="white", font=("Segoe UI", 10), wraplength=300, justify="left")
    label.pack(ipadx=6, ipady=4)
    x, y = event.x_root + 10, event.y_root + 10
    tooltip.wm_geometry(f"+{x}+{y}")

def hide_tooltip(event):
    global tooltip
    if tooltip:
        tooltip.destroy()
        tooltip = None

# ============================
# Global Variables
# ============================
user_inputs = {}
current_index = 0

# ============================
# Frames
# ============================
welcome_frame = tk.Frame(root, bg="#f7f9fc")
input_frame = tk.Frame(root, bg="#f7f9fc")
summary_frame = tk.Frame(root, bg="#f7f9fc")   # NEW ✅
result_frame = tk.Frame(root, bg="#f7f9fc")

for frame in (welcome_frame, input_frame, summary_frame, result_frame):
    frame.grid(row=0, column=0, sticky="nsew")
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)

# ============================
# Welcome Frame
# ============================
welcome_inner = tk.Frame(welcome_frame, bg="#f7f9fc")
welcome_inner.grid(row=0, column=0, sticky="nsew")

title_label = tk.Label(
    welcome_inner,
    text="🧪 Phytochemical Activity Prediction",
    bg="#f7f9fc",
    fg="#2c3e50",
    font=("Segoe UI", 20, "bold")
)
title_label.pack(pady=30)

welcome_text = tk.Label(
    welcome_inner,
    text="Welcome!\nThis tool predicts whether a compound is Anti- or Pro-Angiogenic.\n\nClick below to begin entering each molecular feature step by step.",
    bg="#f7f9fc",
    font=("Segoe UI", 13),
    justify="center"
)
welcome_text.pack(pady=20)

start_btn = ttk.Button(
    welcome_inner,
    text="Start Input",
    command=lambda: [show_frame(input_frame), start_input()]
)
start_btn.pack(pady=30)

# ============================
# Input Frame
# ============================
input_inner = tk.Frame(input_frame, bg="#f7f9fc")
input_inner.place(relx=0.5, rely=0.5, anchor="center")

progress_label = tk.Label(input_inner, text="", bg="#f7f9fc", font=("Segoe UI", 13, "bold"))
progress_label.pack(pady=10)

progress_bar = ttk.Progressbar(input_inner, orient="horizontal", mode="determinate", length=450)
progress_bar.pack(pady=10)

feature_label = tk.Label(input_inner, text="", bg="#f7f9fc", font=("Segoe UI", 14))
feature_label.pack(pady=15)

feature_entry = ttk.Entry(input_inner, font=("Segoe UI", 13), width=35, justify="center")
feature_entry.pack(pady=10)

button_frame = tk.Frame(input_inner, bg="#f7f9fc")
button_frame.pack(pady=20)

feature_label.bind("<Enter>", lambda e: show_tooltip(e, feature_descriptions.get(input_features[0], "")))
feature_label.bind("<Leave>", hide_tooltip)

def handle_next_feature():
    global current_index
    value = feature_entry.get().strip()
    if value == "":
        messagebox.showwarning("Missing Input", "Please enter a value before proceeding.")
        return

    feature_name = input_features[current_index]
    try:
        user_inputs[feature_name] = float(value)
    except ValueError:
        user_inputs[feature_name] = value

    feature_entry.delete(0, tk.END)
    current_index += 1

    if current_index < len(input_features):
        update_feature_prompt()
    else:
        build_summary()
        show_frame(summary_frame)

def update_feature_prompt():
    feature_name = input_features[current_index]
    step = current_index + 1
    total = len(input_features)
    progress_label.config(text=f"Step {step} of {total}")
    feature_label.config(text=f"Enter value for: {feature_name}")
    progress_bar['maximum'] = total
    progress_bar['value'] = step - 1
    feature_label.bind("<Enter>", lambda e: show_tooltip(e, feature_descriptions.get(feature_name, "")))
    feature_label.bind("<Leave>", hide_tooltip)
    feature_entry.focus()

next_btn = ttk.Button(button_frame, text="Next ➝", command=handle_next_feature)
next_btn.grid(row=0, column=0, padx=10)

def start_input():
    global current_index
    current_index = 0
    user_inputs.clear()
    feature_entry.delete(0, tk.END)
    update_feature_prompt()

# ============================
# Summary Frame ✅
# ============================
summary_inner = tk.Frame(summary_frame, bg="#f7f9fc")
summary_inner.place(relx=0.5, rely=0.5, anchor="center")

summary_title = tk.Label(
    summary_inner,
    text="📝 Review Your Inputs",
    bg="#f7f9fc",
    fg="#2c3e50",
    font=("Segoe UI", 18, "bold")
)
summary_title.pack(pady=20)

tree = ttk.Treeview(summary_inner, columns=("Feature", "Value"), show="headings", height=10)
tree.heading("Feature", text="Feature")
tree.heading("Value", text="Value")
tree.column("Feature", width=250, anchor="center")
tree.column("Value", width=200, anchor="center")
tree.pack(pady=10)

button_summary = tk.Frame(summary_inner, bg="#f7f9fc")
button_summary.pack(pady=20)

def build_summary():
    for row in tree.get_children():
        tree.delete(row)
    for k, v in user_inputs.items():
        tree.insert("", tk.END, values=(k, v))

def confirm_and_predict():
    run_prediction()
    show_frame(result_frame)

def edit_inputs():
    global current_index
    current_index = 0
    feature_entry.delete(0, tk.END)
    show_frame(input_frame)
    update_feature_prompt()

confirm_btn = ttk.Button(button_summary, text="✅ Confirm & Predict", command=confirm_and_predict)
confirm_btn.grid(row=0, column=0, padx=10)

edit_btn = ttk.Button(button_summary, text="✏️ Edit Inputs", command=edit_inputs)
edit_btn.grid(row=0, column=1, padx=10)

# ============================
# Result Frame
# ============================
result_inner = tk.Frame(result_frame, bg="#f7f9fc")
result_inner.place(relx=0.5, rely=0.5, anchor="center")

result_title = tk.Label(
    result_inner,
    text="📊 Prediction Results",
    bg="#f7f9fc",
    fg="#2c3e50",
    font=("Segoe UI", 18, "bold")
)
result_title.pack(pady=20)

ml_result_label = tk.Label(result_inner, text="", bg="#f7f9fc", font=("Segoe UI", 14))
ml_result_label.pack(pady=10)

hybrid_result_label = tk.Label(result_inner, text="", bg="#f7f9fc", font=("Segoe UI", 14))
hybrid_result_label.pack(pady=10)

def run_prediction():
    user_inputs['Target'] = -2
    user_inputs['PlantSource'] = "Unknown"
    user_inputs['Phytochemical'] = "Unknown"

    input_df = pd.DataFrame([user_inputs])
    ml_prob = ml_model.predict_proba(input_df)[0][1]
    hybrid_prob = hybrid_model.predict_proba(input_df)[0][1]

    ml_class = "ANTI-ANGIOGENIC" if ml_prob >= 0.5 else "PRO-ANGIOGENIC"
    hy_class = "ANTI-ANGIOGENIC" if hybrid_prob >= 0.5 else "PRO-ANGIOGENIC"

    ml_result_label.config(
        text=f"🔹 ML Model: {ml_class}  (P = {ml_prob:.4f})"
    )
    hybrid_result_label.config(
        text=f"🔸 Hybrid Model: {hy_class}  (P = {hybrid_prob:.4f})"
    )

    result_row = input_df.copy()
    result_row['ML_Prob'] = ml_prob
    result_row['Hybrid_Prob'] = hybrid_prob
    result_row['ML_Pred'] = ml_class
    result_row['Hybrid_Pred'] = hy_class

    file_exists = os.path.isfile("GUIResults.csv")
    result_row.to_csv("GUIResults.csv", mode='a', header=not file_exists, index=False)

button_result_frame = tk.Frame(result_inner, bg="#f7f9fc")
button_result_frame.pack(pady=20)

restart_btn = ttk.Button(button_result_frame, text="🔁 Start Over", command=lambda: [show_frame(input_frame), start_input()])
restart_btn.grid(row=0, column=0, padx=10)

exit_btn = ttk.Button(button_result_frame, text="❌ Exit", command=root.destroy)
exit_btn.grid(row=0, column=1, padx=10)

# ============================
# Frame Navigation
# ============================
def show_frame(frame):
    frame.tkraise()

show_frame(welcome_frame)
root.mainloop()
