Trained model files are written here by `backend/train_models.py`
(`energy_model.pkl`, `occupancy_model.pkl`, `fault_model.pkl` and `metrics.json`).

They are not shipped in the ZIP: the backend trains them automatically on first
start (see the main README, "First start"), so they always match your installed
scikit-learn version and the building simulation's data.
