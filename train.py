import pandas as pd
from pipeline import LoanRecoveryPipeline

if __name__ == "__main__":
    print("Loading training dataset...")
    train_df = pd.read_csv("arc_loan_recovery_train.csv")

    print("Fitting Object-Oriented Pipeline and XGBoost Model...")
    pipeline = LoanRecoveryPipeline()
    pipeline.fit(train_df, target_col="recovery_rate")

    print("Saving pipeline state to model_pipeline.joblib...")
    pipeline.save("model_pipeline.joblib")

    print("Training process complete! Exported model artifact successfully.")