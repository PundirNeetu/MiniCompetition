from dataset import load_datasets, preprocess_data
from features import create_features
from config import DATA_FOLDER

def main():
    # Load all datasets from the data folder
    df = load_datasets(DATA_FOLDER)
    if df is None:
        return  # Exit if no datasets could be loaded
    
    # Check for missing values before preprocessing
    missing_before = df.isnull().sum()
    print("Missing values before preprocessing:")
    print(missing_before[missing_before > 0])
    
    # Preprocess the dataset
    cleaned_df = preprocess_data(df)

    # Check for missing values after preprocessing
    missing_after = cleaned_df.isnull().sum()
    print("Missing values after preprocessing:")
    print(missing_after[missing_after > 0])

    # Create features
    enhanced_df = create_features(cleaned_df)

    # Display some information about the enhanced dataset
    print("Enhanced dataset:")
    print(enhanced_df.head())

if __name__ == "__main__":
    main()
