import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.cleaner import ThaiIngredientCleaner

# Example usage and testing
def main():
    
    # Create test dataframe
    df = pd.read_excel('data/raw/chatbot_phase2.xlsx', sheet_name='Total')
    menu_df = df['List menu']
    
    # Initialize cleaner
    cleaner = ThaiIngredientCleaner()
    
    # Process data
    results_df = cleaner.process_dataframe(df, 'List menu')
    
    # Display results
    results_df.to_excel('data/processed/response_cleaned.xlsx')

if __name__ == "__main__":
    main()