import pandas as pd
from typing import List, Dict, Optional, Tuple

# Additional utility functions
def analyze_cleaning_results(df: pd.DataFrame) -> Dict:
    """Analyze the cleaning results for quality assessment"""
    
    analysis = {
        'success_rates': {
            'total_processed': len(df),
            'successful_cleaning': len(df[df['processing_status'] == 'success']),
            'failed_cleaning': len(df[df['processing_status'] == 'failed'])
        },
        'ingredient_stats': {
            'avg_ingredients_before': df['ingredients'].str.len().mean() if 'ingredients' in df else 0,
            'avg_ingredients_after': df['gemini_count'].mean(),
            'max_ingredients': df['gemini_count'].max(),
            'min_ingredients': df['gemini_count'].min()
        },
        'common_issues': []
    }
    
    # Identify common patterns in failed entries
    failed_entries = df[df['processing_status'] == 'failed']
    if len(failed_entries) > 0:
        analysis['common_issues'] = failed_entries['ingredients'].head(10).tolist()
    
    return analysis

def export_for_manual_review(df: pd.DataFrame, output_path: str = 'manual_review.csv'):
    """Export entries that may need manual review"""
    
    # Criteria for manual review
    review_needed = df[
        (df['processing_status'] == 'failed') |  # Failed processing
        (df['gemini_count'] == 0) |              # No ingredients extracted
        (df['gemini_count'] > 15)                # Too many ingredients (potential error)
    ]
    
    review_df = review_needed[['ingredients', 'gemini_cleaned', 'processing_status', 'gemini_count']].copy()
    review_df.to_csv(output_path, index=False)
    
    print(f"Exported {len(review_df)} entries for manual review to {output_path}")
    return review_df