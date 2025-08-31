#!/usr/bin/env python3
"""
Thai Ingredient Data Cleaning Framework
=====================================

This script provides a complete pipeline for cleaning Thai food ingredient data.
It combines rule-based preprocessing with LLM-powered standardization.
"""

import re
import pandas as pd
from typing import List, Dict, Set
import json

class ThaiIngredientCleaner:
    def __init__(self):
        # Common Thai measurement units to remove
        self.measurement_patterns = [
            r'\d+\s*(ถ้วย|แก้ว|ชิ้น|ลูก|ห่อ|จาน|กิโล|กรัม|oz|ml)',
            r'\d+\s*(ครึ่ง|หนึ่ง|สอง|สาม|สี่|ห้า)',
            r'(ครึ่ง|หนึ่ง)\s*(ถ้วย|แก้ว|ชิ้น|ลูก|จาน)',
            r'\d+\.*\d*\s*(แก้ว|ถ้วย|ชิ้น|ลูก|ห่อ)'
        ]
        
        # Cooking methods to remove
        self.cooking_methods = [
            'ทอด', 'ย่าง', 'ต้ม', 'ผัด', 'นึ่ง', 'ปิ้ง', 'เผา', 'ต้ม', 
            'กรอบ', 'เปื่อย', 'หวาน', 'เปรี้ยว', 'เผ็ด', 'เย็น', 'ร้อน'
        ]
        
        # Brand names and commercial terms
        self.brand_patterns = [
            r'(KFC|7-11|โอวัลติน|นูเทลล่า|B-ready)',
            r'(แบบ|รส|ใส่|ผสม|จาก)',
        ]
        
        # Ingredient standardization dictionary
        self.ingredient_mapping = {
            # Noodles
            'ก้วยเตี๋ยว': 'ก๋วยเตี๋ยว',
            'เส้นเล็ก': 'ก๋วยเตี๋ยวเส้นเล็ก',
            'เส้นใหญ่': 'ก๋วยเตี๋ยวเส้นใหญ่',
            'เส้นหมี่': 'หมี่',
            'รามยอนแห้ง': 'รามยอน',
            
            # Rice
            'ข้าวสวย': 'ข้าว',
            'ข้าวต้ม': 'โจ๊ก',
            'ข้าวโอ๊ต': 'โอ๊ต',
            
            # Beverages
            'น้ำเปล่า': 'น้ำ',
            'นมวัว': 'นม',
            
            # Fruits
            'มะม่วงสุก': 'มะม่วง',
            'ส้มโอ': 'ส้ม',
        }
    
    def step1_basic_cleaning(self, text: str) -> str:
        """Step 1: Basic text cleaning"""
        if pd.isna(text) or text.strip() == '-':
            return ""
        
        # Remove quotes and special characters
        text = re.sub(r'["\'""]', '', text)
        text = re.sub(r'[-–—]', ' ', text)
        
        # Remove measurements
        for pattern in self.measurement_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove brand names
        for pattern in self.brand_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def step2_remove_cooking_descriptions(self, text: str) -> str:
        """Step 2: Remove cooking methods and descriptions"""
        # Remove long cooking descriptions (sentences with cooking verbs)
        cooking_sentence_pattern = r'[^.]*?(สับ|ย่าง|ทอด|ต้ม|นำไป|แล้ว)[^.]*?[.]?'
        text = re.sub(cooking_sentence_pattern, '', text)
        
        # Remove cooking methods attached to ingredients
        for method in self.cooking_methods:
            text = re.sub(rf'\b\w+{method}\b', lambda m: m.group().replace(method, ''), text)
        
        # Clean up
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def step3_extract_ingredients(self, text: str) -> List[str]:
        """Step 3: Extract individual ingredients"""
        if not text:
            return []
        
        # Split by common separators
        ingredients = re.split(r'[,\s]+', text)
        
        # Filter out empty strings and very short words (likely particles)
        ingredients = [ing.strip() for ing in ingredients if len(ing.strip()) > 1]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ingredients = []
        for ing in ingredients:
            if ing not in seen:
                seen.add(ing)
                unique_ingredients.append(ing)
        
        return unique_ingredients
    
    def step4_standardize_ingredients(self, ingredients: List[str]) -> List[str]:
        """Step 4: Standardize ingredient names"""
        standardized = []
        
        for ingredient in ingredients:
            # Apply mapping
            if ingredient in self.ingredient_mapping:
                ingredient = self.ingredient_mapping[ingredient]
            
            # Remove cooking methods still attached
            for method in self.cooking_methods:
                if ingredient.endswith(method):
                    ingredient = ingredient[:-len(method)]
            
            ingredient = ingredient.strip()
            if ingredient:
                standardized.append(ingredient)
        
        return standardized
    
    def process_entry(self, text: str) -> Dict:
        """Process a single entry through all steps"""
        original = text
        
        # Step 1: Basic cleaning
        cleaned = self.step1_basic_cleaning(text)
        
        # Step 2: Remove cooking descriptions
        no_cooking = self.step2_remove_cooking_descriptions(cleaned)
        
        # Step 3: Extract ingredients
        ingredients_raw = self.step3_extract_ingredients(no_cooking)
        
        # Step 4: Standardize
        ingredients_final = self.step4_standardize_ingredients(ingredients_raw)
        
        return {
            'original': original,
            'cleaned_text': no_cooking,
            'ingredients_raw': ingredients_raw,
            'ingredients_final': ingredients_final,
            'ingredient_count': len(ingredients_final)
        }
    
    def process_dataframe(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Process entire dataframe"""
        results = []
        
        for idx, row in df.iterrows():
            result = self.process_entry(str(row[column_name]))
            result['row_id'] = idx
            results.append(result)
        
        results_df = pd.DataFrame(results)
        return results_df



# =============================================================================
# LLM PROMPT TEMPLATES FOR FURTHER CLEANING
# =============================================================================

class LLMPromptGenerator:
    """Generate prompts for LLM-based ingredient cleaning"""
    
    @staticmethod
    def generate_standardization_prompt(ingredients_list: List[str]) -> str:
        """Generate prompt for ingredient standardization"""
        ingredients_str = ", ".join(ingredients_list)
        
        prompt = f"""
You are an expert in Thai cuisine and ingredients. Your task is to clean and standardize a list of Thai food ingredients.

RULES:
1. Standardize ingredient names to their most common Thai form
2. Remove any remaining cooking methods (ทอด, ย่าง, ต้ม, etc.)
3. Group similar ingredients (e.g., different types of the same ingredient)
4. Remove non-food items or unclear terms
5. Keep only actual ingredients, not dishes or preparations

INPUT INGREDIENTS: {ingredients_str}

OUTPUT FORMAT: Return only a comma-separated list of clean, standardized ingredients in Thai.

Example:
Input: หมูทอด, ไก่ย่าง, ข้าวสวย, น้ำเปล่า
Output: หมู, ไก่, ข้าว, น้ำ

CLEANED INGREDIENTS:"""
        
        return prompt
    
    @staticmethod
    def generate_dish_breakdown_prompt(dish_name: str) -> str:
        """Generate prompt for breaking down complex dishes into ingredients"""
        
        prompt = f"""
You are an expert Thai chef. Break down this Thai dish into its main ingredients.

DISH: {dish_name}

RULES:
1. List only the main ingredients, not cooking methods
2. Use standard Thai ingredient names
3. Include seasonings and sauces if they're key components
4. Don't include water, oil, or basic cooking ingredients unless they're special

OUTPUT FORMAT: Comma-separated list of ingredients in Thai

INGREDIENTS:"""
        
        return prompt
    
    @staticmethod
    def generate_validation_prompt(ingredients: List[str]) -> str:
        """Generate prompt for validating ingredient list"""
        
        ingredients_str = ", ".join(ingredients)
        
        prompt = f"""
Review this list of Thai ingredients and identify any issues:

INGREDIENTS: {ingredients_str}

Check for:
1. Misspelled Thai words
2. Non-food items
3. Duplicate ingredients with different names
4. Cooking methods still attached to ingredients
5. Brand names or commercial terms

OUTPUT FORMAT:
ISSUES FOUND: [list any problems]
CORRECTED LIST: [clean ingredient list]
CONFIDENCE: [High/Medium/Low]

REVIEW:"""
        
        return prompt


# =============================================================================
# PROCESSING PIPELINE CONFIGURATION
# =============================================================================

class ProcessingPipeline:
    """Complete processing pipeline with configurable steps"""
    
    def __init__(self, use_llm: bool = False):
        self.cleaner = ThaiIngredientCleaner()
        self.prompt_generator = LLMPromptGenerator()
        self.use_llm = use_llm
    
    def run_pipeline(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Run complete cleaning pipeline"""
        
        print("Step 1: Rule-based cleaning...")
        results_df = self.cleaner.process_dataframe(df, column_name)
        
        if self.use_llm:
            print("Step 2: LLM-based refinement...")
            results_df = self.apply_llm_cleaning(results_df)
        
        print("Step 3: Final validation...")
        results_df = self.final_validation(results_df)
        
        return results_df
    
    def apply_llm_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply LLM cleaning (placeholder - requires actual LLM integration)"""
        # This would integrate with your LLM API
        print("Note: LLM integration requires API setup")
        return df
    
    def final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation and statistics"""
        df['is_valid'] = df['ingredients_final'].apply(lambda x: len(x) > 0)
        df['has_duplicates'] = df['ingredients_final'].apply(
            lambda x: len(x) != len(set(x)) if x else False
        )
        
        return df
    
    def generate_report(self, df: pd.DataFrame) -> Dict:
        """Generate cleaning report"""
        total_entries = len(df)
        valid_entries = df['is_valid'].sum()
        avg_ingredients = df['ingredient_count'].mean()
        
        report = {
            'total_entries': total_entries,
            'valid_entries': valid_entries,
            'success_rate': valid_entries / total_entries * 100,
            'avg_ingredients_per_entry': avg_ingredients,
            'most_common_ingredients': self.get_ingredient_frequency(df)
        }
        
        return report
    
    def get_ingredient_frequency(self, df: pd.DataFrame) -> Dict:
        """Get frequency of ingredients across all entries"""
        all_ingredients = []
        for ingredients_list in df['ingredients_final']:
            all_ingredients.extend(ingredients_list)
        
        from collections import Counter
        return dict(Counter(all_ingredients).most_common(20))


# Export functions for easy use
__all__ = ['ThaiIngredientCleaner', 'LLMPromptGenerator', 'ProcessingPipeline']