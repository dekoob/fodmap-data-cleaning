#!/usr/bin/env python3
"""
Thai Ingredient Data Cleaning Framework - REFINED VERSION
=========================================================

This script provides a complete pipeline for cleaning Thai food ingredient data
with enhanced rules based on real-world data analysis.
"""

import re
import pandas as pd
from typing import List, Dict, Set
import json

class ThaiIngredientCleaner:
    def __init__(self):
        # Enhanced measurement patterns - more comprehensive
        self.measurement_patterns = [
            r'\d+\s*(ถ้วย|แก้ว|ชิ้น|ลูก|ห่อ|จาน|กิโล|กรัม|oz|ml|ตัว|ผล|เม็ด|สกูป|แผ่น)',
            r'\d+\s*(ครึ่ง|หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ)',
            r'(ครึ่ง|หนึ่ง|หน่อย|นิด|เล็กน้อย)\s*(ถ้วย|แก้ว|ชิ้น|ลูก|จาน|ตัว|ผล)',
            r'\d+/\d+\s*(ถ้วย|แก้ว|ชิ้น|ลูก|สกูป)',  # Fractions like 1/4
            r'\d+\.*\d*\s*(แก้ว|ถ้วย|ชิ้น|ลูก|ห่อ|ตัว|ผล|ml|เม็ด)',
            r'\b\d+\s*(?=\w)',  # Numbers before words
        ]
        
        # Expanded cooking methods - more comprehensive
        self.cooking_methods = [
            'ทอด', 'ย่าง', 'ต้ม', 'ผัด', 'นึ่ง', 'ปิ้ง', 'เผา', 'อบ', 'ลวก',
            'กรอบ', 'เปื่อย', 'หวาน', 'เปรี้ยว', 'เผ็ด', 'เย็น', 'ร้อน', 'แห้ง',
            'สด', 'เก่า', 'ใหม่', 'ดิบ', 'สุก', 'เด้ง', 'แดง', 'ขาว', 'เขียว',
            'คั่ว', 'ชุบ', 'ราด', 'จิ้ม', 'คลุก', 'ผสม', 'เกรียบ', 'กวน'
        ]
        
        # Enhanced brand patterns and commercial terms
        self.brand_patterns = [
            r'(KFC|7-11|โอวัลติน|นูเทลล่า|B-ready|Ensure|nutella|บอดีคีย์)',
            r'(แบบ|รส|ใส่|ผสม|จาก|ยี่ห้อ|แบรนด์)',
            r'(ครึ่งจาน|ตัวเล็ก|ค้างคืน|จากเมื่อวาน|อุ่นร้อน)',
            r'(โฮมสวิท|ไลเบอรี่|แลคโตสฟรี)',
        ]
        
        # Dish description patterns to remove
        self.dish_descriptions = [
            r'"[^"]*"',  # Text in quotes
            r'มี[^"]*?(?=\s|$)',  # "มี" descriptions
            r'ประกอบด้วย[^"]*?(?=\s|$)',  # "ประกอบด้วย" descriptions
            r'ส่วนประกอบ[^"]*?(?=\s|$)',  # Component descriptions
            r'ใส่[^"]*?(?=\s|$)',  # "ใส่" descriptions
            r'ทำจาก[^"]*?(?=\s|$)',  # "ทำจาก" descriptions
        ]
        
        # Non-food items to remove
        self.non_food_items = [
            'มื้อเช้า', 'มื้อกลางวัน', 'มื้อเย็น', 'เช้า', 'กลางวัน', 'เย็น',
            'ก่อนอาหาร', 'หลังอาหาร', 'ไม่ได้ทาน', 'ไม่ได้ทานอะไร',
            'ครึ่งจาน', 'ใส่', 'แต่ง', 'กลิ่น', 'นำมา', 'แล้ว', 'เมื่อ',
            'เครื่องเคียง:'
        ]
        
        # Enhanced ingredient mapping with more variants observed
        self.ingredient_mapping = {
            # Noodles - more variants
            'ก้วยเตี๋ยว': 'ก๋วยเตี๋ยว',
            'ก๊วยเตี๋ยว': 'ก๋วยเตี๋ยว',
            'เส้นเล็ก': 'ก๋วยเตี๋ยวเส้นเล็ก',
            'เส้นใหญ่': 'ก๋วยเตี๋ยวเส้นใหญ่',
            'เส้นหมี่': 'หมี่',
            'รามยอนแห้ง': 'รามยอน',
            'รามยอน': 'รามยอน',
            'บะหมี่': 'บะหมี่',
            
            # Rice varieties
            'ข้าวสวย': 'ข้าว',
            'ข้าวเปล่า': 'ข้าว',
            'ข้าวกล้อง': 'ข้าวกล้อง',
            'ข้าวต้ม': 'โจ๊ก',
            'ข้าวโจ๊ก': 'โจ๊ก',
            'ข้าวโอ๊ต': 'โอ๊ต',
            'ข้าวไรเบอรี่': 'ข้าวไรเบอรี่',
            'ข้าวหมาก': 'ข้าวหมาก',
            
            # Meat standardization - remove cooking methods
            'หมูทอด': 'หมู',
            'หมูย่าง': 'หมู',
            'หมูปิ้ง': 'หมู',
            'หมูหวาน': 'หมู',
            'หมูกรอบ': 'หมู',
            'หมูเด้ง': 'หมู',
            'หมูสับ': 'หมู',
            'หมูฝอย': 'หมู',
            'หมูหยอง': 'หมู',
            'คอหมู': 'หมู',
            'ขาหมู': 'หมู',
            'ซี่โครงหมู': 'หมู',
            'กระดูกหมู': 'กระดูกหมู',  # Keep as specific item
            
            # Chicken variations
            'ไก่ทอด': 'ไก่',
            'ไก่ย่าง': 'ไก่',
            'ไก่นึ่ง': 'ไก่',
            'เนื้อไก่': 'ไก่',
            'น่องไก่': 'ไก่',
            'ตีนไก่': 'ไก่',
            
            # Fish varieties - keep specific types but remove cooking methods
            'ปลาทอด': 'ปลา',
            'ปลากรอบ': 'ปลา',
            'ปลาทู': 'ปลาทู',
            'ปลากะพง': 'ปลากะพง',
            'ปลาดุก': 'ปลาดุก',
            'ปลาหมึก': 'ปลาหมึก',
            'ปลากราย': 'ปลากราย',
            'ปลานิล': 'ปลานิล',
            'ปลากะเบน': 'ปลากะเบน',
            'ปลาจาระเม็ด': 'ปลาจาระเม็ด',
            'เนื้อปลา': 'ปลา',
            
            # Beverages
            'น้ำเปล่า': 'น้ำ',
            'นม': 'นม',
            'นมวัว': 'นม',
            'นมข้น': 'นมข้น',
            'นมข้าวโอ๊ต': 'นมข้าวโอ๊ต',
            'กาแฟดำ': 'กาแฟ',
            'ชาไทย': 'ชา',
            'ชาเขียว': 'ชาเขียว',
            'โกโก้': 'โกโก้',
            
            # Fruits - remove ripeness indicators
            'มะม่วงสุก': 'มะม่วง',
            'มะม่วงดิบ': 'มะม่วง',
            'มะม่วงเปรี้ยว': 'มะม่วง',
            'กล้วยหอม': 'กล้วย',
            'กล้วยน้ำว้า': 'กล้วย',
            'ส้มโอ': 'ส้ม',
            
            # Vegetables - standardize names
            'ผักกาดขาว': 'ผักกาดขาว',
            'ผักกวางตุ้ง': 'ผักกวางตุ้ง',
            'ผักบุ้ง': 'ผักบุ้ง',
            'ผักโขม': 'ผักโขม',
            'คะน้า': 'คะน้า',
            'กะหล่ำปลี': 'กะหล่ำปลี',
            'บร็อกโคลี่': 'บร็อกโคลี่',
            'บอคโคลี่': 'บร็อกโคลี่',
            'แครอท': 'แครอท',
            'แครรอท': 'แครอท',
            'มะเขือเทศ': 'มะเขือเทศ',
            'แตงกวา': 'แตงกวา',
            'แตงกว่า': 'แตงกวา',
            
            # Eggs - standardize all preparations
            'ไข่ไก่': 'ไข่',
            'ไข่ดาว': 'ไข่',
            'ไข่เจียว': 'ไข่',
            'ไข่ต้ม': 'ไข่',
            'ไข่ตุ๋น': 'ไข่',
            'ไข่ลวก': 'ไข่',
            'ไข่พะโล้': 'ไข่',
            'ไข่คน': 'ไข่',
        }
    
    def step1_basic_cleaning(self, text: str) -> str:
        """Step 1: Enhanced basic text cleaning"""
        if pd.isna(text) or text.strip() == '-':
            return ""
        
        # Remove quotes and content within quotes
        text = re.sub(r'"[^"]*"', ' ', text)
        text = re.sub(r'["\'""]', '', text)
        
        # Remove dish descriptions
        for pattern in self.dish_descriptions:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
        # Remove newlines and special characters
        text = re.sub(r'[\n\r\u200b]+', ' ', text)  # Include zero-width space
        text = re.sub(r'[-–—()[\]{}]', ' ', text)
        text = re.sub(r'[+=#%]', ' ', text)
        
        # Remove measurements (enhanced)
        for pattern in self.measurement_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
        # Remove brand names and commercial terms
        for pattern in self.brand_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
        # Remove non-food items
        for item in self.non_food_items:
            text = re.sub(rf'\b{item}\b', ' ', text, flags=re.IGNORECASE)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def step2_remove_cooking_descriptions(self, text: str) -> str:
        """Step 2: Enhanced removal of cooking methods and descriptions"""
        # Remove complex cooking descriptions and instructions
        cooking_instruction_patterns = [
            r'[^.]*?(สับ|ย่าง|ทอด|ต้ม|นำไป|แล้ว|เผา|อบ|ลวก)[^.]*?[.]?',
            r'[^.]*?(ให้ละเอียด|นำมา|แต่งกลิ่น|ประกอบด้วย)[^.]*?[.]?',
            r'มี[^"]*?(?=\s)',  # Remove "มี..." descriptions
            r'ใส่[^"]*?(?=\s)',  # Remove "ใส่..." descriptions
        ]
        
        for pattern in cooking_instruction_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
        # Remove cooking methods attached to ingredients more aggressively
        for method in self.cooking_methods:
            # Remove method as suffix
            text = re.sub(rf'\b(\w+){method}\b', r'\1', text)
            # Remove method as standalone word
            text = re.sub(rf'\b{method}\b', ' ', text)
        
        # Clean up parenthetical content that's usually descriptions
        text = re.sub(r'\([^)]*\)', ' ', text)
        
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
            if ingredient and len(ingredient) > 1:  # Filter very short items
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
            'step1_cleaned': cleaned,
            'step2_no_cooking': no_cooking,
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
# PROCESSING PIPELINE
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
        
        # Get ingredient frequency
        all_ingredients = []
        for ingredients_list in df['ingredients_final']:
            if ingredients_list:  # Check if list is not empty
                all_ingredients.extend(ingredients_list)
        
        from collections import Counter
        most_common = dict(Counter(all_ingredients).most_common(20))
        
        report = {
            'total_entries': total_entries,
            'valid_entries': valid_entries,
            'success_rate': valid_entries / total_entries * 100 if total_entries > 0 else 0,
            'avg_ingredients_per_entry': avg_ingredients,
            'most_common_ingredients': most_common,
            'total_unique_ingredients': len(set(all_ingredients))
        }
        
        return report


# Export functions for easy use
__all__ = ['ThaiIngredientCleaner', 'LLMPromptGenerator', 'ProcessingPipeline']