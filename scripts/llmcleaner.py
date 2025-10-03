#!/usr/bin/env python3
"""
Gemini 2.0 Thai Ingredient Cleaning Script
=========================================

This script processes large datasets of Thai ingredient data using Google's Gemini 2.0 API
with batch processing, rate limiting, and robust error handling.
"""

import pandas as pd
import time
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import asyncio
import aiohttp
from datetime import datetime
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingredient_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for processing parameters"""
    api_key: str
    batch_size: int = 20  # Process 20 entries per batch (increased)
    max_retries: int = 3
    retry_delay: float = 2.0  # seconds
    rate_limit_delay: float = 1.5  # Increased delay for larger batches
    max_tokens: int = 2000  # Increased for larger batches
    temperature: float = 0.1  # Low temperature for consistent results
    model_name: str = "gemini-2.0-flash-lite"

class GeminiIngredientCleaner:
    """Thai ingredient cleaner using Gemini 2.0 API"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model_name}:generateContent"
        self.session = None
        
        # Define system instruction once - will be reused for all API calls
        self.system_instruction = """You are a Thai cuisine expert. Clean these ingredient lists using this 4-stage process:

STAGE 1: PRE-PROCESSING & NORMALIZATION
- Remove quantities/measurements (1ถ้วย, 5ชิ้น, ครึ่งจาน, etc.)
- Remove brands (KFC, 7-11, Ensure, นูเทลล่า, etc.)
- Remove punctuation and special characters
- Standardize spacing (multiple spaces → single space)
- Remove cooking modifiers: ทอด, ย่าง, ต้ม, นึ่ง, กรอบ, เปื่อย, หวาน, เปรี้ยว, เผ็ด, สด, แห้ง, ตุ๋น
- Remove non-food related items

STAGE 2: STANDARDIZING INGREDIENTS
- Apply synonym mapping, for example:
  * ข้าวสวย → ข้าว
  * หมูทอด/หมูย่าง/หมูปิ้ง → หมู
  * ปลาทอด/ปลากรอบ → ปลา
  * น้ำเปล่า → น้ำ
  * มะม่วงสุก → มะม่วง
- Fix common misspellings:
  * ก้วยเตี๋ยว → ก๋วยเตี๋ยว
  * แครรอท → แครอท
  * บอคโคลี่ → บร็อกโคลี่

STAGE 3: DECONSTRUCTING DISHES
Break down complex dishes or complex ingredients into core ingredients, for example:
- แกงเขียวหวาน → กะทิ, เครื่องแกงเขียวหวาน, เนื้อ/ไก่
- เครื่องแกงเขียวหวาน → ตะไคร้, ข่า, พริก, มะกรูด, ผักชี, กะปิ, ยี่หร่า, หอมแดง, กระเทียม
- ส้มตำ → มะละกอ, มะเขือเทศ, ถั่วฝักยาว, กุ้งแห้ง
- ลาบ → เนื้อ/หมู, สะระแหน่, หอมแดง, น้ำมะนาว
- น้ำพริกกะปิ → กะปิ, พริก, กระเทียม, น้ำมะนาว

STAGE 4: ADD FODMAP DOMAIN KNOWLEDGE (IF APPLICABLE)
When certain ingredients are present, ADD their FODMAP category, for example:
- หอมแดง, กระเทียม, หอมใหญ่, กุยช่าย → Fructan_Rich_Alliums
- มะม่วง, แอปเปิ้ล, น้ำผึ้ง, ลิ้นจี่ → Excess_Fructose_Rich_Foods  
- นม, นมข้น, โยเกิร์ต, ชีส → Lactose_Rich_Foods
- ถั่ว, ถั่วเขียว, ถั่วแดง, ถั่วลันเตา → GOS_Rich_Foods
- เห็ด, กะหล่ำดอก, พลัม, ท้อ → Polyol_Rich_Foods

Keep other ingredients as individual items. Mix both ingredient names and FODMAP categories in the same output.

OUTPUT FORMAT - Return each entry on a new line with the number:
1. ingredient1, Fructan_Rich_Alliums, ingredient3
2. ingredient1, Lactose_Rich_Foods
etc.


Apply ALL 4 stages. Extract ONLY actual food ingredients + FODMAP categories, comma-separated."""
        
        # Stats tracking
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'api_calls': 0,
            'start_time': None,
            'errors': []
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def create_cleaning_prompt(self, ingredients_batch: List[str]) -> str:
        """Create minimal prompt - just pass ingredient text, system instruction does the work"""
        
        # Format ingredients with numbers for easy parsing
        ingredients_text = ""
        for i, ing in enumerate(ingredients_batch, 1):
            ingredients_text += f"{i}. {ing}\n"
        
        # Just the ingredient data - system instruction handles all the rules
        return ingredients_text.strip()
    
    async def call_gemini_api(self, prompt: str) -> Optional[str]:
        """Make API call to Gemini 2.0 with system instructions"""
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        # Use system instruction + shorter user prompt for efficiency
        payload = {
            'system_instruction': {
                'parts': [{'text': self.system_instruction}]
            },
            'contents': [{
                'parts': [{'text': prompt}]
            }],
            'generationConfig': {
                'temperature': self.config.temperature,
                'maxOutputTokens': self.config.max_tokens,
                'topP': 0.8,
                'topK': 40
            }
        }
        
        url = f"{self.api_url}?key={self.config.api_key}"
        
        for attempt in range(self.config.max_retries):
            try:
                self.stats['api_calls'] += 1
                
                async with self.session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if 'candidates' in result and len(result['candidates']) > 0:
                            content = result['candidates'][0]['content']['parts'][0]['text']
                            return content.strip()
                        else:
                            logger.warning(f"No candidates in response: {result}")
                            
                    elif response.status == 429:  # Rate limit
                        wait_time = (attempt + 1) * self.config.retry_delay * 2
                        logger.warning(f"Rate limited. Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"API error {response.status}: {error_text}")
                        
            except Exception as e:
                logger.error(f"API call error (attempt {attempt + 1}): {str(e)}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        return None
    
    def parse_gemini_response(self, response: str, batch_size: int) -> List[List[str]]:
        """Parse Gemini response into ingredient lists"""
        
        results = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove numbering (1. 2. etc.)
            if '. ' in line:
                line = line.split('. ', 1)[1] if len(line.split('. ', 1)) > 1 else line
            
            # Split by comma and clean
            if line:
                ingredients = [ing.strip() for ing in line.split(',') if ing.strip()]
                results.append(ingredients)
        
        # Ensure we have the right number of results
        while len(results) < batch_size:
            results.append([])
            
        return results[:batch_size]
    
    async def process_batch(self, batch: List[str]) -> List[List[str]]:
        """Process a batch of ingredient entries"""
        
        try:
            prompt = self.create_cleaning_prompt(batch)  # Updated function name
            response = await self.call_gemini_api(prompt)
            
            if response:
                parsed_results = self.parse_gemini_response(response, len(batch))
                self.stats['successful'] += len(batch)
                return parsed_results
            else:
                logger.error(f"Failed to process batch of {len(batch)} entries")
                self.stats['failed'] += len(batch)
                return [[] for _ in batch]  # Return empty lists for failed batch
                
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            self.stats['failed'] += len(batch)
            self.stats['errors'].append(str(e))
            return [[] for _ in batch]
    
    async def process_dataframe(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Process entire dataframe with progress tracking"""
        
        self.stats['start_time'] = datetime.now()
        total_entries = len(df)
        
        logger.info(f"Starting processing of {total_entries} entries...")
        
        # Prepare results dataframe
        results_df = df.copy()
        results_df['gemini_cleaned'] = None
        results_df['gemini_count'] = 0
        results_df['processing_status'] = 'pending'
        
        # Process in batches
        for start_idx in range(0, total_entries, self.config.batch_size):
            end_idx = min(start_idx + self.config.batch_size, total_entries)
            batch_indices = list(range(start_idx, end_idx))
            
            # Get batch data - handle both DataFrame and Series
            if isinstance(df, pd.Series):
                batch_data = [df.iloc[i] for i in batch_indices]
            else:
                batch_data = [df.iloc[i][column_name] for i in batch_indices]
            
            logger.info(f"Processing batch {start_idx//self.config.batch_size + 1}/{(total_entries-1)//self.config.batch_size + 1} (entries {start_idx}-{end_idx-1})")
            
            # Process batch
            batch_results = await self.process_batch(batch_data)
            
            # Store results
            for i, result in enumerate(batch_results):
                idx = batch_indices[i]
                results_df.at[idx, 'gemini_cleaned'] = result
                results_df.at[idx, 'gemini_count'] = len(result) if result else 0
                results_df.at[idx, 'processing_status'] = 'success' if result else 'failed'
            
            self.stats['total_processed'] = end_idx
            
            # Progress update
            progress = (end_idx / total_entries) * 100
            elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
            rate = end_idx / elapsed if elapsed > 0 else 0
            
            logger.info(f"Progress: {progress:.1f}% ({end_idx}/{total_entries}) - Rate: {rate:.1f} entries/sec")
            
            # Rate limiting between batches
            await asyncio.sleep(self.config.rate_limit_delay)
        
        return results_df
    
    def generate_final_report(self) -> Dict:
        """Generate comprehensive processing report"""
        
        end_time = datetime.now()
        total_time = (end_time - self.stats['start_time']).total_seconds()
        
        return {
            'processing_summary': {
                'total_entries': self.stats['total_processed'],
                'successful': self.stats['successful'],
                'failed': self.stats['failed'],
                'success_rate': (self.stats['successful'] / self.stats['total_processed']) * 100 if self.stats['total_processed'] > 0 else 0
            },
            'performance': {
                'total_time_seconds': total_time,
                'average_time_per_entry': total_time / self.stats['total_processed'] if self.stats['total_processed'] > 0 else 0,
                'api_calls_made': self.stats['api_calls'],
                'entries_per_second': self.stats['total_processed'] / total_time if total_time > 0 else 0
            },
            'errors': {
                'error_count': len(self.stats['errors']),
                'unique_errors': list(set(self.stats['errors']))
            }
        }

def save_checkpoint(df: pd.DataFrame, checkpoint_path: str):
    """Save processing checkpoint"""
    df.to_pickle(checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path: str) -> Optional[pd.DataFrame]:
    """Load processing checkpoint"""
    if os.path.exists(checkpoint_path):
        df = pd.read_pickle(checkpoint_path)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return df
    return None