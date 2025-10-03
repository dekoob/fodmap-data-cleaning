import pandas as pd
import json
import sys
import os
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.cleaner import ThaiIngredientCleaner
from scripts.llmcleaner import *
from scripts.utils import *

# # Example usage and testing
# def main():
    
#     # Create test dataframe
#     df = pd.read_excel('data/raw/chatbot_phase2.xlsx', sheet_name='Total')
#     menu_df = df['List menu']
    
#     # Initialize cleaner
#     cleaner = ThaiIngredientCleaner()
    
#     # Process data
#     results_df = cleaner.process_dataframe(df, 'List menu')
    
#     # Display results
#     results_df.to_excel('data/processed/response_cleaned3.xlsx')

# if __name__ == "__main__":
#     main()

async def main():
    """Main processing function with example usage"""

    load_dotenv()
    
    # Configuration
    config = ProcessingConfig(
        api_key=os.getenv('GEMINI_API_KEY', 'your-api-key-here'),
        batch_size=30,  # Smaller batches for better success rate
        max_retries=3,
        rate_limit_delay=2.0,  # Conservative rate limiting
        temperature=0.1,
        max_tokens=3000
    )
    
    if config.api_key == 'your-api-key-here':
        logger.error("Please set your GEMINI_API_KEY environment variable")
        return
    
    # Load your data

    # # # For demo, create sample data
    # sample_data = [
    #     'ปลานิลนึง กะหล่ำปลีลวก น้ำปั่นผลไม้ 1แก้ว',
    #     '"ผัดเผ็ดปลากะเบน"มีเนื้อปลากะเบน"พริกแกง"',
    #     'ไก่ทอด ชีส หัวหอม รามยอนแห้ง หมู ไข่ดาว',
    #     'ข้าวสวย1จาน น้ำพริกปลาทู บะหมี่ ลูกชิ้นปลา',
    #     'วัฟเฟิล ทำจากแป้งสาลี และน้ำตาลเป็นหลัก'
    # ] * 5  # Duplicate for demo

    # # Create test dataframe
    # df = pd.DataFrame({'ingredients': sample_data})
    
    df = pd.read_excel('data/raw/chatbot_phase2.xlsx', sheet_name='Total')
    df = df.rename(columns={'List menu': 'ingredients'})
    
    logger.info(f"Loaded {len(df)} entries for processing")
    
    # Set up checkpoint
    checkpoint_path = 'processing_checkpoint.pkl'
    
    # Check for existing checkpoint
    checkpoint_df = load_checkpoint(checkpoint_path)
    if checkpoint_df is not None:
        df = checkpoint_df
        logger.info("Resuming from checkpoint")
    
    # Process data
    async with GeminiIngredientCleaner(config) as cleaner:
        try:
            results_df = await cleaner.process_dataframe(df, 'ingredients')
            
            # Save final results
            output_path = 'data/processed/version2/cleaned_ingredients.csv'
            results_df.to_csv(output_path, encoding="utf-8-sig", index=False)
            logger.info(f"Results saved to {output_path}")
            
            # Generate and save report
            report = cleaner.generate_final_report()
            
            with open('data/processed/version1/processing_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # Display summary
            print("\n=== PROCESSING COMPLETE ===")
            print(f"Total Entries: {report['processing_summary']['total_entries']}")
            print(f"Success Rate: {report['processing_summary']['success_rate']:.1f}%")
            print(f"Total Time: {report['performance']['total_time_seconds']:.1f} seconds")
            print(f"Rate: {report['performance']['entries_per_second']:.1f} entries/second")
            print(f"API Calls: {report['performance']['api_calls_made']}")
            
            # Show sample results
            print("\n=== SAMPLE RESULTS ===")
            for i in range(min(5, len(results_df))):
                row = results_df.iloc[i]
                print(f"\nOriginal: {row['ingredients'][:60]}...")
                print(f"Cleaned:  {row['gemini_cleaned']}")
                print(f"Status:   {row['processing_status']}")
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted. Saving checkpoint...")
            save_checkpoint(results_df, checkpoint_path)
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise
    
    


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())