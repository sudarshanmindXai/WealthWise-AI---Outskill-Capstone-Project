import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Initialize Engines once (Global to avoid reloading)
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def clean_and_redact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scans the DataFrame for PII (Names, Phone Numbers) and replaces them with [REDACTED].
    Focuses mainly on the 'Description' column as that's where messy PII lives.
    """
    
    # We only want to scrub object (string) columns, specifically Description
    # But let's look at all string columns just in case
    string_cols = df.select_dtypes(include=['object']).columns
    
    for col in string_cols:
        # Skip Date column to prevent false positives
        if col.lower() == 'date':
            continue
            
        df[col] = df[col].apply(lambda x: redact_text(str(x)) if pd.notnull(x) else x)
        
    return df

def redact_text(text: str) -> str:
    """
    Helper function to redact a single string.
    """
    if not text:
        return ""
        
    # 1. Analyze
    results = analyzer.analyze(text=text, entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "IN_PAN"], language='en')
    
    # 2. Anonymize
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={
            "PERSON": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
             # Add IN_PAN if your Presidio setup supports it or use custom regex
        }
    )
    
    return anonymized_result.text
