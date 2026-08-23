import ollama
from pydantic import BaseModel, Field
from typing import List, Optional

# 1. Define single row schema
class DimensionRow(BaseModel):
    dim_number: str = Field(description="Dimension number, e.g., No.1503")
    cc_sc_tag: str = Field(description="Tag, either CC or SC")
    confidence_score: float = Field(description="Clarity score from 0.0 to 1.0")

# 2. Define top-level container schema for all rows
class ExtractedDataSchema(BaseModel):
    dimensions: List[DimensionRow] = Field(description="List of all extracted table rows")

client = ollama.Client(host='http://localhost:11434')

# 3. Request extraction
response = client.chat(
    model='gemma4:12b',
    messages=[{
        'role': 'user',
        'content': 'Extract all rows from the table in the image directly into the required JSON schema without excessive commentary.',
        'images': [r"C:\Users\sanjaytharan.tamilse\Downloads\cc_sc_test_1.jpg"]
    }],
    options={
        'temperature': 0.0,
        'num_predict': 4096,  # Allows enough output tokens
    },
    format=ExtractedDataSchema.model_json_schema()
)

print(response.message.content)