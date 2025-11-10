import pandas as pd
import sys
import os
import array
from sentence_transformers import SentenceTransformer
package_path = "/Users/jayanthan/Learnings/Python"
if package_path not in sys.path:
    sys.path.append(package_path)
from python_my_packages.AI.pdf_processor.extraction import *
from python_my_packages.AI.pdf_processor.flattening import *
from python_my_packages.AI.pdf_processor.cleaning import *
from python_my_packages.AI.db_processor.db_connection import *
#from python_my_packages.AI.pdf_processor.db_config import get_db_connection
import sys
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("intfloat/multilingual-e5-base")
#def generate_embedding(text):
 #   return model.encode(text).tolist()

def generate_embedding(text):
    return model.encode(text)

def process_pdf(pdf_path):
    if has_table(pdf_path):
        print("This PDF contains tables.so going with hi_res extraction.")
        elements = extract_pdf_elements(pdf_path,'hi_res')
        grouped_chunks = group_pdf_elements(elements)
    else:
        print("No tables detected.so going with Auto extraction.")
        elements = extract_pdf_elements(pdf_path,'auto')
        grouped_chunks = group_pdf_elements(elements)

    #elements = extract_pdf_elements(pdf_path)
    print (grouped_chunks)
    df = flatten_elements_grouped(grouped_chunks, pdf_name=os.path.basename(pdf_path), department="Finance")
    #print(df.to_string(index=False))
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 500)

    df.to_csv("flattened_pdf.csv", index=False)


    connection, cursor = get_db_connection('wksp_jayorch','Welcome#2024','/Users/jayanthan/Learnings/AI/oci_alert_logagent/wallet','jaytest_public_high')



    upsert_sql = """
    MERGE INTO pdf_elements t
    USING (SELECT :filename AS filename, :row_id AS row_id FROM dual) s
    ON (t.filename = s.filename AND t.row_id = s.row_id)
    WHEN MATCHED THEN
    UPDATE SET element_type = :element_type,
                department = :department,
                text = :text,
                EMBEDDING = :embedding,
                ingestion_time = :ingestion_time
    WHEN NOT MATCHED THEN
    INSERT (row_id, filename, page_num, element_type, department, text, ingestion_time,EMBEDDING)
    VALUES (:row_id, :filename, :page_num, :element_type, :department, :text, :ingestion_time,:embedding)
    """

    try:
        for _, row in df.iterrows(): 
            embedding = list(model.encode(row["text"]))
            vec = array.array("f", embedding)
            cursor.execute(upsert_sql, {
                "row_id": row["row_id"],
                "filename": row["filename"],
                "page_num": row["page_num"],
                "element_type": row["element_type"],
                "department": row["department"],
                "text": row["text"],
                "ingestion_time": row["ingestion_time"],
                "embedding": vec
            })
        connection.commit()
        print("Data inserted/upserted successfully!")
    except oracledb.DatabaseError as e:
        print(f"Database error occurred: {e}")
        connection.rollback()
    finally:
        cursor.close()
        connection.close()
    print("Database connection closed.")




if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_etl_process.py <path_to_pdf_or_folder>")
    else:
        path = sys.argv[1]
        if os.path.isdir(path):
            # Loop through all PDF files in the folder
            for filename in os.listdir(path):
                if filename.lower().endswith(".pdf"):
                    pdf_file_path = os.path.join(path, filename)
                    process_pdf(pdf_file_path)
        elif os.path.isfile(path) and path.lower().endswith(".pdf"):
            process_pdf(path)
        else:
            print("No PDF files found at the specified path.")
