from openai import OpenAI
from sentence_transformers import SentenceTransformer
import array
import sys
import os
package_path = "/Users/jayanthan/Learnings/Python"
if package_path not in sys.path:
    sys.path.append(package_path)
from python_my_packages.AI.db_processor.db_connection import *

def query_similar_texts(question, top_k=5):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
   # model = SentenceTransformer("all-MiniLM-L6-v2")
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    query_vec = array.array("f", list(model.encode(question)))
   # print (query_vec)

    connection, cursor = get_db_connection(
        'wksp_jayorch', 'Welcome#2024', '/Users/jayanthan/Learnings/AI/oci_alert_logagent/wallet', 'jaytest_public_high')

    sql = f"""
        SELECT rid,to_char(text), filename, department,element_type,
               VECTOR_DISTANCE(embedding, :query_vec, COSINE) AS distance
        FROM pdf_elements
        ORDER BY distance
        FETCH FIRST {top_k} ROWS ONLY
    """

    cursor.execute(sql, {"query_vec": query_vec})
    results = cursor.fetchall()
    # return results

    extended_results = []
    neighbor_rows = 1  # number of rows before and after

    for row in results:
        rid, text, filename, department, element_type, distance = row

        # Fetch neighbors using RID
        neighbor_rids = [nrid for nrid in range(
            rid - neighbor_rows, rid + neighbor_rows + 1) if nrid > 0]

        neighbor_sql = f"""
            SELECT rid, to_char(text) AS text, filename, department, element_type
            FROM pdf_elements
            WHERE rid IN ({','.join(str(nrid) for nrid in neighbor_rids)})
            ORDER BY rid
        """
        cursor.execute(neighbor_sql)
        neighbors = cursor.fetchall()
        extended_results.extend(neighbors)

    return extended_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_simlar_text.py 'your question here'")
        sys.exit(1)

    question = sys.argv[1]
    results = query_similar_texts(question)

    print("\nTop matches:")
    top_texts = []

    for row in results:
        # adjust if SELECT order is different
        text, filename, department, distance, element_type = row
        top_texts.append(
            f"File: {filename}, pdf extracting element_type: {element_type}\n,Dept: {department}\n{text}")

    context = "\n\n---\n\n".join(top_texts)
    # print(context)

    client = OpenAI(
        api_key=os.getenv("GAPI_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

    # Combine context and question into a prompt
    prompt = f"""

    You are a helpful assistant. Use the following extracted content from press releases to answer the question. The data comes from PDFs that were processed using Python’s unstructured library, vectorized, and stored in an Oracle database. I have provided the closest matching vectors in the format: File name, Department, Extracted Text, and Element Type (such as text, narrative text, table, or heading. In the end , Give the source from which you construct this output).  


    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    print(prompt)
    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt
    )

    # Extract the generated answer
    answer = response.output_text
    print("LLM Answer:\n", answer)
