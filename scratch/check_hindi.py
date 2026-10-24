import json

with open('results/rag/raw_ocr_results.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)
with open('results/rag/corrected_ocr_results.json', 'r', encoding='utf-8') as f:
    corr = json.load(f)
with open('results/rag/ground_truth_results.json', 'r', encoding='utf-8') as f:
    gt = json.load(f)

# Also check query dataset to get question text and language
with open('data/evaluation/queries.json', 'r', encoding='utf-8') as f:
    queries = json.load(f)
query_meta = {q['query_id']: q for q in queries}

raw_dict = {item['question_id']: item for item in raw['qa_results']}
corr_dict = {item['question_id']: item for item in corr['qa_results']}
gt_dict = {item['question_id']: item for item in gt['qa_results']}

print("--- HINDI QUERIES WITH DIFFERENCE ---")
for qid, item in corr_dict.items():
    meta = query_meta.get(qid, {})
    lang = meta.get('language', '')
    if lang == 'hi':
        r_item = raw_dict[qid]
        g_item = gt_dict[qid]
        em_corr = item.get('exact_match', 0)
        em_raw = r_item.get('exact_match', 0)
        f1_corr = item.get('f1_score', 0)
        f1_raw = r_item.get('f1_score', 0)
        if em_corr != em_raw or f1_corr != f1_raw:
            print(f"Query ID: {qid}")
            print(f"Question: {meta.get('question')}")
            print(f"Expected Answer: {item['expected_answer']}")
            print(f"Raw OCR Answer:  '{r_item.get('generated_answer')}' (EM={em_raw}, F1={f1_raw:.4f})")
            print(f"Corr OCR Answer: '{item.get('generated_answer')}' (EM={em_corr}, F1={f1_corr:.4f})")
            print(f"GT OCR Answer:   '{g_item.get('generated_answer')}'")
            print("-" * 50)

print("\n--- ALL HINDI QUERIES STATS ---")
raw_hi_em = [r_item.get('exact_match', 0) for qid, r_item in raw_dict.items() if query_meta.get(qid, {}).get('language') == 'hi']
corr_hi_em = [item.get('exact_match', 0) for qid, item in corr_dict.items() if query_meta.get(qid, {}).get('language') == 'hi']
gt_hi_em = [g_item.get('exact_match', 0) for qid, g_item in gt_dict.items() if query_meta.get(qid, {}).get('language') == 'hi']

print(f"Ground Truth Hindi EM: {sum(gt_hi_em)} / {len(gt_hi_em)} = {sum(gt_hi_em)/len(gt_hi_em):.4f}")
print(f"Raw OCR Hindi EM: {sum(raw_hi_em)} / {len(raw_hi_em)} = {sum(raw_hi_em)/len(raw_hi_em):.4f}")
print(f"Corrected OCR Hindi EM: {sum(corr_hi_em)} / {len(corr_hi_em)} = {sum(corr_hi_em)/len(corr_hi_em):.4f}")
