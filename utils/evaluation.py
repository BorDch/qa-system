# === evalution.py ===
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score as bert_score

rouge = Rouge()
sim_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def rerank_candidates(prompts, outputs, tokenizer, ref_answers, num_return_sequences=3):
    best_answers = []
    for i in range(len(prompts)):
        prompt_text = prompts[i]
        ref = ref_answers[i]
        ref_emb = sim_model.encode(ref, convert_to_tensor=True)

        candidates = []
        for k in range(num_return_sequences):
            idx = i * num_return_sequences + k
            full_output = tokenizer.decode(outputs[idx], skip_special_tokens=True)
            gen_answer = full_output[len(prompt_text):].strip()
            candidates.append(gen_answer)

        best_answer = max(candidates, key=lambda a: util.pytorch_cos_sim(sim_model.encode(a, convert_to_tensor=True), ref_emb).item())
        best_answers.append((best_answer, ref))
    return best_answers

def evaluate_answers(best_pairs):
    sim_scores, bleu_scores, rouge_scores, bert_f1s = [], [], [], []
    smooth_fn = SmoothingFunction().method1

    for best, ref in best_pairs:
        ref_emb = sim_model.encode(ref, convert_to_tensor=True)
        best_emb = sim_model.encode(best, convert_to_tensor=True)

        sim_scores.append(util.pytorch_cos_sim(ref_emb, best_emb).item())
        bleu_scores.append(sentence_bleu([ref.split()], best.split(), smoothing_function=smooth_fn))
        rouge_score = rouge.get_scores(best, ref)[0]['rouge-l']['f']
        rouge_scores.append(rouge_score)
        _, _, f1 = bert_score([best], [ref], lang="ru")
        bert_f1s.append(f1.item())

    return {
        "Semantic Similarity": sum(sim_scores)/len(sim_scores),
        "BLEU": sum(bleu_scores)/len(bleu_scores),
        "ROUGE-L": sum(rouge_scores)/len(rouge_scores),
        "BERTScore-F1": sum(bert_f1s)/len(bert_f1s)
    }