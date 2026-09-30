"""Blind question set: written from the source text BEFORE any retrieval was run on these documents."""
import json
from pathlib import Path


def q(i, doc, typ, text, ev, ans):
    return {"id": i, "doc": doc, "type": typ, "q": text, "evidence": ev, "answer": ans}


def U(i, doc, text):
    return q(i, doc, "unanswerable", text, [], None)


B, G, P, F, L, I = ("bert.pdf", "gpt3_fewshot.pdf", "wiki_en_python.pdf", "wiki_en_photosynthesis.pdf",
                    "wiki_it_leonardo.pdf", "wiki_it_fibonacci.pdf")
QA = [
    q("bert01", B, "number", "How many parameters does BERT-BASE have?", ["Total Parameters=110M"], [["110m"], ["110 million"]]),
    q("bert02", B, "number", "How many parameters does BERT-LARGE have?", ["Total Parameters=340M"], [["340m"], ["340 million"]]),
    q("bert03", B, "number", "What vocabulary size does the WordPiece tokenizer used by BERT have?", ["30,000 token vocabulary"], [["30000"], ["30,000"]]),
    q("bert04", B, "number", "What percentage of WordPiece tokens are masked in each sequence in the masked LM task?", ["mask 15% of all WordPiece tokens"], [["15%"], ["15 percent"]]),
    q("bert05", B, "text", "Which corpora was BERT pre-trained on?", ["BooksCorpus (800M words)", "English Wikipedia (2,500M words)"], [["bookscorpus", "wikipedia"]]),
    q("bert06", B, "number", "With what batch size and for how many steps was BERT pre-trained?", ["batch size of 256 sequences", "1,000,000 steps"], [["256", "1,000,000"], ["256", "1000000"], ["256", "1 million"]]),
    q("bert07", B, "number", "What Adam learning rate was used for BERT pre-training?", ["learning rate of 1e-4"], [["1e-4"], ["0.0001"]]),
    q("bert08", B, "number", "What GLUE score did BERT reach?", ["GLUE score to 80.5%"], [["80.5"]]),
    q("bert09", B, "number", "What Test F1 did BERT obtain on SQuAD v2.0?", ["SQuAD v2.0 Test F1 to 83.1"], [["83.1"]]),
    q("bert10", B, "number", "How many layers, what hidden size and how many attention heads does BERT-LARGE have?", ["L=24, H=1024, A=16"], [["24", "1024", "16"]]),
    q("bert11", B, "number", "How long did each BERT pre-training run take?", ["4 days"], [["4 days"], ["four days"]]),
    U("bert12", B, "How much did it cost in US dollars to pre-train BERT-LARGE?"),
    q("gpt01", G, "number", "How many parameters does GPT-3 have?", ["175 billion parameters"], [["175 billion"], ["175b"]]),
    q("gpt02", G, "number", "What is the context window size of GPT-3?", ["nctx = 2048"], [["2048"], ["2,048"]]),
    q("gpt03", G, "number", "On how many tokens in total were all GPT-3 models trained?", ["300 billion tokens"], [["300 billion"]]),
    q("gpt04", G, "number", "How large is the filtered Common Crawl dataset used for GPT-3?", ["570GB after"], [["570"]]),
    q("gpt05", G, "number", "What accuracy does GPT-3 achieve on TriviaQA in the few-shot setting?", ["71.2% in the few-shot"], [["71.2"]]),
    q("gpt06", G, "number", "What Adam beta values were used to train GPT-3?", ["β1 = 0.9, β2 = 0.95"], [["0.9", "0.95"]]),
    q("gpt07", G, "text", "Which attention patterns does GPT-3 use in its transformer layers?", ["alternating dense and locally banded sparse attention patterns"], [["dense", "sparse"], ["dense", "banded"]]),
    q("gpt08", G, "text", "On which GPUs were the GPT-3 models trained?", ["V100 GPU"], [["v100"]]),
    q("gpt09", G, "number", "What accuracy does GPT-3 reach on TriviaQA in the zero-shot setting?", ["64.3% accuracy on TriviaQA in the zero-shot"], [["64.3"]]),
    q("gpt10", G, "table", "What weight does Common Crawl have in the GPT-3 training mix?", ["60% 22% 8% 8% 3%"], [["60%"], ["60 percent"]]),
    q("gpt11", G, "table", "How much training compute in FLOPs did GPT-3 175B need?", ["3.14E+23"], [["3.14e+23"], ["3.14e23"], ["3.14 x 10^23"], ["3.14 × 10^23"], ["3.14 × 10²³"]]),
    U("gpt12", G, "What was the electricity bill in euros for training GPT-3?"),
    q("py01", P, "text", "Who created the Python programming language?", ["Guido van Rossum began working on Python in the late 1980s"], [["guido van rossum"]]),
    q("py02", P, "number", "In which year and as which version was Python first released?", ["first released it in 1991 as Python 0.9.0"], [["1991", "0.9.0"]]),
    q("py03", P, "number", "When was Python 3.0 released?", ["Python 3.0, released in 2008"], [["2008"]]),
    q("py04", P, "number", "On what date was Python 2.0 released?", ["Python 2.0 was released on 16 October 2000"], [["16 october 2000"]]),
    q("py05", P, "text", "Who wrote the Zen of Python?", ["Zen of Python (PEP 20) written by Tim Peters"], [["tim peters"]]),
    q("py06", P, "text", "What title was Guido van Rossum given for leading Python?", ["benevolent dictator for life"], [["benevolent dictator for life"], ["bdfl"]]),
    q("py07", P, "text", "Where does the name Python come from?", ["Monty Python's Flying Circus"], [["monty python"]]),
    q("py08", P, "text", "Which document covers Python coding style?", ["Python coding style is covered in PEP 8"], [["pep 8"]]),
    q("py09", P, "text", "What limits typical Python implementations from scaling across cores?", ["limited by the global interpreter lock"], [["global interpreter lock"], ["gil"]]),
    q("py10", P, "number", "On what date did Guido van Rossum announce he was stepping down as BDFL?", ["12 July 2018"], [["12 july 2018"]]),
    U("py11", P, "What is Python's market share percentage among Fortune 500 companies?"),
    q("ph01", F, "text", "Who discovered photosynthesis in 1779 by showing plants need light?", ["discovered in 1779 by Jan Ingenhousz"], [["ingenhousz"]]),
    q("ph02", F, "number", "What is the average rate of energy captured by global photosynthesis?", ["approximately 130 terawatts"], [["130"]]),
    q("ph03", F, "text", "Which enzyme captures CO2 from the atmosphere in the Calvin cycle?", ["RuBisCO captures CO2"], [["rubisco"]]),
    q("ph04", F, "text", "Who first proposed the general equation for photosynthesis?", ["Cornelis van Niel"], [["van niel"]]),
    q("ph05", F, "text", "What does CAM stand for in photosynthesis?", ["Crassulacean acid metabolism (CAM)"], [["crassulacean acid metabolism"]]),
    q("ph06", F, "text", "Who began the research on photosynthesis in the mid-17th century by measuring the mass of soil and a plant?", ["Jan van Helmont began"], [["van helmont"]]),
    q("ph07", F, "text", "How does the power captured by photosynthesis compare with human power consumption?", ["about eight times the total power consumption"], [["eight times"], ["8 times"]]),
    q("ph08", F, "text", "Which pigment absorbs a photon in the light-dependent reactions?", ["pigment chlorophyll"], [["chlorophyll"]]),
    U("ph09", F, "How many chloroplasts are there in a human leaf cell on average?"),
    q("fib01", I, "number", "In che anno Fibonacci pubblicò il Liber abbaci?", ["Nel 1202"], [["1202"]]),
    q("fib02", I, "text", "Di quale città era il matematico Leonardo Fibonacci?", ["matematico pisano Leonardo Fibonacci"], [["pisa"], ["pisano"]]),
    q("fib03", I, "text", "A quale numero tende il rapporto tra numeri di Fibonacci successivi?", ["sezione aurea"], [["sezione aurea"], ["numero di fidia"]]),
    q("fib04", I, "number", "Chi dimostrò la formula di Binet e in quale anno?", ["Jacques Binet la dimostrò nel 1843"], [["1843"]]),
    q("fib05", I, "text", "Quale problema aritmetico riguardava la crescita di una popolazione nel Liber abbaci?", ["popolazione di conigli"], [["conigli"]]),
    q("fib06", I, "text", "Da chi è stato citato per la prima volta il teorema sui divisori primitivi?", ["citato per la prima volta da Édouard Lucas"], [["lucas"]]),
    U("fib07", I, "Quanto è stato venduto all'asta il manoscritto originale del Liber abbaci?"),
    q("leo01", L, "text", "Dove e quando nacque Leonardo da Vinci?", ["Anchiano, 15 aprile 1452"], [["anchiano", "15 aprile 1452"]]),
    q("leo02", L, "text", "Dove e quando morì Leonardo da Vinci?", ["Amboise, 2 maggio 1519"], [["amboise", "2 maggio 1519"]]),
    q("leo03", L, "text", "Presso quale maestro fu messo a bottega Leonardo?", ["Andrea del Verrocchio"], [["verrocchio"]]),
    q("leo04", L, "text", "Per quale convento Leonardo ricevette nel 1494 la commissione dell'Ultima Cena?", ["Santa Maria delle Grazie"], [["santa maria delle grazie"]]),
    q("leo05", L, "number", "In quali anni fu realizzata l'Ultima Cena?", ["L'Ultima Cena (1494-1498)"], [["1494", "1498"]]),
    q("leo06", L, "text", "A quale duca Leonardo portò un omaggio da Firenze?", ["duca Ludovico il Moro"], [["ludovico il moro"]]),
    q("leo07", L, "text", "Quale re di Francia conobbe Leonardo a Bologna?", ["Francesco I di Francia"], [["francesco i"]]),
    q("leo08", L, "number", "A che anno risale la Dama con l'ermellino?", ["Dama con l'ermellino (1485)"], [["1485"]]),
    q("leo09", L, "number", "Quali anni comprende il periodo fiorentino di Leonardo?", ["Firenze (1468-1482)"], [["1468", "1482"]]),
    U("leo10", L, "Quanto pagò Leonardo di tasse nel 1503 in fiorini?"),
]

if __name__ == "__main__":
    Path(__file__).with_name("qa_blind.json").write_text(json.dumps(QA, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(QA), "questions;", sum(1 for x in QA if x["answer"] is None), "unanswerable")
