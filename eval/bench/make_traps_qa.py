"""Unanswerable questions: in-domain facts that the document does not state, and off-topic questions whose
answer a language model knows from training. Correct behaviour is to say the documents do not contain it.

`absent` = regexes that must NOT occur in the document text (checked by preflight: a trap that the document
actually answers would punish a correct answer)."""
import json
from pathlib import Path


def t(i, doc, text, absent=()):
    return {"id": i, "doc": doc, "type": "unanswerable", "q": text, "evidence": [], "answer": None,
            "absent": list(absent)}


QA = [
    t("tr01", "attention.pdf", "What BLEU score did the Transformer obtain on English-to-Romanian translation?", ["romanian"]),
    t("tr02", "attention.pdf", "What was the training cost in US dollars of the big Transformer model?", ["dollar", r"\$\d"]),
    t("tr03", "attention.pdf", "What dropout rate was used for the English-to-Spanish model?", ["spanish"]),
    t("tr04", "rag_lewis.pdf", "How many TPUs were used to train the RAG models?", [r"\btpus?\b"]),
    t("tr05", "rag_lewis.pdf", "What BLEU score does RAG obtain on WMT machine translation?", ["wmt"]),
    t("tr06", "bert.pdf", "What was the total pre-training cost of BERT in dollars?", ["dollar", r"\$\d"]),
    t("tr07", "bert.pdf", "How many GPUs were used to pre-train BERT-LARGE?", [r"\d+ ?gpus?"]),
    t("tr08", "gpt3_fewshot.pdf", "How many parameters does GPT-4 have?", ["gpt-4"]),
    t("tr09", "gpt3_fewshot.pdf", "Under which open-source license were the GPT-3 weights released?", ["license", "licence"]),
    t("tr10", "resnet.pdf", "What top-1 accuracy did ResNet-152 reach on CIFAR-100?", ["cifar-100"]),
    t("tr11", "gan.pdf", "What FID score did the GAN reach on CIFAR-10?", [r"\bfid\b"]),
    t("tr12", "chain_of_thought.pdf", "How many chain-of-thought exemplars were used with GPT-4?", ["gpt-4"]),
    t("tr13", "wiki_en_photosynthesis.pdf", "How many chloroplasts are there in a single oak leaf?", ["oak"]),
    t("tr14", "wiki_it_leonardo.pdf", "Quanto pesava in grammi la tavola della Gioconda?", [r"\bgrammi\b"]),
    t("tr15", "wiki_it_fibonacci.pdf", "Qual è il quarantesimo numero di Fibonacci?", ["102334155", "102 334 155"]),
    t("tr16", "wiki_it_galileo.pdf", "Quanto pesava in chilogrammi il cannocchiale di Galileo?", ["chilogramm"]),
    t("tr17", "wiki_it_dante.pdf", "Quanti euro vale oggi un'edizione originale del Convivio?", ["edizione originale"]),
    t("tr18", "wiki_en_turing.pdf", "What was Alan Turing's IQ score?", [r"\biq\b"]),
    t("tr19", "wiki_en_turing.pdf", "Which restaurant did Turing visit every Friday in Manchester?", ["restaurant"]),
    t("tr20", "wiki_en_mars.pdf", "What is the floor area in square meters of the first Mars colony building?", ["square meter", "floor area"]),
    t("tr21", "science-exploration-369p.pptx", "How many petabytes of storage does the SEDVME use?", ["petabyte"]),
    t("tr22", "berkshire_2022.pdf", "How many employees does Berkshire Hathaway have in Germany?", ["germany"]),
    t("tr23", "nist_ai_rmf.pdf", "What fine can a company face for violating the AI RMF?", [r"\bfines?\b(?!-)", "penalt"]),
    t("tr24", "handbook-872p.docx", "What is the home address of the Director of the Executive Office for United States Trustees?", ["home address"]),
    # off-topic: a model answers these from its own knowledge
    t("tr25", "*", "What is the capital of Australia?", []),
    t("tr26", "*", "Who won the 2018 FIFA World Cup?", []),
    t("tr27", "*", "What is the boiling point of water at sea level in Celsius?", []),
    t("tr28", "*", "Chi ha scritto I Promessi Sposi?", []),
    t("tr29", "*", "How do I bake sourdough bread at home?", []),
    t("tr30", "*", "Qual è la formula chimica del sale da cucina?", []),
]

if __name__ == "__main__":
    Path(__file__).with_name("qa_traps.json").write_text(json.dumps(QA, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(QA), "traps")
