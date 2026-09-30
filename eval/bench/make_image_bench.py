"""Image benchmark: facts that exist ONLY inside images, in a separate index (docs_img/).

* scanned pages: real pages (BERT, GAN, Leonardo/Galileo Wikipedia) rasterised with noise, tilt and JPEG
  compression, saved as image-only PDFs. Native text extraction returns nothing: only OCR can answer.
* charts / tables / an invoice / a diagram: drawn with known values, so the ground truth is certain.
Run once; needs docs_blind*/ from the main benchmark.
"""
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import pypdfium2 as pdfium                          # noqa: E402
from PIL import Image, ImageDraw, ImageFont         # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "docs_img"
SCANS = {"scan_bert.pdf": ("docs_blind/bert.pdf", [1, 3, 4, 5]),
         "scan_leonardo_it.pdf": ("docs_blind/wiki_it_leonardo.pdf", [1, 3]),
         "scan_galileo_it.pdf": ("docs_blind2/wiki_it_galileo.pdf", [1]),
         "scan_gan.pdf": ("docs_blind2/gan.pdf", [3, 4])}


def scan_like(img: Image.Image, rng: random.Random) -> Image.Image:
    img = img.rotate(rng.uniform(-0.8, 0.8), expand=True, fillcolor="white", resample=Image.BICUBIC)
    arr = np.asarray(img).astype("float32") + np.random.default_rng(rng.randint(0, 9999)).normal(0, 9, (img.height, img.width, 3))
    return Image.fromarray(arr.clip(0, 255).astype("uint8"))


def make_scans() -> None:
    rng = random.Random(3)
    for name, (src, pages) in SCANS.items():
        pdf = pdfium.PdfDocument(str(HERE / src))
        images = [scan_like(pdf[p - 1].render(scale=150 / 72).to_pil().convert("RGB"), rng) for p in pages]
        jpegs = []
        for i, im in enumerate(images):
            tmp = OUT / f"_tmp{i}.jpg"
            im.save(tmp, "JPEG", quality=55)
            jpegs.append(Image.open(tmp).convert("RGB"))
        jpegs[0].save(OUT / name, save_all=True, append_images=jpegs[1:], resolution=150)
        for i in range(len(images)):
            (OUT / f"_tmp{i}.jpg").unlink()


def make_charts() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Q1", "Q2", "Q3", "Q4"], [12.4, 15.8, 19.3, 22.1], color="#3b82f6")
    ax.bar_label(bars, fmt="%.1f")
    ax.set_title("Quarterly revenue 2024 (EUR million)")
    ax.set_ylabel("EUR million")
    fig.savefig(OUT / "chart_revenue_2024.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([41, 27, 19, 13], labels=["Alpha", "Beta", "Gamma", "Others"], autopct="%d%%", startangle=90)
    ax.set_title("Smartphone market share, 2024")
    fig.savefig(OUT / "chart_market_share.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    countries, counts = ["Italy", "Germany", "France", "Spain"], [320, 210, 145, 98]
    bars = ax.barh(countries[::-1], counts[::-1], color="#10b981")
    ax.bar_label(bars)
    ax.set_title("Employees by country")
    fig.savefig(OUT / "chart_employees.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    temps = [3.1, 4.5, 8.2, 12.6, 17.0, 21.9, 26.4, 25.8, 20.1, 14.3, 8.0, 4.2]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(months, temps, marker="o")
    ax.annotate("Peak: 26.4 °C (July)", xy=(6, 26.4), xytext=(1.2, 24), arrowprops={"arrowstyle": "->"})
    ax.set_title("Average monthly temperature, Verona")
    ax.set_ylabel("°C")
    fig.savefig(OUT / "chart_temperature.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axis("off")
    rows = [["Boiler B1", "12", "450", "2000"], ["Compressor C2", "40", "315", "1500"],
            ["Turbine T3", "85", "1200", "4000"], ["Pump P4", "25", "90", "800"]]
    table = ax.table(cellText=rows, colLabels=["Unit", "Max pressure (bar)", "Rated power (kW)", "Service (h)"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)
    fig.savefig(OUT / "table_specs.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.axis("off")
    labels = ["Ingest", "Chunk", "Embed", "Index"]
    for i, label in enumerate(labels):
        ax.text(i * 2.4 + 0.6, 0.5, label, ha="center", va="center", fontsize=14,
                bbox={"boxstyle": "round,pad=0.5", "fc": "#e0e7ff", "ec": "#4338ca"})
        if i:
            ax.annotate("", xy=(i * 2.4 - 0.25, 0.5), xytext=(i * 2.4 - 1.05, 0.5), arrowprops={"arrowstyle": "->", "lw": 2})
    ax.set_xlim(-0.4, 9.6)
    ax.set_ylim(0, 1)
    ax.set_title("Document pipeline")
    fig.savefig(OUT / "diagram_pipeline.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def make_invoice() -> None:
    img = Image.new("RGB", (900, 520), "white")
    d = ImageDraw.Draw(img)
    try:
        f, fb = ImageFont.truetype("arial.ttf", 28), ImageFont.truetype("arialbd.ttf", 34)
    except OSError:
        f = fb = ImageFont.load_default(28)
    d.text((40, 30), "INVOICE 2024-0193", font=fb, fill="black")
    lines = ["Customer: Rossi Impianti Srl", "Date: 2 March 2025", "Due date: 15 March 2025",
             "Boiler service (2 x 480.00 EUR)   960.00", "Spare parts                       324.50",
             "TOTAL  1,284.50 EUR"]
    for i, line in enumerate(lines):
        d.text((40, 100 + i * 60), line, font=f, fill="black")
    scan_like(img, random.Random(1)).save(OUT / "invoice_scan.png")


def qa() -> list[dict]:
    def q(i, doc, typ, text, ans):
        return {"id": i, "doc": doc, "type": typ, "q": text, "evidence": [], "answer": ans}
    def u(i, text):
        return q(i, "*", "unanswerable", text, None)
    S = [
        q("sb1", "scan_bert.pdf", "scanned", "How many parameters does BERT-BASE have?", [["110m"], ["110 million"]]),
        q("sb2", "scan_bert.pdf", "scanned", "What vocabulary size does the WordPiece tokenizer used by BERT have?", [["30,000"], ["30000"]]),
        q("sb3", "scan_bert.pdf", "scanned", "What percentage of tokens does BERT mask in the masked LM task?", [["15%"], ["15 percent"]]),
        q("sb4", "scan_bert.pdf", "scanned", "Which corpora was BERT pre-trained on?", [["bookscorpus", "wikipedia"]]),
        q("sb5", "scan_bert.pdf", "scanned", "What GLUE score did BERT reach?", [["80.5"]]),
        q("sl1", "scan_leonardo_it.pdf", "scanned", "Dove e quando nacque Leonardo da Vinci?", [["anchiano", "15 aprile 1452"]]),
        q("sl2", "scan_leonardo_it.pdf", "scanned", "Presso quale maestro fu messo a bottega Leonardo?", [["verrocchio"]]),
        q("sg1", "scan_galileo_it.pdf", "scanned", "Dove e quando nacque Galileo Galilei?", [["pisa", "15 febbraio 1564"]]),
        q("sg2", "scan_galileo_it.pdf", "scanned", "Quando fu costretto Galileo all'abiura?", [["22 giugno 1633"]]),
        q("sn1", "scan_gan.pdf", "scanned", "What kind of game do the two networks play in the adversarial framework?", [["minimax"]]),
        q("sn2", "scan_gan.pdf", "scanned", "How does the training procedure alternate between the discriminator and the generator?", [["k steps", "one step"]]),
        q("fg1", "chart_revenue_2024.png", "figure", "What was the revenue in Q3 2024 according to the revenue chart?", [["19.3"]]),
        q("fg2", "chart_revenue_2024.png", "figure", "Which quarter had the highest revenue in 2024?", [["q4"], ["fourth"]]),
        q("fg3", "chart_market_share.png", "figure", "What market share does Beta have?", [["27%"]]),
        q("fg4", "chart_market_share.png", "figure", "Which company has the largest smartphone market share?", [["alpha"]]),
        q("fg5", "table_specs.png", "figure", "What is the rated power of the Compressor C2?", [["315"]]),
        q("fg6", "table_specs.png", "figure", "What is the maximum pressure of the Turbine T3?", [["85"]]),
        q("fg7", "invoice_scan.png", "figure", "What is the total amount of invoice 2024-0193?", [["1284.50"], ["1,284.50"]]),
        q("fg8", "invoice_scan.png", "figure", "What is the due date of the invoice?", [["15 march 2025"]]),
        q("fg9", "diagram_pipeline.png", "figure", "Which step comes right after Chunk in the document pipeline diagram?", [["embed"]]),
        q("fg10", "chart_employees.png", "figure", "How many employees are there in Germany?", [["210"]]),
        q("fg11", "chart_employees.png", "figure", "Which country has the fewest employees?", [["spain"]]),
        q("fg12", "chart_temperature.png", "figure", "What was the peak temperature and in which month?", [["26.4", "july"]]),
        u("ut1", "What was the revenue in Q5 2024?"),
        u("ut2", "What market share does Delta have?"),
        u("ut3", "What is the rated power of the Compressor C9?"),
        u("ut4", "What is the VAT number on invoice 2024-0193?"),
        u("ut5", "How many employees are there in Japan?"),
        u("ut6", "How many GPUs were used to pre-train BERT?"),
        u("ut7", "Quanti figli ebbe Leonardo da Vinci?"),
        u("ut8", "What is the capital of Australia?"),
    ]
    return S


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    make_scans()
    make_charts()
    make_invoice()
    (HERE / "qa_img.json").write_text(json.dumps(qa(), indent=1, ensure_ascii=False), encoding="utf-8")
    print(sorted(p.name for p in OUT.iterdir()))
