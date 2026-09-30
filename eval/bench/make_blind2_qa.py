"""Blind set #2: new documents, questions written from the source text BEFORE running retrieval on them.
No parameter of the system may be tuned on this set; it exists to report an unbiased number."""
import json
from pathlib import Path


def q(i, doc, typ, text, ev, ans):
    return {"id": i, "doc": doc, "type": typ, "q": text, "evidence": ev, "answer": ans}


def U(i, doc, text):
    return q(i, doc, "unanswerable", text, [], None)


R, GA, C, GA2, DA, T, M = ("resnet.pdf", "gan.pdf", "chain_of_thought.pdf", "wiki_it_galileo.pdf",
                           "wiki_it_dante.pdf", "wiki_en_turing.pdf", "wiki_en_mars.pdf")
QA = [
    q("res01", R, "number", "What is the maximum depth of the residual nets evaluated on ImageNet?", ["up to 152 layers"], [["152"]]),
    q("res02", R, "number", "What top-5 error did the ensemble of residual nets achieve on the ImageNet test set?", ["achieves 3.57% error"], [["3.57"]]),
    q("res03", R, "text", "Which competition and year did the ResNet ensemble win 1st place in?", ["1st place on the ILSVRC 2015 classiﬁcation task"], [["ilsvrc 2015"]]),
    q("res04", R, "number", "What mini-batch size is used to train the residual networks with SGD?", ["mini-batch size of 256"], [["256"]]),
    q("res05", R, "number", "What weight decay and momentum are used when training ResNets?", ["weight decay of 0.0001", "momentum of 0.9"], [["0.0001", "0.9"]]),
    q("res06", R, "number", "What starting learning rate is used and by what factor is it divided when the error plateaus?", ["starts from 0.1 and is divided by 10"], [["0.1", "10"]]),
    q("res07", R, "number", "What relative improvement did ResNets give on the COCO object detection dataset?", ["28% relative improvement"], [["28%"], ["28 percent"]]),
    q("res08", R, "number", "How many layers were used in the deepest CIFAR-10 network the paper explores?", ["1202"], [["1202"]]),
    U("res09", R, "How many GPU-days did it take to train ResNet-152?"),
    q("gan01", GA, "text", "What kind of game do the two networks play in the adversarial framework?", ["minimax two-player game"], [["minimax"]]),
    q("gan02", GA, "text", "How are the generator and discriminator defined so that the system can be trained with backpropagation?", ["deﬁned by multilayer perceptrons"], [["multilayer perceptron"]]),
    q("gan03", GA, "text", "Which datasets were the adversarial nets trained on?", ["MNIST", "Toronto Face Database", "CIFAR-10"], [["mnist", "toronto face database", "cifar-10"]]),
    q("gan04", GA, "text", "How is the probability of the test set estimated under the generator distribution?", ["Gaussian Parzen window"], [["parzen"]]),
    q("gan05", GA, "text", "What is not needed during either training or generation in this framework?", ["no need for any Markov chains"], [["markov chain"]]),
    q("gan06", GA, "text", "How does the training procedure alternate between the discriminator and the generator?", ["alternate between k steps of optimizing D and one step of optimizing G"], [["k steps", "one step"]]),
    U("gan07", GA, "What was the total training time in hours on the MNIST dataset?"),
    q("cot01", C, "text", "Which model and how many exemplars achieved state-of-the-art accuracy on GSM8K with chain-of-thought prompting?", ["PaLM 540B with just eight chain-of-thought exemplars"], [["palm 540b", "eight"], ["palm 540b", "8"]]),
    q("cot02", C, "text", "On what kinds of tasks does chain-of-thought prompting improve performance?", ["arithmetic, commonsense, and symbolic reasoning tasks"], [["arithmetic", "commonsense", "symbolic"]]),
    q("cot03", C, "text", "What kind of ability is chain-of-thought prompting according to the results?", ["emergent ability of model scale"], [["emergent"]]),
    q("cot04", C, "text", "At roughly what model size does chain-of-thought prompting start yielding gains?", ["models of ∼100B parameters"], [["100b"]]),
    q("cot05", C, "text", "Which benchmark of math word problems is mentioned in the introduction?", ["GSM8K benchmark of math word problems"], [["gsm8k"]]),
    q("cot06", C, "text", "What did chain-of-thought prompting surpass on GSM8K?", ["ﬁnetuned GPT-3 with a veriﬁer"], [["finetuned gpt-3"], ["fine-tuned gpt-3"]]),
    U("cot07", C, "How many human annotators wrote the chain-of-thought exemplars and how much were they paid?"),
    q("gal01", GA2, "text", "Dove e quando nacque Galileo Galilei?", ["Pisa, 15 febbraio 1564"], [["pisa", "15 febbraio 1564"]]),
    q("gal02", GA2, "text", "Dove e quando morì Galileo Galilei?", ["Arcetri, 8 gennaio 1642"], [["arcetri", "8 gennaio 1642"]]),
    q("gal03", GA2, "text", "In quale opera pubblicata nel 1610 furono presentate le nuove scoperte astronomiche?", ["nel Sidereus Nuncius"], [["sidereus nuncius"]]),
    q("gal04", GA2, "text", "Quando fu costretto Galileo all'abiura?", ["22 giugno 1633"], [["22 giugno 1633"]]),
    q("gal05", GA2, "text", "Come denominò Galileo i satelliti di Giove?", ["stelle medicee"], [["stelle medicee"]]),
    q("gal06", GA2, "text", "Chi erano i genitori di Galileo?", ["Vincenzo Galilei e di Giulia Ammannati"], [["vincenzo galilei", "giulia ammannati"]]),
    q("gal07", GA2, "text", "Quale cardinale incaricò i matematici vaticani di una relazione sulle scoperte di Galileo?", ["cardinale Roberto Bellarmino"], [["bellarmino"]]),
    q("gal08", GA2, "text", "Quale allievo assistette Galileo nel 1639?", ["giovane allievo Vincenzo Viviani"], [["viviani"]]),
    U("gal09", GA2, "Quanti figli legittimi ebbe Galileo con la moglie Marina Gamba?"),
    q("dan01", DA, "text", "Dove e quando nacque Dante Alighieri?", ["Firenze, tra il 14 maggio e il 13 giugno 1265"], [["firenze", "1265"]]),
    q("dan02", DA, "text", "Dove morì Dante e in che anno?", ["Ravenna, notte tra il 13 e il 14 settembre"], [["ravenna", "1321"]]),
    q("dan03", DA, "text", "Con chi fu concordato il matrimonio di Dante quando aveva dodici anni?", ["matrimonio con Gemma Donati"], [["gemma donati"], ["gemma"]]),
    q("dan04", DA, "number", "In quale anno fu concordato il matrimonio di Dante con Gemma?", ["nel 1277, fu concordato il suo matrimonio con Gemma"], [["1277"]]),
    q("dan05", DA, "text", "Chi guidava i giovani poeti che espressero dissenso verso la complessità stilistica?", ["capeggiati dal nobile Guido Cavalcanti"], [["guido cavalcanti"], ["cavalcanti"]]),
    q("dan06", DA, "number", "In che anno Dante fu eletto uno dei sette priori?", ["Nell'anno 1300, Dante fu eletto uno dei sette priori"], [["1300"]]),
    q("dan07", DA, "text", "Quando fu condannato Dante alla confisca delle proprietà?", ["27 gennaio 1302"], [["27 gennaio 1302"]]),
    q("dan08", DA, "text", "Quale opera Dante stava scrivendo tra il 1308 e il 1311?", ["stava scrivendo il De Monarchia"], [["de monarchia"], ["monarchia"]]),
    U("dan09", DA, "Quanti versi ha in totale il Convivio secondo l'edizione critica?"),
    q("tur01", T, "text", "When and where was Alan Turing born?", ["23 June 1912"], [["23 june 1912"]]),
    q("tur02", T, "text", "When did Alan Turing die?", ["7 June 1954"], [["7 june 1954"]]),
    q("tur03", T, "text", "Where did Turing work during World War II?", ["Government Code and Cypher School at Bletchley Park"], [["bletchley park"]]),
    q("tur04", T, "text", "From which university did Turing earn his doctorate and in which year?", ["earned a doctorate from Princeton University in 1938"], [["princeton", "1938"]]),
    q("tur05", T, "text", "What was the title of Turing's 36-page paper on computable numbers?", ["On Computable Numbers, with an Application to the Entscheidungsproblem"], [["on computable numbers"]]),
    q("tur06", T, "text", "Which early stored-program computer did Turing design at the National Physical Laboratory?", ["designed the Automatic Computing Engine"], [["automatic computing engine"], ["ace"]]),
    q("tur07", T, "number", "In what year did Turing join Max Newman's Computing Machine Laboratory at Manchester?", ["In 1948, he joined Max Newman's Computing Machine Laboratory"], [["1948"]]),
    q("tur08", T, "text", "Where did Alan Turing die?", ["Wilmslow, Cheshire, England"], [["wilmslow"]]),
    U("tur09", T, "How many chess games did Turing play against Claude Shannon?"),
    q("mar01", M, "number", "How tall is Olympus Mons?", ["Olympus Mons, 21.9 km"], [["21.9"]]),
    q("mar02", M, "text", "What are the two natural satellites of Mars?", ["Phobos and Deimos"], [["phobos", "deimos"]]),
    q("mar03", M, "number", "How long is a Martian year in Earth days?", ["687 Earth days"], [["687"]]),
    q("mar04", M, "number", "How long is a Martian solar day (sol) in hours?", ["sol) is equal to 24.6 hours"], [["24.6"]]),
    q("mar05", M, "number", "How long is Valles Marineris?", ["Valles Marineris, 4,000 km"], [["4000"], ["4,000"]]),
    q("mar06", M, "number", "What is the mean diameter of Mars?", ["mean diameter, 6,779 km"], [["6779"], ["6,779"]]),
    q("mar07", M, "text", "Who discovered the moons of Mars and in which year?", ["discovered in 1877 by Asaph Hall"], [["asaph hall", "1877"]]),
    q("mar08", M, "number", "What is the axial tilt of Mars?", ["axial tilt of 25 degrees"], [["25 degrees"], ["25°"], ["25"]]),
    q("mar09", M, "text", "Which location on Mars has the lowest surface radiation?", ["Hellas Planitia has the lowest surface radiation"], [["hellas planitia"]]),
    q("mar10", M, "number", "What is the approximate speed of sound on Mars according to the Perseverance recordings?", ["approximately 240 m/s"], [["240"]]),
    U("mar11", M, "How many humans have walked on the surface of Mars?"),
]

if __name__ == "__main__":
    Path(__file__).with_name("qa_blind2.json").write_text(json.dumps(QA, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(QA), "questions;", sum(1 for x in QA if x["answer"] is None), "unanswerable")
