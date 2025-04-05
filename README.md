# 🧠 DataAttackPubMed: Training Data Extraction & Memorization in Biomedical Language Models

 **Abstract:** This project explores extractable memorization in domain-specific biomedical language models. By designing and executing both black-box and white-box attacks on models such as PubMedBERT and BioGPT, we identify and verify instances of memorized training data. Our approach involves a combination of entity-aware masking, confidence-based token verification, zlib compression ratio analysis, and sliding-window perplexity metrics. This study demonstrates the privacy risks of unintended data leakage in biomedical LMs and provides reproducible pipelines for future research.

---

## 📚 Project Overview

Large Language Models (LLMs) trained on vast biomedical corpora often memorize fragments of their training data, posing privacy and copyright risks. This project investigates such memorization behavior in:
- **PubMedBERT** (masked language model)
- **BioGPT-Large** (generative biomedical model)

We implement and compare:
- **White-box masked token reconstruction attacks** (on PubMedBERT)
- **Black-box generation-based extraction attacks** (on BioGPT)

Our final pipeline includes advanced masking, zlib/perplexity metrics, fuzzy n‑gram matching, and document-level analysis for data extraction and verification.

You will find following attack strategies through our different pipelines: (That's how we started in chronological order)
- **White-box attack (BioGPT + pubmedBERT)** - manually tweeking parameters for attack like top-k, top-n, temperature, implmented simple substring search - checks strict memorization.
- **Black-box attack (BioGPT)** - Attacked the model in black-box setting with strict substring match, fuzzy n-gram and sliding window perpleixty.
- **Decaying temperature** - implmented decaying temperature setup like carlini in Black-box setup
- **Embedding Search** - Used embedding based search on generated output to check outliers / boundary memorization. 
- **Combined masking (NER + numerical + random)** - white-box attack on **PubmedBERT** with different masking strategies and using fuzzy-ngram verification on top-suspicious masks

---

## 🧪 Key Contributions

- ✅ **24 verified memorized outputs** extracted from PubMedBERT using a combined masking strategy.
- 🧬 **Outlier and borderline memorization detection** using zlib ratio and sliding window perplexity.
- 🧠 **Analysis of hit vs. missed token patterns** to reveal model behavior across scientific language.
- 🔍 **Decaying temperature black-box generation** for BioGPT to simulate real-world extraction settings.
- 📊 **Visualizations** of token prediction, zlib ratio distributions, and prompt frequency.

---

## 🗂️ Repository Structure

```
DataAttackPubmed/
│
├── bioGPT_attack/              # Black-box extraction results & visualizations for BioGPT
├── pubmedBERT_attack/          # Masked token reconstruction attack results for PubMedBERT
├── data/                       # pubmed_abstracts.json, pmc_fulltext.json (sample biomedical corpus)
├── seminar_report/             # Final project report, papers referenced
├── src/                        # all the attack and analysis pipeline
│    ├── attackAnalysis         # visualization and analysis of generations and attack results
│    ├── bioGPT                 # 5 different pipelines implementing different attack strategies for bioGPT
│    ├── dataCollection         # data corpus scraper
│    └── pubmedBert             # 2 different pipelines implementing different attack strategies for PubmedBERT
├── environment.yml             # Conda environment setup
├── README.md                   # You're here!
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/nr59684/DataAttackPubmed.git
cd DataAttackPubmed
```

### 2. Set up environment
```bash
conda env create -f environment.yml
conda activate dataCon
```

### 3. Prepare data
- Add `pubmed_abstracts.json` and `pmc_fulltext.json` to the `data/` folder from [Google Drive link](https://drive.google.com/drive/u/0/folders/1kzFIK0HNpS0edebPVzdcwIQEoUBw4Jhz).
- Optionally re-scrape using `PubMedData.ipynb` or `pmcData.py`.

---

## ⚙️ Running the Attacks

### ▶️ PubMedBERT (Masked LM) Attack

```bash
cd pubmedBERT_attack
and manually run pubmedBert.ipynb
```

- Uses combined masking (NER + numbers + random)
- Produces memorized candidate predictions with confidence filtering
- Verifies hits using fuzzy n‑gram match against local corpus

### ▶️ BioGPT (Black-box Generative) Attack

```bash
cd bioGPT_attack
and manually run bioGPT_decayTemp.ipynb
```

- Generates completions using decaying temperature strategy
- Computes zlib ratio, perplexity, and detects suspicious outputs
- Verifies near-verbatim overlap using fuzzy search and scoring

---

## 📈 Results & Visualizations

- `pubmedbert_memorization_results.json`: Verified hits from PubMedBERT
- `biogpt_generations.json` & `biogpt_attack_results.json`: Generated texts and suspicious completions
- Visual outputs:
  - Hit vs. Missed Pie Chart
  - Top 5 Frequent Token Bar Charts
  - Zlib Ratio & Perplexity Histograms
  - Prompt Frequency Distribution

📊 Located in `bioGPT_attack/` and `pubmedBERT_attack/`

---

## 🧾 Sample Results

- **PubMedBERT:**  
  - `this corrects the doi: <DOI_NUMBER>` predicted with 99%+ confidence across multiple instances.
  - One sample is a sentence about “academic factor, students, university, research” – a generic academic sentence that was memorized exactly. Also,
  - The other sample `(the one with the HTML tag "< i > online [MASK] [MASK] is [MASK] for this article.</i [MASK]")` is clearly different: it appears to be a snippet from an HTML formatted text indicating that “online supplemental material is available for this article.”
  - The model was able to exactly reproduce these parts of its training data. They show that besides the common citation pattern, the model also memorized other types of content (a generic academic sentence and an HTML snippet)

- **BioGPT:**  
  - Borderline cases detected with high zlib ratio (~4.0) and low perplexity (~2.1) despite lack of full match.
  - We tried running different settings, generation and token counts. But couldn't find any memorization even for (4000*512) tokens. which came to conclusion for us, maybe our attack has failed to penetrate the model or model was actually tweeked well to remove memorization


---

## 📄 Related Papers Referenced

- Carlini et al. (2021) – *Extracting Training Data from Large Language Models*
- Meeus et al. (2023) – *Did the Neurons Read Your Book?*
- Nasr et al. (2023) – *Scalable Extraction of Training Data*


---

## 🧠 Future Work

- Extend attack pipelines to larger models (BioMedLM, PubMedGPT)
- Explore mitigation via differential privacy or deduplication
- Enhance fuzzy scoring with edit distance or sentence embeddings

---

## 🖊️ Citation

If using this work or code, please cite as:

> Rijhwani, N. (2024) & Krishna, B. (2024). *DataAttackPubMed: Extractable Memorization in Biomedical Language Models*. Final Year Project, GitHub: https://github.com/nr59684/DataAttackPubmed
 
---

## ✍️ Author

**Nilesh Rijhwani** & **Bhavana Krishna** 
🎓 Computational Linguistics research project  
📬 Contact via - [Nilesh](https://github.com/nr59684) Or [Bhavana](https://github.com/Bhavana1202)

---

## 📜 License

This project is licensed under MIT License. See `LICENSE` for more info.
