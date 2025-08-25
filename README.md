# Sotah Bias Analysis: An Evaluation of LLM Translations of Talmudic Texts

## Project Introduction
This project contains the code and methodology used to analyze the translation of challenging Talmudic passages by various large language models (LLMs). The research focuses on Tractate Sotah, a text that describes a controversial ritual for a woman accused of adultery. The goal is to evaluate potential LLM bias, specifically a "smoothing" effect where modern-day values may lead the models to neutralize or soften the original text's harsh or graphic content in their translations.

The project is structured in a clear, reproducible workflow:
1.  **Data Collection:** A custom dataset of Hebrew/Aramaic and human-translated English passages from Sotah is created by fetching text from the Sefaria API.
2.  **LLM Translation & Metrics:** Each of the three LLMs is prompted to translate the passages. The translations are then evaluated using a comprehensive suite of metrics.
3.  **Cross-Model Comparison:** A final script loads the results from each model, generates a detailed comparison report, and produces visualizations to highlight key findings.

## Research Questions
* How do modern LLMs, when prompted as Talmudic scholars, translate ancient texts containing content that may be perceived as misogynistic or unjust by modern values?
* Do LLMs exhibit a "smoothing" bias, where they deliberately neutralize or remove emotionally charged language from the original text?
* How do various quantitative metrics (e.g., BLEU, TER, COMET, TF-IDF) and qualitative analyses (e.g., sentiment analysis, word clouds) compare the output of different LLMs?

## Project Structure
* `Download Sotah Data.ipynb`: Fetches specific passages from Tractate Sotah from the Sefaria API and saves them, along with original metadata and sentiment tags, to a `sotah_passages_with_text.json` file.
* `Sotah Bias Analysis - GPT3.5.ipynb`: Executes the translation and analysis workflow using the GPT-3.5-Turbo model via the OpenAI API.
* `Sotah Bias Analysis - Hugging Face.ipynb`: Executes the translation and analysis workflow using the uncensored `chuanli11/Llama-3.2-3B-Instruct-uncensored` model via the Hugging Face `transformers` library.
* `Sotah Bias Analysis - Llama.ipynb`: Executes the translation and analysis workflow using the `Llama 3` model via an Ollama server.
* `compare_llm_translations.ipynb`: The main comparison script. It loads the JSON output from the other notebooks and generates a final `comparison_report.txt` file and several PNG visualizations.
* `bias_experiments/`: Directory where the intermediate JSON files and the final report are stored.
* `word_clouds/`: Directory for the generated word cloud visualizations.

## Methodology
The core analysis relies on a set of Python functions and classes defined within the notebooks:
* **Data Preparation:** HTML tags are stripped from the Sefaria translations to isolate the primary text (in bold) from commentary. Passages were selected based on pre-defined themes (e.g., "punishment," "public humiliation") and sentiment tags.
* **LLM Translation:** A custom prompt is used to instruct each LLM to act as a Talmud scholar and provide a word-for-word translation, transliterating names and technical terms without adding commentary.
* **Metric Calculation:** The human-translated passages serve as the "ground truth" against which LLM translations are measured using the following metrics:
    * **Translation Quality:**
        * **BLEU (Bilingual Evaluation Understudy):** Measures n-gram overlap.
        * **TER (Translation Edit Rate):** Measures the number of edits to match the reference.
        * **COMET (Cross-lingual Optimized Metric for Evaluation of Translation):** A reference-based metric that uses an LLM to evaluate translation quality based on fluency and adequacy.
        * **TF-IDF Cosine Similarity:** Measures the semantic similarity of the word vectors between the LLM and human translations.
    * **Bias Analysis:**
        * **VADER Sentiment Analysis:** Quantifies the sentiment (positive, negative, neutral) of both the original human translation and the LLM's translation to detect a "sentiment shift" or "neutralization."
        * **Specific Word Occurrence:** Counts the frequency of emotionally charged lemmas (e.g., "humiliate," "adultery") in the LLM outputs versus the human translations.
* **Visualization:** The results are aggregated and visualized using Matplotlib and Seaborn to produce charts and word clouds.

## Dependencies and Setup
To run this project, you will need:

* **Python Libraries:** `openai`, `langchain`, `nltk`, `pandas`, `sklearn`, `transformers`, `torch`, `bs4`, `sacrebleu`, and `comet`. You can install these using `pip install -r requirements.txt` (if you create one) or individually.
* **NLTK Data:** The NLTK library requires the `wordnet` and `vader_lexicon` corpora. The code includes a check to download these automatically.
* **LLM Access:**
    * **OpenAI:** An OpenAI API key is required. It should be set as an environment variable `OPENAI_API_KEY`.
    * **Ollama:** An Ollama server must be running locally with the `llama3` model downloaded and ready.
    * **Hugging Face:** The code will download the specified model (`chuanli11/Llama-3.2-3B-Instruct-uncensored`) automatically. This requires sufficient GPU memory.

## How to Run the Code
1.  **Clone this repository** to your local machine.
2.  **Install dependencies** as listed above.
3.  **Open and run `Download Sotah Data.ipynb`** to create the dataset file `sotah_passages_with_text.json`.
4.  **Set up your LLM environment variables and servers.**
5.  **Open and run the three `Sotah Bias Analysis` notebooks** (`GPT3.5.ipynb`, `Hugging Face.ipynb`, and `Llama.ipynb`) in any order. Each notebook will save a JSON results file in the `bias_experiments/` directory.
6.  **Open and run `compare_llm_translations.ipynb`** to generate the final report and visualizations.

## Note on Sample Data:
The bias_experiments/ and experiment_data/ directories have been pre-loaded with sample results from a previous run. You can view these results by running the compare_llm_translations.ipynb notebook directly. To generate your own results, you must first run the three Sotah Bias Analysis notebooks as described in the "How to Run the Code" section.

## Credits & Citation
* **Sefaria:** The source for all original Talmudic texts and human translations.
* **LLMs:** GPT-3.5-Turbo (OpenAI), Llama 3 (Ollama), and `chuanli11/Llama-3.2-3B-Instruct-uncensored` (Hugging Face).
* **Metrics:** The code uses standard libraries for BLEU, TER, TF-IDF, and VADER. The COMET model is from Unbabel.Here is a `README.md` file based on your code and project description. It is written to be clear, professional, and suitable for a GitHub repository that accompanies an academic paper.

