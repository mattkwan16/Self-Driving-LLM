# Self-Driving LLM
*Matt Kwan*

*April 2025*

This project explores how large language models (LLMs) can predict medium-horizon ego-vehicle trajectories using natural language descriptions of surrounding objects and scenes, built on real-world nuScenes data.

### Project Highlights
- Converts nuScenes sensor and annotation data into egocentric natural language prompts
- Fine-tunes LLMs (eg, Meta-LLaMA 3.1) using Alpaca format via [Unsloth](https://github.com/unslothai/unsloth)
- Predicts vehicle motion (e.g. "In 3 seconds, arrive 20.7m to your 1 o'clock.") using only language

### Docs
- **Technical Report:** [`Kwan - Directed Study - Final Technical Report.docx`](https://umich-my.sharepoint.com/:w:/r/personal/mattkwan_umich_edu/Documents/Kwan%20-%20Directed%20Study%20-%20Final%20Technical%20Report.docx?d=wc9b73689bd8a4c89b0a22baf8f7a36f9&csf=1&web=1&e=soennm)
- **Related Works Summary:** [`Kwan_DirectedStudy_RelatedWorks.docx`](https://umich-my.sharepoint.com/:w:/r/personal/mattkwan_umich_edu/Documents/Kwan_DirectedStudy_RelatedWorks.docx?d=wb7c07f32911b4863aad76a47bfbbfa26&csf=1&web=1&e=gguQmD)

### Install
Setup and dependencies are managed in the Colab notebooks. Please see:
- ['Kwan - Extract Future Positions.ipynb'](https://colab.research.google.com/drive/1gZ20hAuE0xRV7Gh1c59FCetn4_1qwpcw?usp=sharing)
- ['Kwan - Self-Driving Llama3.1_(8B)-Alpaca.ipynb'](https://colab.research.google.com/drive/1SGg1sXx-S3KPGEwHza1esCvL5V_HQOBH?usp=sharing)
