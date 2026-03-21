# Chapter 1: Introduction

[ < Back to Abstract ](abstract.md) | [ Master Index ](dissertation.md) | [ Next: Chapter 2 > ](chapter2.md)

---

### 1.1 Research Objective and Core Questions
The overall research objective is to establish a systematic analysis framework that bridges the physical world and financial markets, and to conduct a comprehensive and in-depth investigation of the entire transmission chain from farm fields to financial markets.

This overall objective will be realised by answering the questions below.

**1.** How do abnormal events, categorised by severity and occurrence time, impact the yield of the four main crops in the primary production regions? 

**2.** Is the impact and relationship linear or non-linear?

**3.** How do the two or more events interact and impose compound effects?

**4.** How does the physical world's shock transmission affect the financial market? 

**5.** What level of impact, and how does the market mechanism price those weather-related risks? 

### 1.2 The Data Preparation and Feature Engineering
We have utilised multiple sources of heterogeneous data, including ERA5-Land global climate data, the Iizumi global crops’ yield dataset, and the S&P GSCI wheat Index. Based on these datasets, this research has implemented a three-stage data preparation procedure, spanning the scope of the data range to spatial-temporal standardisation. Finally, it constitutes a composite index of quantitative abnormal weather events. 

### 1.3 The Analysis Framework and Model Choice
A two-stage analysis framework has been developed, which includes two internally linked models: an agricultural physical impact model and a financial transmission model. The second model input is derived from the first model's conclusions, ensuring the internal logic connections between these two analyses.

This study highlights and validates the theoretical mismatch and empirical shortcomings of the traditional linear model. Hence, this evidence suggests that the model is unsuitable for our analysis and justifies the final choice of XGBoost + SHAP values as our modelling strategy.

This methodology has clear advantages for handling our dataset, which is characterised by complex interactions, high dimensionality, sparsity, and a low signal-to-noise ratio. Additionally, it can provide reliable attribution explanations by combining it with the TreeSHAP method.

### 1.4 Dissertation Structure
This research has six chapters; the allocation is as follows:

**Chapter 1: Introduction.** This chapter introduces the background, research questions, motivation, and objectives of this study. Additionally, this chapter presents the methodology and structure of this dissertation.

**Chapter 2: Literature Review.** This chapter systematically and critically reviews the related literature, identifies the gaps in current research, including methodology, non-linear relationships, integrated analysis, and explainability, and provides an entry point and Theoretical Fundamentals.

**Chapter 3: Research Design and Analysis Methodology.** This chapter discusses the two-stage model strategy and justifies the choice of XGBoost+SHAP values as a core analysis tool, describing how this method addresses the challenges caused by data characteristics and fills the research gap.

**Chapter 4: Data Preparation.** This chapter describes the data preparation process in detail, including the adoption of feature engineering methods for multi-source heterogeneous data (weather, yield raw data). Through the complex process of spatial-temporal standardisation, event quantisation, and index construction, a structured dataset is built for modelling. 

**Chapter 5: Model and Result Analysis.** This chapter presents and thoroughly investigates the analysis results of the two core models. The first model analysis will decompose agricultural vulnerability, revealing three fundamental principles: time sensitivities, condition dependence, and regional baseline. The second model bridges the gap between the physical world and the financial market, analysing the transmission mechanisms of weather-related risks' impact on the financial market.

**Chapter 6: Conclusion.** This chapter summarises the core findings and main contributions of this study, providing an honest critique of the study's limitations in methodology, data choice, and model design. Based on this, it proposes future research directions.

---
[ < Back to Abstract ](abstract.md) | [ Master Index ](dissertation.md) | [ Next: Chapter 2 > ](chapter2.md)
