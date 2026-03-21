# Chapter 3: Research Design and Analytical Methodology

[ < Back to Chapter 2 ](chapter2.md) | [ Master Index ](dissertation.md) | [ Next: Chapter 4 > ](chapter4.md)

---

### 3.1 Overall Framework and Research Design
This research aims to provide a theoretical foundation for weather-related risk mitigation and adaptation strategies, which are designed and implemented in the context of global warming and climate change. In general, traditional studies have often treated the physical world and financial systems in isolation, focusing separately on the impacts of abnormal weather events on agricultural production and the climate-related risks on the financial market. Firstly, this research will attempt to go beyond the linear assumptions of the physical world's impacts, providing insights into the complexity of non-linear relationships and various interaction effects. It will then develop a systematic analysis framework that bridges the two aspects of previous studies, conducting a comprehensive investigation of the full-chain transmission process from the physical world to financial markets.

**First Stage: Physical Impact Model**
This stage will construct our first model using the highly homogeneous spatio-temporal units that were calculated in the data preparation chapter. Based on this, this study will identify and quantify the patterns and principles of weather-related risk impacts on agricultural production across different countries, covering four main crops. We will accurately extract the key response pattern, including time sensitivity, condition dependence, and regional baseline, through a robust data analysis procedure. 

**Second Stage: Financial Transmission Model**
This stage will analyse how the weather-related risk impacts the financial market and how the market absorbs this risk information and reflects it in the pricing mechanism. The input weather features are not chosen subjectively; instead, the choices are based on the core findings of the previous stage (the first model), using the two most influential weather variables as model inputs. Thus, this methodology ensures that our analysis is not isolated from the broader context. It constitutes an internal link for these two model analyses.

The key point of this design is the internal logical progression of these two parts, making it an integrated framework that realises the research target from farm field to financial futures market, covering the entire chain of transmission and providing a comprehensive view of how weather-related risks impact the economy and society.

### 3.2 The Failure of Linear Models
The first point we need to clarify is why not choose the explainable linear model, as the statistical coefficient has a defined economic meaning and is widely used in previous research.

**3.2.1 Theoretical Mismatch**
The theoretical mismatch is the basic reason why we abandon this traditional method. The sparsity and high-dimensional dataset were determined to have lower accuracy. Furthermore, the complex non-linear relationship, interaction effects, and thresholds are also fundamental characteristics of this study’s research data, which means that the static coefficients and linear assumptions have inherent conflicts with the target data frame. Although some researchers use a piecewise function to capture the non-linear relationships, this approach heavily depends on preset conditions, which are, to some level, subjective and not a data-driven method.

**3.2.2 Empirical Failure in Practice**
This theoretical mismatch has been illustrated in practice. For the first model, we implement a panel regression model using carefully designed weather feature indices, which were calculated through a complex feature engineering procedure discussed in the next chapter. The model explainability is generally at a lower level for all four crops (especially for soybean, R²=-0.527). The same scenario exists for the second model. When we introduced raw weather variables to the ARIMAX model, the model accuracy dropped dramatically. These failures of linear models all justified the choice of a more advanced non-linear model.

### 3.3 The Unique Suitability of XGBoost + SHAP
The research framework chosen for XGBoost+SHAP values is not simply the addition of two methods; it is based on the unique characteristics of the dataset, and the well-functional synergy effect demonstrates a clear advantage over other non-linear methodologies. The XGBoost provided a highly accurate model which well captured the internal pattern and principle of our dataset. However, the non-linear model, such as XGBoost and other machine learning algorithms, is often described as a “black box”; thus, SHAP is employed as a powerful tool to open the “black box” of the XGBoost model. Compared with other explainability methods, it has the following advantages, which make it better suited for our research questions: 1) only attribution methodology, which holds all three ideal characteristics: local accuracy, missingness, and consistency; 2) both local and global interpretability; 3) revealing complex interactions (Molnar, 2025).

**3.3.1 Complex Interactions & The Demand for Trustworthy Attribution**
Our first model dataset's most essential characteristic is high-dimensional & complex Interactions, it covers a list of countries, each of which has multiple climate-grain zones, across 27 years, with 24 periods, each column represents a specific weather event which occurs in a particular period, each row represents a data point (a vector with over 150 dimensions), cover all periodic weather events defined for each zone in a specific country of single year range from 1990 to 2016. It has over one hundred features, but a relatively minor set of data points (less than 1500)

The second model dataset has a similar high-dimensional structure, with each column representing a weather statistic value for a specific zone of each country, and each row is a daily time point for the high-dimensional vector (multiple zones of several countries with different weather information). This dataset contains over 60 features and utilises a financial index as the target variable.

Through the recursive splitting and regularised objectives of the tree, the model can natively capture high-order feature interactions, which enables it to identify interaction patterns, such as the impact of an intense cold wave that occurred last winter and the effect of the spring drought. Other models cannot easily identify such a complex interaction.  However, the core advantage of this method lies in its combination with TreeSHAP, which is specially optimised to utilise the model's tree structure and quickly compute exact (not approximate) SHAP values, providing efficient and accurate attribution. (Molnar, 2025)
Other machine learning algorithms, such as neural networks, typically require a model-agnostic method for explanation, like KernelSHAP. This method tends to assign weights to unrealistic data instances and conduct predictions based on these unrealistic samples, leading to misleading and incorrect attribution results.

**3.3.2 The Natural Sparsity of Abnormal Weather Events**
The second essential character of the first dataset is Sparsity. Based on the definition of abnormal weather events, it will have a large number of null value cells, as these events do not occur frequently. Even though we aggregated the values in multiple fragile areas within each zone, the final values of the dataset will still be continuous rather than discrete. XGBoost has advanced capabilities in handling tabular data with many missing values. By utilising the built-in sparsity-aware split finding algorithm, it focuses only on the non-empty cells. It determines an optimal direction for the missing value, resulting in high calculation efficiency. (Chen & Guestrin, 2016; Grinsztajn et al., 2022; Molnar, 2025)

Unlike XGBoost, other tree algorithms, such as random forest, will scan through all the data points, which leads to numerous unnecessary calculations and increased running time. Different machine learning algorithms like NNs, which have better performance in unstructured data, due to the smooth bias and rotation invariance, do not have good performance in tabular data (Chen & Guestrin, 2016; Grinsztajn et al., 2022)

**3.3.3 Low Signal-to-Noise Ratio & Overfitting Control**
A key characteristic shared by datasets of the two models is a low signal-to-noise ratio. Both the weather-related risk and the financial datasets are full of complex relationships and contain significant noise. XGBoost, as an advanced implementation of gradient boosting, addresses this by focusing on hard-to-predict samples; each new tree in the ensemble is trained to fit the pseudo-residuals of the previous ones. This process amplifies weak signals and performs well in identifying subtle relationships.

As a result, it outperformed the other tree mode, random forest, which implements bagging to average predictions and smooth variance (simulating the reduction of subtlety). The NNs’ bias mismatch leads to the omission of irregular signals and the loss of feature individuality. Thus, on small to medium-sized datasets, like those used in this study, NNs often exhibit lower accuracy and a higher risk of overfitting. Even after extensive parameter tuning, their performance still did not reach the same level as XGBoost (Chen & Guestrin, 2016; Grinsztajn et al.,2022).

### 3.4 Chapter Summary 
This chapter outlines the overall research design and analysis approach of this study. Firstly, we conduct a two-stage integrated analysis framework, aiming to break through the transmission chain from the physical world to the financial market to address the integrated gap in the current study. Then, we evidenced the limitations of linear model adaptation for this case study, through a theoretical and empirical review, justifying the necessity of conducting non-linear model analysis. 

Lastly, this chapter focuses on the reasons for choosing XGBoost+SHAP values as the central methodology. This method not only accurately captures the complex characteristics of the research dataset, such as high dimensionality, sparsity, strong interactions, and non-linearity, but also provides both global and local attribution explanations by opening the “black box” of machine learning. This carefully chosen method is designed to address the methodology gap, the interaction and non-linearity gap, and the explainability gap, which were identified in the literature review, providing a valid foundation for the following in-depth investigation and result analysis.

---
[ < Back to Chapter 2 ](chapter2.md) | [ Master Index ](dissertation.md) | [ Next: Chapter 4 > ](chapter4.md)
