### 6.3 Limitations of This Study

#### 6.3.1 Methodology level

**1. The systematic bias of the growing season algorithm**
The algorithm, based on the daily average temperature curve crossing the preset threshold, exhibits good performance in high-latitude regions with distinct seasons. However, it cannot identify the clear start and end points of the growing season for subtropical and warm-temperature zones (for example, the middle and lower reaches of the Yangtze River in China), which have more complex climate models. This algorithm bias leads to those regions being systematically filtered out in the data preparation stages; the following model build is based on the region samples with relatively simple climate and weather patterns. Hence, this led to a geographic bias in our conclusion, limiting the generalizability of our findings to low- and middle-latitude regions.

**2. The generalisability of the Standardised 24×15-day periods**
The standardised 24×15-day periods-division method, have not been adjusted for adverse seasonality of the Southern Hemisphere, potentially leading to mismatch of weather events with the core growth stages of crops in countries like Australia, may misalign with the weather factors impacting on those countries, in addition, this framework also cannot capture the multi growth seasons in tropical regions and semi-tropical regions, this lead to compromise model accuracy.

#### 6.3.2 Data level

**1. The inherent limitations of the first model (Yield) are the estimation nature of the target variables.** Iizumi's global yield information is an estimate of production based on a downscaling model that integrates satellite remote sensing and statistical data. This means that these data are not actual farm field data, indicating that the research's model study target is to estimate another model prediction, thereby causing risks of error propagation and learning model artefacts.

**2. The static assumption of this yield information dataset imposes further limitations.**
This model employs a 2000 static planting area distribution for model building, which fails to capture the dynamic changes in farm field utilisation due to urbanisation, agricultural development, and climate adaptation. Thus, the model cannot capture the yield change caused by the shifting of cultivation zones, which may lead to less accurate attribution analysis.

**3. Another limitation is the large-scale weather information data.**
The primary dataset of ERA5-Land is a gridded dataset with an 11 km resolution. Although this is sufficient to support a global-level comparison, it may average or smooth the changes in local region characteristics, such as microclimate, Terrain, and slope direction, which are also crucial for crop planting.

**4. A further limitation of the dataset is its shorter period of coverage.** The physical shock model utilised a dataset from 1990 to 2016, which lacks the most recent data. However, recent data may exhibit significant differences from previous periods due to the acceleration of climate change resulting from the rapid accumulation of CO2 levels in the atmosphere. Hence, the model might lose its most valuable conclusions. The financial model dataset only covers the period from 2008 to 2020, omitting the pre-crisis and post-COVID periods, both of which exhibit different patterns of market change. This omission compromises the model's ability to capture the trends of long-term cycles and structural changes, thereby influencing the long-term stability and generalizability of the model's conclusions.

#### 6.3.3 Model specification level 

**1. The omitted variable in the agricultural impact model**
This research focused on weather factors and failed to incorporate other key influence factors, such as soil conditions and quality, irrigation, fertiliser applications, and changes in crop varieties, all of which are controlled by agricultural technology. In a real-world scenario, those non-weather factors are the primary or essential moderating factors of agricultural yield; omitting these might lead to overestimating or underestimating the influence of weather factors.

**2. The simplification of driver factors for the financial transmission model**
The second-stage model is more focused on weather risk transfers from the physical world. However, in an actual scenario, the financial market is heavily and directly influenced by macroeconomic policy, individual market player behaviours, and liquidity levels, which represent the inherent financial drivers. This model has not accounted for the factors that impact it, which will limit its explanatory ability, especially in understanding the significant market turning point.

### 6.4 Avenue for Future Research

#### 6.4.1 Methodology Refinement and Improvement
Future research should continue to refine the algorithm for defining the growing season, which is adaptable to multiple regions with both complex and simple seasonal patterns. The new algorithm should not only consider temperature changes but also incorporate additional weather factors, such as rainfall rhythms, changes in photoperiod, and soil moisture, combined with a specific crop's phenology knowledge base, to build a regional adaptation multidimensional model.

Prioritising the time frame is another field the future study should focus on. For the Southern Hemisphere, it should align with the period of the Northern Hemisphere or build a more accurate period division matched to the growing stage of a specific crop; for multi planting seasons, it should separate the different seasons' yield information, like spring wheat, Winter wheat, major season rice, second season rice, etc., to capture risk factors more dynamically.

#### 6.4.2 Dataset Level Integration and Validation
The future study should overcome the limitation of current data by utilising a dataset with higher quality and more dynamic information. The future model should investigate and cross-validate with the other global yield dataset, which contains dynamic farm field statistics data, especially for data-rich regions like the USA and the EU, conduct a regional validation process, utilising the officially published high-precision yield data, which is based on the actual field surveys, to test and align the global model performance.

#### 6.4.3 Analysis Framework Level Expansion and Synthesis
The agricultural impact model should be extended by incorporating more non-weather factors that represent the agricultural technical level and are more controllable. Setting these factors as static or dynamic features will further investigate the interaction of non-weather factors with weather factors, providing a comprehensive view of the aggregate impact on crop planting.

The financial model should be further developed by accounting for more financial inheritance factors like exchange rates, interest rates, PMI, VIX and holding report to get full and better understanding of market movements, and also combined with the change of international trade price volume and patterns to investigate the both future market and spot market of agricultural commodity to get information of how different markets linked internally and transfer information and risk factors.

Ultimately, the Future model should aim to internalise human behaviour, incorporating climate adaptation behaviours such as upgrading and changing crop varieties, adjusting planting dates, investing in irrigation facilities, and investors adjusting their trading strategies as internal features. Through the above model adjustment, we can conduct more accurate and comprehensive evaluations and estimations of agricultural climate risk.

---
[ < Back to Chapter 5 ](chapter5.md) | [ Master Index ](dissertation.md) | [ Next: Appendix 1 > ](appendix1.md)
