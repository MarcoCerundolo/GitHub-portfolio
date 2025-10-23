## The Technological Legacy of the Cold War: Military Procurement and US Innovation

## Overview

The goal of the project is to assess the causal impact of the surge in Cold War military procurment spending that occured under Reagan on the formation of regional innovation clusters. I construct a panel dataset (3,100 counties x 37 years) with the number of patents produced, number of citation-weighted patents and amount of Department of Defence procurement spending in each county-year combination.


## Key Features

- Difference-in-Differences to evaluate long-run effect on Reagan spending surge 
- Instrumental Variables analysis to overcome ommited variable bias
- Heterogeneity analysis based on the degree of local firm competition in a county
- Constructing a panel dataset from a range of messy data sources

## Dataset

1) Universe of US patents [private source]
2) All procurment contracts by the US Department of Defence (DoD) 1966-2003 [public records]
3) County-level descriptive variables from a range of sources [public records]

The raw data is not included in the repo because the patent data is private. However, intermediate datasets are included

## Methodology

To read about the methodology in more detail refer to the [thesis pdf](thesis_pdf/Thesis.pdf).

 - Difference-in-differences: the treatment year is when Reagan is first inaugrated into office. The treated counties are those who experience above the median level increase in county level spending in the years immediately prior to immediatley post his election.

The estimates from this approach suffer from omitted variable bias since treatment status is correlated with innnovative potential and therefore I use an IV analysis

 - Instrumental Variables: I use a shift-share instrument where the shares capture the specialisation of each county in different technological categories and the shocks capture exogenous changes to which technologies the DoD primarily invests in.
  
 - Heterogeneity Analysis: I construct a measure of local firm competition and I run the analysis on counties in distinct quartiles of this index.

## Results & Insights

To read more about the results and their interpretation refer to the [thesis pdf](thesis_pdf/Thesis.pdf).

- Event study suggests that the Reagan spending surge had a [long-run effect on innovation](output/figures/trendplot_num_final.png).

- [Shift-share IV estimates](thesis_pdf/Thesis.pdf) that county patenting has an elasticity to procurement spending of 0.11% (sig. 1%).

- [Heterogeneity analysis](output/figures/hhi.png): treatment effect is larger in counties with a higher degree of local competition . This is consistent with the presence of local knowledge spillovers. 


## Folder Layout

```
us_innovation/
├── data/
│   ├── raw/          # empty
│   ├── interim/      # empty
│   └── processed/    # contains constructed files used for analysis
├── output/
│   ├── figures/      # 
│   └──  tables/      #           
├── code/   
│   └──  analysis/    #         
├── thesis_pdf        # Contains the thesis pdf and the slides from the thesis defence 
└── 
```
