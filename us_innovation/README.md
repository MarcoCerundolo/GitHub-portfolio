## The Technological Legacy of the Cold War: Military Procurement and US Innovation

## Overview

The goal of the project is to assess the causal impact of the surge in Cold War military procurment spending that occured under Reagan on the formation of regional innovation clusters. I construct a panel dataset (3,100 counties x 37 years) with the number of patents produced, number of citation-weighted patents and amount of Department of Defence procurement spending in each county-year combination.


## Key Features

- Difference-in-Differences to evaluate long-run effect on Reagan spending surge 
- Instrumental Variables analysis to overcome ommited variable bias
- Heterogeneity analysis based on the degree of local firm competition in a county
- Constructing a panel dataset from a range of messy data sources

## Dataset

1) Universe of US patents (private source)
2) All procurment contracts by the US Department of Defence (DoD) 1966-2003 (public records)
3) County-level descriptive variables from a range of sources (public records)

The raw data is not included in the repo because the patent data is private. However, intermediate datasets are included

## Results & Insights

- Event study suggests that the Reagan spending surge had a long-run effect on innovation [differences-in-differences](output/figures/trendplot_num_final.png).

- Shift-share IV estimates county patenting elasticity to procurement of 0.11% (sig. 1%) [thesis pdf](thesis_pdf/Thesis.pdf).

- Heterogeneity analysis: treatment effect larger in counties with more local competition [heterogeneous effects](output/figures/hhi.png). This is consistent with the presence of local knowledge spillovers. 


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
