clear all 

global path "GitHub-portfolio/us_innovation"

global output "$path/output/tables"

use "$path/data/processed/built5.dta"

******************
*** SECOND STAGE ****
******************

egen state_yr = group(state fyear)

*** Weighted Patents - Simple spec. - OLS ***

reghdfe lw_cites_sub ltotal_dollars if semi_intens == 1 & fyear > 1975, abs(fyear county_fips) cluster(county_fips) 

estadd ysumm
local ymean=e(ymean)
local ysd=e(ysd)

outreg2 using "$output/main_results_6676_final.tex", lab dec(5) nocons ///
		title(Intensive Margin Results) ctitle(Citation Weighted Patents)
		

*** Weighted Patents - Simple spec. - IV ***

ivreghdfe lw_cites_sub (ltotal_dollars = log_spending_6675_iv) if semi_intens == 1 & fyear > 1975, first abs(fyear county_fips) cluster(county_fips) 

estadd ysumm
local ymean=e(ymean)
local ysd=e(ysd)

outreg2 using "$output/main_results_6676_final.tex", append lab dec(5) nocons ///
		title(Intensive Margin Results) ctitle(Citation Weighted Patents)
		
*** Weighted Patents - Full spec - OLS ***
reghdfe lw_cites_sub ltotal_dollars avg_wages   pop emp if semi_intens == 1 & fyear > 1975, abs(fyear county_fips) cluster(county_fips) 

estadd ysumm
local ymean=e(ymean)
local ysd=e(ysd)

outreg2 using "$output/main_results_6676_final.tex", append lab dec(5) nocons ///
		title(Intensive Margin Results) ctitle(Citation Weighted Patents)
		
*** Weighted Patents - Full spec - IV ***
ivreghdfe lw_cites_sub avg_wages   pop emp (ltotal_dollars = log_spending_6675_iv) if semi_intens == 1 & fyear > 1975, first abs(fyear county_fips) cluster(county_fips) 

estadd ysumm
local ymean=e(ymean)
local ysd=e(ysd)

outreg2 using "$output/main_results_6676_final.tex", append lab dec(5) nocons ///
		title(Intensive Margin Results) ctitle(Citation Weighted Patents)	
		
*** Weighted Patents - Full spec + state-year FE - OLS ***
reghdfe lw_cites_sub ltotal_dollars avg_wages   pop emp if semi_intens == 1 & fyear > 1975, abs(fyear county_fips state_yr) cluster(county_fips) 

estadd ysumm
local ymean=e(ymean)
local ysd=e(ysd)

outreg2 using "$output/main_results_6676_final.tex", append lab dec(5) nocons ///
		title(Intensive Margin Results) ctitle(Citation Weighted Patents)
		
*** Weighted Patents - Full spec + state-year FE - IV ***
ivreghdfe lw_cites_sub avg_wages   pop emp (ltotal_dollars = log_spending_6675_iv) if semi_intens == 1 & fyear > 1975, first abs(fyear county_fips state_yr) cluster(county_fips) 

estadd ysumm
local ymean=e(ymean)
local ysd=e(ysd)

outreg2 using "$output/main_results_6676_final.tex", append lab dec(5) nocons ///
		title(Intensive Margin Results) ctitle(Citation Weighted Patents)			

erase "$output/main_results_6676_final.txt"	


******************
** ALL COUNTIES
******************	

*** Weighted Patents - Simple spec. - OLS ***

reghdfe lw_cites_sub ltotal_dollars if  fyear > 1975, abs(fyear county_fips) cluster(county_fips) 

estadd ysumm
local ymean=e(ymean)
local ysd=e(ysd)

outreg2 using "$output/main_results_6676_all_final.tex", lab dec(5) nocons ///
		title(Intensive Margin Results) ctitle(Citation Weighted Patents)
		

*** Weighted Patents - Simple spec. - IV ***

ivreghdfe lw_cites_sub (ltotal_dollars = lspending_6675_iv) if  fyear > 1975, first abs(fyear county_fips) cluster(county_fips) 

estadd ysumm
local ymean=e(ymean)
local ysd=e(ysd)

outreg2 using "$output/main_results_6676_all_final.tex", append lab dec(5) nocons ///
		title(Intensive Margin Results) ctitle(Citation Weighted Patents)
		
*** Weighted Patents - Full spec - OLS ***
reghdfe lw_cites_sub ltotal_dollars avg_wages   pop emp if fyear > 1975, abs(fyear county_fips) cluster(county_fips) 

estadd ysumm
local ymean=e(ymean)
local ysd=e(ysd)

outreg2 using "$output/main_results_6676_all_final.tex", append lab dec(5) nocons ///
		title(Intensive Margin Results) ctitle(Citation Weighted Patents)
		
*** Weighted Patents - Full spec - IV ***
ivreghdfe lw_cites_sub avg_wages   pop emp (ltotal_dollars = lspending_6675_iv) if fyear > 1975, first abs(fyear county_fips) cluster(county_fips) 

estadd ysumm
local ymean=e(ymean)
local ysd=e(ysd)

outreg2 using "$output/main_results_6676_all_final.tex", append lab dec(5) nocons ///
		title(Intensive Margin Results) ctitle(Citation Weighted Patents)	
		
*** Weighted Patents - Full spec + state-year FE - OLS ***
reghdfe lw_cites_sub ltotal_dollars avg_wages   pop emp if fyear > 1975, abs(fyear county_fips state_yr) cluster(county_fips) 

estadd ysumm
local ymean=e(ymean)
local ysd=e(ysd)

outreg2 using "$output/main_results_6676_all_final.tex", append lab dec(5) nocons ///
		title(Intensive Margin Results) ctitle(Citation Weighted Patents)
		
*** Weighted Patents - Full spec + state-year FE - IV ***
ivreghdfe lw_cites_sub avg_wages   pop emp (ltotal_dollars = lspending_6675_iv) if fyear > 1975, first abs(fyear county_fips state_yr) cluster(county_fips) 

estadd ysumm
local ymean=e(ymean)
local ysd=e(ysd)

outreg2 using "$output/main_results_6676_all_final.tex", append lab dec(5) nocons ///
		title(Intensive Margin Results) ctitle(Citation Weighted Patents)			

erase "$output/main_results_6676_all_final.txt"	
