clear all 

global path "/Users/marcocerundolo/Dropbox/DL_marco"

global output "$path/output/outreg/regressions/baseline"

use "$path/output/intermediate/panel/built5.dta"

******************
*** FIRST STAGE****
******************

preserve

keep if semi_intens == 1 & fyear > 1975 & ltotal_dollars> 0 & lspending_6675_iv > 0

quietly sum lspending_6675_iv, d
return list
gen lspending_6675_iv_mean = r(mean)

replace lspending_6675_iv = lspending_6675_iv - lspending_6675_iv_mean

* First, residualize ltotal_dollars on controls and fixed effects:
reghdfe ltotal_dollars if semi_intens==1 & fyear > 1975 & ltotal_dollars> 0, absorb(fyear) cluster(county_fips) resid
predict resid_lt, resid

* Next, residualize spending_iv on the same controls and fixed effects:
reghdfe lspending_6675_iv if semi_intens==1 & fyear > 1975 & ltotal_dollars> 0, absorb(fyear) cluster(county_fips) resid
predict resid_sp, resid

* Now, plot the residuals:
twoway scatter resid_lt resid_sp || lfit resid_lt resid_sp , ///
    title("Partial Relationship: ltotal_dollars vs. spending_iv") ///
    ytitle("Residuals of ltotal_dollars") ///
    xtitle("Residuals of spending_iv")	
	
twoway scatter ltotal_dollars lspending_6675_iv, msize(tiny) || lfit ltotal_dollars lspending_6675_iv, ///
	xtitle("IV") ///
	ytitle("log(spending)") ///
	 legend(off)

graph export "$path/output/figures/descriptives/iv.png", replace

restore

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
