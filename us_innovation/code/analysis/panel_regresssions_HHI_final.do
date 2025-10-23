clear all 

global path "GitHub-portfolio/us_innovation"

global output "$path/output/figures"

use "$path/data/processed/built5.dta", clear

* Merge distinct_firms data

merge m:1 county_fips using "$path/data/processed/hhi_aggregate_all1_pre80.dta"

keep if _merge == 3
drop _merge

egen state_yr = group(state fyear)

*** select sample

keep if semi_intens == 1

/*
bysort county_fips: gen patent = (num_patents>0 & fyear < 1976)
bysort county_fips: egen patent_yrs = sum(patent) 

gen intensive = (patent_yrs >= 1)

keep if intensive == 1
*/

***

sort comp_wtd 
gen group = ceil(4 * _n/_N)

* 1) Create group dummies
tabulate group, generate(g)    // creates g1…g4

* 2) Create the "treatment × group" and "instrument × group" terms
foreach i of numlist 1/4 {
    gen xg`i' = ltotal_dollars * g`i'
    gen zg`i' = log_spending_iv5 * g`i'
}

//ivreghdfe lw_cites_sub (xg1 xg2 xg3 xg4 = zg1 zg2 zg3 zg4), abs(fyear county_fips) cluster(county_fips) 

//ivreghdfe lw_cites_sub avg_wages  pop emp (xg1 xg2 xg3 xg4 = zg1 zg2 zg3 zg4), abs(fyear county_fips) cluster(county_fips) 

ivreghdfe lw_cites_sub avg_wages pop emp (xg1 xg2 xg3 xg4 = zg1 zg2 zg3 zg4), abs(fyear county_fips state_yr) cluster(county_fips) 	

estimates store iv1

/// Baseline

* Manually define coefficient and standard error
matrix b_extra = (0.1122621)
matrix se_extra = (0.0369543)

* Assign names so coefplot knows what it's referring to
matrix colnames b_extra = xg5
matrix colnames se_extra = xg5

coefplot iv1 ///
    (matrix(b_extra) , se(se_extra) label("Group 5")) ///
    , keep(xg1 xg2 xg3 xg4 xg5) ///
    vertical ///
    ci ///
    ciopts(recast(rcap) lwidth(medium)) ///
    xline(0) ///
    xlabel(1 "q1" 2 "q2" 3 "q3" 4 "q4" 5 "Baseline", angle(0)) ///
    xtitle("Group") ///
    ytitle("Estimated Treatment Effect") ///
    legend(off)
	
graph export "$output/hhi.png", as(png) replace

