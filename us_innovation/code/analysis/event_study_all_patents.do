clear all

global path "GitHub-portfolio/us_innovation"

global output "$path/output/figures"

use "$path/data/processed/built.dta"

merge 1:1 county_fips fyear using "$path/data/processed/defcon_patent_merge_county_year_final.dta"

drop if fyear < 1965 | fyear > 2003

drop county_id

bysort county_fips: egen semi_intens_1 = max(semi_intens)
drop semi_intens
rename semi_intens_1 semi_intens

keep if semi_intens == 1 

egen county_id = group(county_fips)

*----------------------------------------------------
* 1. Compute county-level mean spending for each period.
*    We use the cond() function to calculate the mean 
*    only for the years in each period.
bysort county_fips: egen mean_spend1 = mean(cond(inrange(fyear,1976,1981), total_dollars, .))
bysort county_fips: egen mean_spend2 = mean(cond(inrange(fyear,1981,1989), total_dollars, .))

*----------------------------------------------------
* 2. Define the surge as the difference between period means.
gen surge = mean_spend2 - mean_spend1

* 2. Calculate percentage change (surge)
//gen surge = 100 * (mean_spend2 - mean_spend1) / mean_spend1

*----------------------------------------------------
* 3. Establish a threshold for a "large surge"
*    Here we use the overall mean plus one standard deviation 
*    (across counties with valid surge data) as the cutoff.
summarize surge, d

scalar surge_med = r(p50) 
scalar surge_mean = r(mean)
scalar surge_thresh2 = r(mean) + r(sd)/2
scalar surge_thresh = r(mean) + r(sd)

display "Surge threshold = " surge_thresh

*----------------------------------------------------
* 4. Create the treatment indicator.
*    Counties with a surge above the threshold are treated.

//bysort county_fips: gen treated = surge > surge_thresh
//bysort county_fips: gen treated = (surge > surge_mean)
bysort county_fips: gen treated = (surge > surge_med)


**** D-in-D

gen after = (fyear > 1981)
gen treatment = after * treated


 * Run DiD regression

xtdidregress (num_patents) (treatment), group(county_id) time(fyear)

estat ptrends // test

* Generate the trend plot
estat trendplot, ///
    ytitle("Patent Count") ///
    xtitle("Year") ///
	xlabel(1965(5)2005) ///
    legend(label(1 "Treated") label(2 "Control")) ///
	notitle
	
// spec that works: treat = abs/89/med, 1965-2003
	
graph export "$output/trendplot_num_final.png", replace
	
// no table 	
estat grangerplot, base(1981) ///
	xtitle("year") 

	
graph export "$output/grangerplot_num_final.png", replace

outreg2 using "$output/grangerplot_num_final.tex", lab dec(3) nocons ///
		title(Event Study Granger Plot) ctitle(Number of Patents)
erase "$output/grangerplot_num_final.txt"

