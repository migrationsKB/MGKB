cd "C:\Users\heidland\Dropbox (MEDAM)\H2020 - ITFLOWS\Research\WP5\D5.4"

*** Prepare Twitter data
local data="hsd sentiment"
	foreach data in `data'{
	import delimited `data'_country_01.csv, clear
	rename v1 year 

local ccodes="gb de se fr it gr es at hu ch pl nl be bg cz dk hr cy ee fi ie lv lt lu mt pt ro sk si is li no"

	foreach var in `ccodes'{
		rename `var' country_`var'
		}
		
	reshape long country_, i(year `data') j(country) string
	rename country_ value

	reshape wide value, i(country year) j(`data') string

	kountry country, from(iso2c)
	replace country=NAMES_STD
	tab country, mi
	drop if country=="avg"
	drop NAMES_STD
	save `data'_prepared.dta, replace
	}

	
* Unemployment rate
import delimited total_unemployment.csv, varnames(1) encoding(ISO-8859-2) clear 
drop sex unit
kountry geo, from(iso2c)
rename NAMES_STD country
tab country, mi
drop if country=="avg"

forval y=1/8{
	local i=`y'+2012
	rename v`y' value`i'    
}

reshape long value, i(country) j(year)
rename value total_unemployment 
lab var total_unemployment "Total unemployment rate (15-74)"
keep country year total_unemployment
save total_unemployment.dta, replace


* Real GDP growth rate
import delimited real_gdp_r.csv, varnames(1) encoding(ISO-8859-2) clear
kountry geo, from(iso2c)
rename NAMES_STD country
tab country, mi
drop if country=="avg"

destring v*, force replace // Forces all observations to string, will cause missings in 2021 but will allow loop because all v* not numeric
forval y=1/9{
	local i=`y'+2012
	rename v`y' value`i'    
}

reshape long value, i(country) j(year)
rename value real_gdp_r
lab var real_gdp_r "Real GDP growth rate"
keep country year real_gdp_r
save real_gdp_r.dta, replace



* Immigrant groups (stocks)
import delimited "migr_pop1ctz_linear", varnames(1) clear 
keep if time>=2013
rename time_period year

keep if age=="TOTAL"&sex=="T"

* We keep only immigrants from EU and non-EU (plus EU28 as Croatians+EU27 for the year 2013)
keep if (year==2013&(cit=="EU27_FOR"|cit=="NEU27_FOR"|cit=="HR"))|cit=="EU28_FOR"|cit=="NEU28_FOR" 
gen aux=obs_value if year==2013&cit=="HR"
bysort geo year: egen aux2=total(aux)
replace obs_value=obs_value+aux2 if cit=="EU27_FOR"&year==2013
replace obs_value=obs_value-aux2 if cit=="NEU27_FOR"&year==2013
replace cit="EU28_FOR" if cit=="EU27_FOR"&year==2013
replace cit="NEU28_FOR" if cit=="NEU27_FOR"&year==2013

keep if cit=="EU28_FOR"|cit=="NEU28_FOR"

reshape wide obs_value, i(geo year) j(cit) string

rename obs_valueEU28 EU_migrant_stock
rename obs_valueNEU28 NonEU_migrant_stock

kountry geo, from(iso2c)
rename NAMES_STD country
replace country="Greece" if country=="el" // Eurostat uses a non-ISO variant for the country code for Greece >_<
tab country, mi

keep country year *stock
save migr_stock.dta, replace


* Immigration flow
import delimited migr_imm8.csv, varnames(1) encoding(ISO-8859-2) clear
kountry geo, from(iso2c)
rename NAMES_STD country
replace country="Greece" if country=="el" // Eurostat uses a non-ISO variant for the country code for Greece >_<
tab country, mi

rename time_period year
keep if year>=2013

rename obs_value immigration_flow
lab var immigration_flow "Total yearly immigration flow"
* We do not care about the gender of migrants here, hence keep only total migrants
keep if sex=="T"
* Keep ony variables we need
keep country year immigration_flow 
save migr_immigflow.dta, replace


* Asylum applications
import delimited migr_asyappctza.csv, varnames(1) encoding(ISO-8859-2) clear
kountry geo, from(iso2c)
rename NAMES_STD country
replace country="Greece" if country=="el" // Eurostat uses a non-ISO variant for the country code for Greece >_<
tab country, mi
rename time_period year
keep if year>=2013
* We only care about asylum applicants regardless of flows here and all age groups
keep if sex=="T"&age=="TOTAL"
* We only keep first time asylum applicants here
keep if asyl_app=="NASY_APP"
rename obs_value asylum_applications
lab var asylum_applications "Yearly first-time asylum applications"
keep country year asylum_applications
save migr_asylumapp.dta, replace




* Population
import delimited Population.csv, varnames(1) encoding(ISO-8859-2) clear
kountry geo, from(iso2c)
rename NAMES_STD country
replace country="Greece" if country=="el" // Eurostat uses a non-ISO variant for the country code for Greece >_<
tab country, mi
rename time_period year
keep if year>=2013
* We only care about total population here because we need the variable for standardizing other variables
keep if sex=="T"
rename obs_value total_population
lab var total_population "Total population"
keep country year total_population 
save total_population.dta, replace





*** Merge datasets
use hsd_prepared, clear
merge 1:1 country year using sentiment_prepared.dta, gen(sentiment_merge)
* complete match of sentiments
merge 1:1 country year using total_unemployment.dta, gen(unemployment_merge)
tab country year if unemployment_merge==1
* Drop Liechtenstein because we do not have the control variables --->  ADRESS THIS IN FUTURE VERSIONS? Liechtenstein not in EU though
drop if country=="Liechtenstein"
merge 1:1 country year using real_gdp_r.dta, gen(real_gdp_r_merge)
tab year if real_gdp_r_merge!=3
* Incomplete matches are 2021 missing obs -> can be safely dropped from dataset because we do not have sentiment dataset
drop if valueNegative==.
tab real_gdp_r_merge


merge 1:1 country year using migr_immigflow.dta, gen(immigflow_merge)
tab country year if immigflow_merge!=3
* Here, were missing the 2020 data! -> should be updating later
merge 1:1 country year using migr_asylumapp.dta, gen(asylumapp_merge)

merge 1:1 country year using total_population.dta, gen(population_merge)

merge 1:1 country year using migr_stock.dta, gen(migrstock_merge)


* Standardize the dataset to only keep the observations with sentiment data
keep if valueNegative!=.|valueoffensive!=.

tab country year

*** WE SHOULD REALLY ADD IMMIGRANT STOCK AND GROWTH RATE OR NUMBER OF INCOMING ASYLUM SEEKERS HERE!
* Generate outcome variables
gen share_offensive=valueoffensive/(valuenormal+valueoffensive)
lab var share_offensive "Share of offensive posts"
gen share_negative=valueNegative/(valueNegative+valueNeutral+valuePositive)
lab var share_negative "Share of posts with negative sentiment"
gen share_positive=valuePositive/(valueNegative+valueNeutral+valuePositive)
lab var share_positive "Share of posts with positive sentiment"


* Generate explantory variables
gen asyl_per100k=asylum_applications/(total_population/100000)
lab var asyl_per100k "Yearly first-time asylum application per 100k population"

gen immigrationflow_per100k=immigration_flow/(total_population/100000)
lab var immigrationflow_per100k "Yearly immigration flow per 100k population"

gen total_migrant_stock=EU_migrant_stock+NonEU_migrant_stock
lab var total_migrant_stock "Total migrant stock"

gen migrant_stock_per100k=total_migrant_stock/(total_population/100000)
lab var migrant_stock_per100k "Total migrant stock per 100k population"

gen nonEU_stock_per100k=NonEU_migrant_stock/(total_population/100000)
lab var nonEU_stock_per100k "Non-EU migrant stock per 100k population"


drop *merge
kountry country, from(other) stuck
rename _ISO3N_ iso3n
save working_dataset.dta, replace
*also export as csv
export delimited using "working_dataset.csv", replace


********************************************************************
*** Analyis starts here
use working_dataset.dta, clear



xtset iso3n year

gen asyl_per100k_sq=asyl_per100k^2
gen asyl_x_unemployment=asyl_per100k*total_unemployment

* Correlation table
pwcorr share_negative share_positive share_offensive total_unemployment real_gdp_r immigrationflow_per100k asyl_per100k migrant_stock_per100k, star(0.05)


* Set control variables
global controls="migrant_stock_per100k total_unemployment real_gdp_r"

* Table 2
reg share_negative asyl_per100k $controls, cluster(iso3n)
outreg2 using table2.xls, label replace
reg share_offensive asyl_per100k $controls, cluster(iso3n)
outreg2 using table2.xls, label append
*reg share_negative asyl_per100k $controls asyl_x_unemployment, cluster(iso3n) ---> THIS GIVES A WEAKLY SIGNIF COEFFICIENT BUT IT DOES NOT MAKE SENSE, HENCE EXCLUDE FOR THE MOMENT
*outreg2 using table2.xls, label replace
*reg share_offensive asyl_per100k $controls asyl_x_unemployment, cluster(iso3n)
*outreg2 using table2.xls, label append
reg share_negative immigrationflow_per100k $controls, cluster(iso3n)
outreg2 using table2.xls, label append
reg share_offensive immigrationflow_per100k $controls, cluster(iso3n)
outreg2 using table2.xls, label append

** Comment from rewiever: Does the pattern in the table above regarding asylum seekers also hold if we use a lagged specification?
* Table 2_lagged
* We lag all explanatory variables by using the forward value of the outcome variable
reg f.share_negative asyl_per100k $controls, cluster(iso3n)
outreg2 using table2_lagged.xls, label replace
reg f.share_offensive asyl_per100k $controls, cluster(iso3n)
outreg2 using table2_lagged.xls, label append
reg f. share_negative immigrationflow_per100k $controls, cluster(iso3n)
outreg2 using table2_lagged.xls, label append
reg f. share_offensive immigrationflow_per100k $controls, cluster(iso3n)
outreg2 using table2_lagged.xls, label append
 



// Does the effect depend on previous experience with migrants?
* Table 3: cutoff at the median of migrantstock/non-EU-Stock respectively
qui: sum migrant_stock_per100k, det
local median=r(p50)
di `median'
reg share_negative asyl_per100k $controls if migrant_stock_per100k<=`median', cluster(iso3n)
outreg2 using table3.xls, label replace
reg share_negative asyl_per100k $controls if migrant_stock_per100k>=`median', cluster(iso3n)
outreg2 using table3.xls, label append
qui: sum nonEU_stock_per100k, det
local median=r(p50)
di `median'
reg share_negative asyl_per100k $controls if nonEU_stock_per100k<=`median', cluster(iso3n)
outreg2 using table3.xls, label append
reg share_negative asyl_per100k $controls if nonEU_stock_per100k>=`median', cluster(iso3n)
outreg2 using table3.xls, label append




// Getting rid of time-invariant difference (and yearly EU-wide effects)
* Table 4: First difference estimation (country FEs)
gen trend=year-2013
tab country, gen(ctryfx)
forval i=1/31{
	gen ctry_time_trend`i'=ctryfx`i'*trend
	}
	
xtreg share_negative asyl_per100k $controls, cluster(iso3n) fe
outreg2 using table4.xls, label replace
xtreg share_offensive asyl_per100k $controls, cluster(iso3n) fe
outreg2 using table4.xls, label append
xtreg share_negative asyl_per100k $controls i.year, cluster(iso3n) fe
outreg2 using table4.xls, label append
xtreg share_offensive asyl_per100k $controls i.year, cluster(iso3n) fe
outreg2 using table4.xls, label append
xtreg share_negative asyl_per100k $controls ctry_time_trend*, cluster(iso3n) fe
outreg2 using table4.xls, label append
xtreg share_offensive asyl_per100k $controls ctry_time_trend*, cluster(iso3n) fe
outreg2 using table4.xls, label append

 
