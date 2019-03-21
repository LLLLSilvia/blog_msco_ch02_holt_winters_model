import pandas as pd

# This starts a timer
import time
start = time.time()

# This ingests just the period and demand data, for the 13 periods
df = pd.read_excel('filepath/chapter_2_instruction.xlsx', sheet_name='holt_winters', header=4, index_col=None, nrows=8, usecols=[0,1])

# This renames the columns using a dict
df = df.rename(columns={'t':'Period','Dt':'Demand'})

# These rows calculate various factors used to calculate seasonality
period_count = len(df['Period'])
cycle_count = 2 # At least one of (cycle_count, season_count_per_cycle) has to be known
season_count_per_cycle = int(round((period_count/cycle_count)+0.5,0))

# This creates lists for raw and average seasonality, of the appropriate legnth
raw_seasonality=[None for i in range(0, period_count)]
avg_seasonality=[None for i in range(0, season_count_per_cycle)]

# This creates a base forecast
import statsmodels.api as sm
X = sm.add_constant(df['Period'])
results = sm.OLS(df['Demand'], X).fit()
df['OLS_forecast'] = pd.DataFrame(results.predict(X))

# This calculates the raw seasonality for each period and replaces the None in the list with the proper value
for i in range(0, period_count):
    raw_seasonality[i] = df['Demand'][i]/df['OLS_forecast'][i]

# This averages the raw seasonality across all cycles to get the average seasonality and replaces the None in the list with the proper value
for i in range(0, season_count_per_cycle):
    seasonality_factor = 0
    for q in range(0,cycle_count):
        seasonality_factor = raw_seasonality[i+(q*season_count_per_cycle)] + seasonality_factor
        if q == cycle_count-1:
            avg_seasonality[i] = seasonality_factor/(q+1)

df['base_seas_fac'] = None
df['SA_OLS_forecast'] = None
# This adds in a Seas Fac and adjusts said Seas Fac for each season
for i in range (0, period_count):
    season = i
    if season < season_count_per_cycle:
        df['base_seas_fac'][season] = avg_seasonality[season]
    else:
        while season >= season_count_per_cycle:
            season = season - season_count_per_cycle
        df['base_seas_fac'][i] = avg_seasonality[season]
    df['SA_OLS_forecast'] = df['OLS_forecast']*df['base_seas_fac']


# So now we have the seasonality factors which will can use for our predictions
# Let's set the initial base and growth we'll take from the OLS model
holt_winters_base = results.params[0]
holt_winters_growth = results.params[1]

# Now let's define the model *with HW_base and HW_growth as inputs)
def holtWintersModel(period, HW_base, HW_growth, alpha, beta, gamma):
    global holt_winters_persisted_list
    global avg_seasonality
    holt_winters_persisted_list = [['Period','HW Base','HW growth','HW unadj Ft for t+1', 'HW Seas Fac','HW Ft for t+1'],[1,HW_base,HW_growth, HW_base+HW_growth*2,avg_seasonality[0],(HW_base+HW_growth*2)*avg_seasonality[1]]]
    for i in range(1, period+1):
        if i == 1:
            holt_winters_persisted_list[i][3]
        else:
            current_demand = df['Demand'][i-1]
            last_base = holt_winters_persisted_list[i-1][1]
            last_growth = holt_winters_persisted_list[i-1][2]
            last_forecast = holt_winters_persisted_list[i-1][5]
            if i-1 < season_count_per_cycle:
                new_seasonal_factor = df['base_seas_fac'][i]
                last_seasonal_factor = df['base_seas_fac'][i-1]
            else:
                new_seasonal_factor = gamma*(current_demand/new_base) + (1-gamma)*df['base_seas_fac'][i-season_count_per_cycle]
                avg_seasonality.append(new_seasonal_factor)
                last_seasonal_factor = df['base_seas_fac'][i-1-season_count_per_cycle]
            new_base = alpha*(current_demand/last_seasonal_factor)+((1-alpha)*(last_base+last_growth))
            new_growth = beta*(new_base-last_base)+(1-beta)*last_growth
            holt_winters_persisted_list.append([i,new_base,new_growth,new_base+new_growth*2,new_seasonal_factor,(new_base+new_growth*2)*new_seasonal_factor])
    return holt_winters_persisted_list

# Only the call to holtWintersModel() changes when adding HW_base and HW_growth, as compared to before
mse_max = len(df['Period'])
def holtWintersMSE(x):
    global holt_winters_persisted_list
    sse_val = 0
    sse_count = 0
    holtWintersModel(mse_max, x[0], x[1], x[2], x[3], x[4])
    for i in range(0,mse_max-1):
        global holt_winters_persisted_list
        holt_winters_forecast = holt_winters_persisted_list[i+1][5]
        current_demand = df['Demand'][i+1]
        sse_val = (holt_winters_forecast-current_demand)**2 + sse_val
        sse_count += 1
    mse_val = sse_val/sse_count
    return mse_val

# Let's set the initial base, growth, α,β,γ values
test_alpha = 0.5
test_beta = 0.5
test_gamma = 0.5
initial_guess = [holt_winters_base, holt_winters_growth, test_alpha, test_beta, test_gamma]

# And now we optimize the MSE with scipy.optimize.minimize
from scipy.optimize import minimize
result = minimize(holtWintersMSE, initial_guess, bounds=((0, None), (0, None), (0,1),(0,1),(0,1)))

# This calculates the MSE for the seasonally adjusted OLS forecast
SA_OLS_MSE = pd.DataFrame([(df['SA_OLS_forecast'][i]-df['Demand'][i])**2 for i in range(1,len(df['Demand']))]).mean()[0]

# This prints results
print("\nOptimized Base:\n",result.x[0], "\nOptimized Growth:\n",result.x[1], "\nOptimized alpha:\n",result.x[2],"\nOptimized beta:\n",result.x[3],"\nOptimized gamma:\n",result.x[4],"\nSA OLS Mean Squared Error\n", SA_OLS_MSE, "\nHolt Winters' Mean Squared Error\n", holtWintersMSE(result.x))

# # And stop the timer
# end = time.time()
# print("\n Seconds:\n",end - start)

# This creates a dataframe with all of the information
holt_winters_df = pd.DataFrame(holt_winters_persisted_list[1:], columns=holt_winters_persisted_list[0])
holt_winters_df['HW unadj Ft for t+1'] = holt_winters_df['HW unadj Ft for t+1'].shift(1)
holt_winters_df['HW Seas Fac'] = holt_winters_df['HW Seas Fac'].shift(1)
holt_winters_df['HW Ft for t+1'] = holt_winters_df['HW Ft for t+1'].shift(1)
holt_winters_df = holt_winters_df.rename(columns={'HW unadj Ft for t+1' : 'HW unadj Ft', 'HW Ft for t+1' : 'HW Ft'})
new_df = pd.merge(df, holt_winters_df, how='outer', on=['Period'])

# # And this prints it
# print("\n",new_df)

# This is a statistical test. If we observe a large p-value, for example larger than 0.05 or 0.1, then we cannot reject the null hypothesis of identical average scores. 
from scipy import stats
ttest_results = stats.ttest_ind((new_df['HW Ft'][1:]-new_df['Demand'][1:])**2, (new_df['SA_OLS_forecast'][1:]-new_df['Demand'][1:])**2)
print("\nT-test results:\n",ttest_results)