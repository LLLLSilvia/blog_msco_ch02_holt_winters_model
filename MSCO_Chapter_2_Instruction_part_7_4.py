import pandas as pd

# This starts a timer
import time
start = time.time()

# This ingests just the period and demand data, for the 13 periods
full_df = pd.read_excel('filepath/chapter_2_instruction.xlsx', sheet_name='FRED_Graph', header=10, index_col=None, usecols=[0,1])

# This takes the last 20 full years
df = full_df[-83:-3]

# This creates a period for the dataframe, so I don't have to mess with pandas datetime64[ns] dtype
df['Period'] = [i for i in range(0,len(df['observation_date']))]

# This renames the columns using a dict
df = df.rename(columns={'HOUST1FQ':'Demand'})

# These rows calculate various factors used to calculate seasonality
period_count = len(df['Period'])
cycle_count = 20 # At least one of (cycle_count, season_count_per_cycle) has to be known
season_count_per_cycle = int(round((period_count/cycle_count)+0.5,0))

# This creates lists for raw and average seasonality, of the appropriate legnth
raw_seasonality=[None for i in range(0, period_count)]
avg_seasonality=[None for i in range(0, season_count_per_cycle)]

df = df.reset_index(drop=True)

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
print(results.params)
holt_winters_base = results.params[0]
holt_winters_growth = results.params[1]

# Now let's define the model 
def holtWintersModel(period, alpha, beta, gamma):
    global holt_winters_persisted_list
    global avg_seasonality
    holt_winters_persisted_list = [['Period','HW Base','HW growth','HW unadj Ft for t+1', 'HW Seas Fac','HW Ft for t+1'],[1,holt_winters_base,holt_winters_growth, holt_winters_base+holt_winters_growth*2,avg_seasonality[1],(holt_winters_base+holt_winters_growth*2)*avg_seasonality[1]]]
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

# We can use a very similar MSE function
mse_max = len(df['Period'])
def holtWintersMSE(x):
    global holt_winters_persisted_list
    sse_val = 0
    sse_count = 0
    holtWintersModel(mse_max, x[0], x[1], x[2])
    for i in range(0,mse_max-1):
        global holt_winters_persisted_list
        holt_winters_forecast = holt_winters_persisted_list[i+1][5]
        current_demand = df['Demand'][i+1]
        sse_val = (holt_winters_forecast-current_demand)**2 + sse_val
        sse_count += 1
    mse_val = sse_val/sse_count
    return mse_val

# Let's set the initial α,β,γ values
test_alpha = 0.5
test_beta = 0.5
test_gamma = 0.5
initial_guess = [test_alpha, test_beta, test_gamma]

# And now we optimize the MSE with scipy.optimize.minimize
from scipy.optimize import minimize
result = minimize(holtWintersMSE, initial_guess, bounds=((0,1),(0,1),(0,1)))

# This creates a dataframe with all of the information
holt_winters_df = pd.DataFrame(holt_winters_persisted_list[1:], columns=holt_winters_persisted_list[0])
# holt_winters_df['HW unadj Ft for t+1'] = holt_winters_df['HW unadj Ft for t+1'].shift(-1)
# holt_winters_df['HW Seas Fac'] = holt_winters_df['HW Seas Fac'].shift(-1)
# holt_winters_df['HW Ft for t+1'] = holt_winters_df['HW Ft for t+1'].shift(-1)
holt_winters_df = holt_winters_df.rename(columns={'HW unadj Ft for t+1' : 'HW unadj Ft', 'HW Ft for t+1' : 'HW Ft'})
new_df = pd.merge(df, holt_winters_df, how='outer', on=['Period'])
new_df = new_df[0:80] 

# This calculates the MSE for the seasonally adjusted OLS forecast
SA_OLS_MSE = pd.DataFrame([(df['SA_OLS_forecast'][i]-df['Demand'][i])**2 for i in range(1,len(df['Demand']))]).mean()[0]

# This prints results
print("\nOptimized alpha:\n",result.x[0],"\nOptimized beta:\n",result.x[1],"\nOptimized gamma:\n",result.x[2],"\nSA OLS Mean Squared Error\n", SA_OLS_MSE, "\nHolt Winters' Mean Squared Error\n", holtWintersMSE(result.x))
    
# And this prints it
print("\n",new_df)

# Now let's define the model *with HW_base and HW_growth as inputs)
def holtWintersModel_2(period, HW_base, HW_growth, alpha, beta, gamma):
    global holt_winters_persisted_list_2
    global avg_seasonality
    holt_winters_persisted_list_2 = [['Period','HW Base','HW growth','HW unadj Ft for t+1', 'HW Seas Fac','HW Ft for t+1'],[1,HW_base,HW_growth, HW_base+HW_growth*2,avg_seasonality[0],(HW_base+HW_growth*2)*avg_seasonality[1]]]
    for i in range(1, period+1):
        if i == 1:
            holt_winters_persisted_list_2[i][3]
        else:
            current_demand = df['Demand'][i-1]
            last_base = holt_winters_persisted_list_2[i-1][1]
            last_growth = holt_winters_persisted_list_2[i-1][2]
            last_forecast = holt_winters_persisted_list_2[i-1][5]
            if i-1 < season_count_per_cycle:
                new_seasonal_factor = df['base_seas_fac'][i]
                last_seasonal_factor = df['base_seas_fac'][i-1]
            else:
                new_seasonal_factor = gamma*(current_demand/new_base) + (1-gamma)*df['base_seas_fac'][i-season_count_per_cycle]
                avg_seasonality.append(new_seasonal_factor)
                last_seasonal_factor = df['base_seas_fac'][i-1-season_count_per_cycle]
            new_base = alpha*(current_demand/last_seasonal_factor)+((1-alpha)*(last_base+last_growth))
            new_growth = beta*(new_base-last_base)+(1-beta)*last_growth
            holt_winters_persisted_list_2.append([i,new_base,new_growth,new_base+new_growth*2,new_seasonal_factor,(new_base+new_growth*2)*new_seasonal_factor])
    return holt_winters_persisted_list_2

# Only the call to holtWintersModel() changes when adding HW_base and HW_growth, as compared to before
mse_max = len(df['Period'])
def holtWintersMSE_2(x):
    global holt_winters_persisted_list_2
    sse_val = 0
    sse_count = 0
    holtWintersModel_2(mse_max, x[0], x[1], x[2], x[3], x[4])
    for i in range(0,mse_max-1):
        global holt_winters_persisted_list_2
        holt_winters_forecast = holt_winters_persisted_list_2[i+1][5]
        current_demand = df['Demand'][i+1]
        sse_val = (holt_winters_forecast-current_demand)**2 + sse_val
        sse_count += 1
    mse_val = sse_val/sse_count
    return mse_val

# Let's set the initial base, growth, α,β,γ values
initial_guess_2 = [holt_winters_base, holt_winters_growth, test_alpha, test_beta, test_gamma]

# And now we optimize the MSE with scipy.optimize.minimize
from scipy.optimize import minimize
result = minimize(holtWintersMSE_2, initial_guess_2, bounds=((0, None), (None, None), (0,1),(0,1),(0,1)))

# This calculates the MSE for the seasonally adjusted OLS forecast
SA_OLS_MSE = pd.DataFrame([(df['SA_OLS_forecast'][i]-df['Demand'][i])**2 for i in range(1,len(df['Demand']))]).mean()[0]

# This prints results
print("\nOptimized Base:\n",result.x[0], "\nOptimized Growth:\n",result.x[1], "\n\nOptimized alpha:\n",result.x[2],"\nOptimized beta:\n",result.x[3],"\nOptimized gamma:\n",result.x[4],"\nSA OLS Mean Squared Error\n", SA_OLS_MSE, "\nHolt Winters' Mean Squared Error\n", holtWintersMSE(result.x))

# This creates a df with all the information
holt_winters_df_2 = pd.DataFrame(holt_winters_persisted_list_2[1:], columns=holt_winters_persisted_list_2[0])
# holt_winters_df_2['HW unadj Ft for t+1'] = holt_winters_df_2['HW unadj Ft for t+1'].shift(-1)
# holt_winters_df_2['HW Seas Fac'] = holt_winters_df_2['HW Seas Fac'].shift(-1)
# holt_winters_df_2['HW Ft for t+1'] = holt_winters_df_2['HW Ft for t+1'].shift(-1)
holt_winters_df_2 = holt_winters_df_2.rename(columns={'HW unadj Ft for t+1' : 'HW unadj Ft', 'HW Ft for t+1' : 'HW Ft'})
new_df_2 = pd.merge(df, holt_winters_df_2, how='outer', on=['Period'])
new_df_2 = new_df_2[0:80] 

# And this prints it
print("\n",new_df_2)

# This is a statistical test. If we observe a large p-value, for example larger than 0.05 or 0.1, then we cannot reject the null hypothesis of identical average scores. 
from scipy import stats
ttest_results = stats.ttest_ind((new_df['HW Ft'][1:]-new_df['Demand'][1:])**2, (new_df['SA_OLS_forecast'][1:]-new_df['Demand'][1:])**2)
print("\nT-test results:\n",ttest_results)

# This combines both datasets, drops extra columns, and renames some columns
new_df_3 = pd.merge(new_df, new_df_2, how='outer', on=['Period','Demand','OLS_forecast','base_seas_fac','SA_OLS_forecast','observation_date'])
new_df_3 = new_df_3.drop(columns=['base_seas_fac','HW Base_x','HW growth_x','HW unadj Ft_x','HW Seas Fac_x','HW Base_y','HW growth_y','HW unadj Ft_y','HW Seas Fac_y'])
new_df_3 = new_df_3.rename(columns={'HW Ft_x':'Three-variable HW Ft','HW Ft_y':'Five-variable HW Ft'})
print("\n\n",new_df_3)

# And stop the timer
end = time.time()
print("\n Seconds:\n",end - start)

# Now let's plot the demand and forecasts together
import matplotlib.pyplot as plt
plt.plot(new_df_3['observation_date'],new_df_3['Demand'], color='blue', linewidth=3.0, label='Actual Demand')
plt.plot(new_df_3['observation_date'],new_df_3['OLS_forecast'], color='red', linestyle=':', label='Unadjusted OLS Regression')
plt.plot(new_df_3['observation_date'],new_df_3['SA_OLS_forecast'], color='green', linestyle='-.', label='Seasonally Adjusted OLS Regression')
plt.xlabel('Period')
plt.ylim(0,500)
plt.ylabel('Demand')
plt.title('Housing starts by quarter, last 20 full years (1997-2017)')
plt.legend()
plt.show()

plt.plot(new_df_3['observation_date'],new_df_3['Demand'], color='blue', linewidth=3.0, label='Actual Demand')
plt.plot(new_df_3['observation_date'],new_df_3['OLS_forecast'], color='red', linestyle=':', label='Unadjusted OLS Regression')
plt.plot(new_df_3['observation_date'],new_df_3['Three-variable HW Ft'], color='grey', linestyle='--', label='Three-variable Holt-Winters\' Forecast')
plt.xlabel('Period')
plt.ylim(0,500)
plt.ylabel('Demand')
plt.title('Housing starts by quarter, last 20 full years (1997-2017)')
plt.legend()
plt.show()

plt.plot(new_df_3['observation_date'],new_df_3['Demand'], color='blue', linewidth=3.0, label='Actual Demand')
plt.plot(new_df_3['observation_date'],new_df_3['Three-variable HW Ft'], color='grey', linestyle='--', label='Three-variable Holt-Winters\' Forecast')
plt.plot(new_df_3['observation_date'],new_df_3['Five-variable HW Ft'], color='goldenrod', linestyle=':', label='Five-variable Holt-Winters\' Forecast')
plt.xlabel('Period')
plt.ylim(0,500)
plt.ylabel('Demand')
plt.title('Housing starts by quarter, last 20 full years (1997-2017)')
plt.legend()
plt.show()

plt.plot(new_df_3['observation_date'],new_df_3['Demand'], color='blue', linewidth=3.0, label='Actual Demand')
plt.plot(new_df_3['observation_date'],new_df_3['OLS_forecast'], color='red', linestyle=':', label='Unadjusted OLS Regression')
plt.plot(new_df_3['observation_date'],new_df_3['SA_OLS_forecast'], color='green', linestyle='-.', label='Seasonally Adjusted OLS Regression')
plt.plot(new_df_3['observation_date'],new_df_3['Three-variable HW Ft'], color='grey', linestyle='--', label='Three-variable Holt-Winters\' Forecast')
plt.plot(new_df_3['observation_date'],new_df_3['Five-variable HW Ft'], color='goldenrod', linestyle=':', label='Five-variable Holt-Winters\' Forecast')
plt.xlabel('Period')
plt.ylim(0,500)
plt.ylabel('Demand')
plt.title('Housing starts by quarter, last 20 full years (1997-2017)')
plt.legend()
plt.show()