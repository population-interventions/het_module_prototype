import numpy as np
import pandas as pd
from scipy import optimize
from matplotlib import pyplot as plt
from itertools import chain
import pdb, traceback, sys


def disaggregate_cohort_demography(cohort_data):
    
    #Used for testing a sample of cohorts
    cohort_end = cohort_data['cohort'].max() + 1
    #cohort_end = 3

    #assume each cohort has same number of subpops
    sub_pops = cohort_data.iloc[0]['subPops']

    output_cols = ['cohort', 'age', 'sex', 'year',
                   'mortality_agg','prev_alive_agg', 'prev_dead_agg', 
                   'alive_agg', 'dead_agg', 'PY_agg', 'YLD_agg', 'HALY_agg',
                   'sub_pops']

    sub_pop_cols = list(chain.from_iterable(('mortality{}'.format(k),
                                             'prev_alive{}'.format(k),
                                             'prev_dead{}'.format(k),
                                             'alive{}'.format(k),
                                             'dead{}'.format(k),
                                             'PY{}'.format(k),
                                             'YLD{}'.format(k),
                                             'HALY{}'.format(k))
                                             for k in range(sub_pops)))

    comparison_cols = ['subpop_PY_sum', 'PY_error', 
                       'subpop_HALY_sum', 'HALY_error']

    output_cols += sub_pop_cols + comparison_cols
    output_df = pd.DataFrame(columns=output_cols)

    #loop over cohorts
    for cohort, cohort_df in cohort_data.groupby(['cohort']):
        if cohort >= cohort_end:
            break

        row_0 = cohort_df.iloc[0]
        N = row_0['N']

        age = row_0['age']
        sex = row_0['sex']
        start_year = row_0['year']
        end_year = cohort_df['year'].iloc[-1]
        years = end_year - start_year + 1

        #sub_pops = row_0['subPops']
        sub_pop_prop_cols = ['prop{}'.format(k) for k in range(sub_pops)]
        sub_pop_props = row_0[sub_pop_prop_cols].to_numpy()

        sub_pop_mort_ratio_cols = ['mort_RR{}'.format(k) for k in range(sub_pops)]
        sub_pop_mort_ratios = cohort_df[sub_pop_mort_ratio_cols].to_numpy()
        sub_pop_yld_ratio_cols = ['yld_RR{}'.format(k) for k in range(sub_pops)]
        sub_pop_yld_ratios = cohort_df[sub_pop_yld_ratio_cols].to_numpy()

        init_agg_pop = np.array([N, 0])
        agg_pop = np.zeros([years+1, 2])
        agg_pop[0] = init_agg_pop
        agg_mort_rates = cohort_df['mortality'].to_numpy()
        agg_YLD_rates = cohort_df['yld'].to_numpy()

        for t in range (0, years):
            agg_pop[t+1, 0] = population_state_timestep_2state(agg_pop[t, 0], agg_mort_rates[t])
            agg_pop[t+1, 1] = N - agg_pop[t+1, 0]
        
        agg_PY = compute_PYs(agg_pop[:, 0], agg_pop[:, 1])
        agg_HALY = compute_HALYs(agg_PY, agg_YLD_rates)

        sub_pop_alive, sub_pop_mort = disaggregate_2state(agg_pop[:, 0], 
                                                        agg_mort_rates,
                                                        sub_pop_props,
                                                        sub_pop_mort_ratios,
                                                        years,
                                                        np.zeros(years))

        sub_pop_dead = np.array(N*sub_pop_props - sub_pop_alive, dtype=float)

        sub_pop_PY = compute_PYs(sub_pop_alive, sub_pop_dead)

        #disaggregate yld
        ref_YLD = ((np.sum(sub_pop_PY, axis=1) - agg_HALY)
                    / np.sum(sub_pop_yld_ratios * sub_pop_PY, axis=1)
        )
        sub_pop_YLD =  disaggregate_YLD(agg_HALY, sub_pop_PY, sub_pop_yld_ratios)

        sub_pop_HALY = compute_HALYs(sub_pop_PY, sub_pop_YLD)

        sub_pop_PY_sum = np.sum(sub_pop_PY, axis=1)
        sub_pop_HALY_sum = np.sum(sub_pop_HALY, axis=1)

        cohort_df = pd.DataFrame()
        cohort_df['cohort'] = cohort*np.ones(years)
        cohort_df['age'] = range(age, age +years)
        cohort_df['year'] = range(start_year, end_year+1)
        cohort_df['sex'] = [sex for year in range(start_year, end_year+1)]
        cohort_df['mortality_agg'] = agg_mort_rates
        cohort_df['prev_alive_agg'] = agg_pop[0:-1, 0]
        cohort_df['prev_dead_agg'] = agg_pop[0:-1, 1]
        cohort_df['alive_agg'] = agg_pop[1:, 0]
        cohort_df['dead_agg'] = agg_pop[1:, 1]
        cohort_df['PY_agg'] = agg_PY
        cohort_df['YLD_agg'] = agg_YLD_rates
        cohort_df['HALY_agg'] = agg_HALY
        cohort_df['sub_pops'] = sub_pops

        for k in range(sub_pops):
            cohort_df['mortality{}'.format(k)] = sub_pop_mort[:, k]
            cohort_df['prev_alive{}'.format(k)] = sub_pop_alive[0:-1, k]
            cohort_df['prev_dead{}'.format(k)] = sub_pop_dead[0:-1, k]
            cohort_df['alive{}'.format(k)] = sub_pop_alive[1:, k]
            cohort_df['dead{}'.format(k)] = sub_pop_dead[1:, k]
            cohort_df['PY{}'.format(k)] = sub_pop_PY[:, k]
            cohort_df['YLD{}'.format(k)] = sub_pop_YLD[:, k]
            cohort_df['HALY{}'.format(k)] = sub_pop_HALY[:, k]

        cohort_df['subpop_PY_sum'] = sub_pop_PY_sum
        cohort_df['PY_error'] = agg_PY - sub_pop_PY_sum
        cohort_df['subpop_HALY_sum'] = sub_pop_HALY_sum
        cohort_df['HALY_error'] = agg_HALY - sub_pop_HALY_sum

        output_df = output_df.append(cohort_df, sort=False)

        #make_mortality_pop_plots(agg_pop, sub_pop_alive, sub_pop_dead, years)
        #make_PY_plots(agg_PY, sub_pop_PY, years)
        #make_HALY_plots(agg_HALY, sub_pop_HALY, years)
                            
    #plt.show()
    return output_df


def disaggregate_disease(disease_data):
    
    #Used for testing a sample of cohorts
    cohort_end = cohort_data['cohort'].max() + 1
    #cohort_end = 5

    #assume each cohort has same number of subpops
    sub_pops = disease_data.iloc[0]['subPops']

    output_cols = ['cohort', 'age', 'sex', 'year',
                   'csm_risk_agg',
                   'prev_alive_agg', 
                   'prev_diseased_agg', 
                   'prev_dead_agg',
                   'N', 
                   'alive_agg', 'diseased_agg', 'dead_agg',
                   'prevalence_agg',
                   'i_agg', 'f_agg', 'r_agg', 'DR_agg',
                   'sub_pops']

    sub_pop_cols = list(chain.from_iterable(('csm_risk{}'.format(k),
                                             'N{}'.format(k),
                                             'prev_alive{}'.format(k),
                                             'prev_diseased{}'.format(k),
                                             'prev_dead{}'.format(k),
                                             'alive{}'.format(k),
                                             'diseased{}'.format(k),
                                             'dead{}'.format(k),
                                             'prevalence{}'.format(k),
                                             'i{}'.format(k),
                                             'f{}'.format(k),
                                             'r{}'.format(k),
                                             'DR{}'.format(k))
                                             for k in range(sub_pops)))

    comparison_cols = ['subpop_alive_sum', 'alive_error', 
                       'subpop_diseased_sum', 'diseased_error',
                       'subpop_dead_sum', 'dead_error',
                       'csm_risk_other_weighted_average', 'csm_risk_other_error'
                      ]

    output_cols += sub_pop_cols + comparison_cols
    output_df = pd.DataFrame(columns=output_cols)

    #loop over cohorts
    for cohort, cohort_disease_df in disease_data.groupby(['cohort']):
        if cohort >= cohort_end:
            break

        row_0 = cohort_disease_df.iloc[0]
       
        age = row_0['age']
        sex = row_0['sex']
        start_year = row_0['year']
        end_year = cohort_disease_df['year'].iloc[-1]
        years = end_year - start_year + 1

        N = row_0['N']

        agg_prev = row_0['init_prev']

        init_agg_pop = np.array([N * (1-agg_prev), N * agg_prev, 0])
        agg_pop = np.zeros((years+1, 3))
        agg_pop[0] = init_agg_pop


        agg_i = cohort_disease_df['i'].to_numpy()
        agg_f = cohort_disease_df['f'].to_numpy()
        agg_r = cohort_disease_df['r'].to_numpy()
        agg_DR = cohort_disease_df['DR'].to_numpy()

        #sub_pops = row_0['subPops']
        sub_pop_prop_cols = ['pop_prop{}'.format(k) for k in range(sub_pops)]
        sub_pop_props = row_0[sub_pop_prop_cols].to_numpy()
        init_sub_pop_N = N * sub_pop_props

        sub_pop_prevalence_ratio_cols = ['prevalence_ratio{}'.format(k) for k in range(sub_pops)]
        sub_pop_prevalence_ratios = row_0[sub_pop_prevalence_ratio_cols].to_numpy()
        sub_pop_prevalence_denom = np.sum(sub_pop_prevalence_ratios * init_sub_pop_N)
        sub_pop_prevalence = (init_agg_pop[1]/sub_pop_prevalence_denom)*sub_pop_prevalence_ratios

        init_sub_pops = init_sub_pop_N * np.vstack((1-sub_pop_prevalence, 
                                                    sub_pop_prevalence,
                                                    np.zeros(sub_pops)))

        init_sub_pop_props = np.divide(init_sub_pops, init_agg_pop[:, None],
                                       out=np.zeros_like(init_sub_pops),
                                       where=init_agg_pop[:, None]!=0)
        init_sub_pop_props = init_sub_pop_props.astype(float)

        sub_pop_i_ratio_cols = ['i_RR{}'.format(k) for k in range(sub_pops)]
        sub_pop_i_ratios = cohort_disease_df[sub_pop_i_ratio_cols].to_numpy()
        sub_pop_f_ratio_cols = ['f_RR{}'.format(k) for k in range(sub_pops)]
        sub_pop_f_ratios = cohort_disease_df[sub_pop_f_ratio_cols].to_numpy()
        sub_pop_r_ratio_cols = ['r_RR{}'.format(k) for k in range(sub_pops)]
        sub_pop_r_ratios = cohort_disease_df[sub_pop_r_ratio_cols].to_numpy()

        sub_pop_DR_scale_cols = ['DR_scale{}'.format(k) for k in range(sub_pops)]
        sub_pop_DR_scale = cohort_disease_df[sub_pop_DR_scale_cols].to_numpy()
        sub_pop_DR = agg_DR[:, None] * sub_pop_DR_scale

        csm_risk_agg = np.zeros(years)

        if cohort_disease_df['r'].any():
            #Remission
            pass
        else:
            #No remission
            for t in range (0, years):
                #healthy
                agg_pop[t+1, 0] = population_state_timestep_2state(agg_pop[t, 0], agg_i[t])
                #diseased
                agg_pop[t+1, 1] = (population_state_timestep_2state(agg_pop[t, 1], agg_f[t]) + 
                                   agg_pop[t, 0] - agg_pop[t+1, 0])
                #dead
                agg_pop[t+1, 2] = N - agg_pop[t+1, 0] - agg_pop[t+1, 1]

                #Get risks
                #(D - D_prev) / (S_prev + C_prev)
                csm_risk_agg[t] = (agg_pop[t+1, 2] - agg_pop[t, 2]) / (agg_pop[t, 0] + agg_pop[t, 1])

            agg_pop_rates = np.stack((agg_i, agg_f), axis=1)
            sub_pop_rate_ratios = np.stack((sub_pop_i_ratios, sub_pop_f_ratios), axis = 1)

            (sub_pop, sub_pop_rates) = disaggregate_path(agg_pop, 
                                                         agg_pop_rates,
                                                         init_sub_pop_props, 
                                                         sub_pop_rate_ratios,
                                                         years)

            sub_pop_alive = sub_pop[:, 0, :]
            sub_pop_diseased = sub_pop[:, 1, :]
            sub_pop_dead = sub_pop[:, 2, :]

            sub_pop_N_sum = np.sum(init_sub_pop_N)
            sub_pop_alive_sum = np.sum(sub_pop_alive, axis=1)
            sub_pop_diseased_sum = np.sum(sub_pop_diseased, axis=1)
            sub_pop_dead_sum = np.sum(sub_pop_dead, axis=1)

            sub_pop_i = sub_pop_rates[:, 0, :]
            sub_pop_f = sub_pop_rates[:, 1, :]
            sub_pop_r = np.zeros((years, sub_pops))

        cohort_disease_df = pd.DataFrame()
        cohort_disease_df['cohort'] = cohort*np.ones(years)
        cohort_disease_df['age'] = range(age, age + years)
        cohort_disease_df['year'] = range(start_year, end_year + 1)
        cohort_disease_df['sex'] = [sex for year in range(start_year, end_year + 1)]
        cohort_disease_df['csm_risk_agg'] = csm_risk_agg
        cohort_disease_df['prev_alive_agg'] = agg_pop[0:-1, 0]
        cohort_disease_df['prev_diseased_agg'] = agg_pop[0:-1, 1]
        cohort_disease_df['prev_dead_agg'] = agg_pop[0:-1, 2]
        cohort_disease_df['N'] = N
        cohort_disease_df['alive_agg'] = agg_pop[1:, 0]
        cohort_disease_df['diseased_agg'] = agg_pop[1:, 1]
        cohort_disease_df['dead_agg'] = agg_pop[1:, 2]
        cohort_disease_df['prevalence_agg'] = (cohort_disease_df['prev_diseased_agg'] 
                                              / (cohort_disease_df['prev_diseased_agg']
                                                + cohort_disease_df['prev_alive_agg']
                                                )
        )
        cohort_disease_df['i_agg'] = agg_i
        cohort_disease_df['f_agg'] = agg_f
        cohort_disease_df['r_agg'] = agg_r
        cohort_disease_df['DR_agg'] = agg_DR
        cohort_disease_df['sub_pops'] = sub_pops

        for k in range(sub_pops):
            cohort_disease_df['prev_alive{}'.format(k)] = sub_pop_alive[0:-1, k]
            cohort_disease_df['prev_diseased{}'.format(k)] = sub_pop_diseased[0:-1, k]
            cohort_disease_df['prev_dead{}'.format(k)] = sub_pop_dead[0:-1, k]
            cohort_disease_df['N{}'.format(k)] = init_sub_pop_N[k]
            cohort_disease_df['alive{}'.format(k)] = sub_pop_alive[1:, k]
            cohort_disease_df['diseased{}'.format(k)] = sub_pop_diseased[1:, k]
            cohort_disease_df['dead{}'.format(k)] = sub_pop_dead[1:, k]
            cohort_disease_df['prevalence{}'.format(k)] = (cohort_disease_df['diseased{}'.format(k)]
                                                          / (cohort_disease_df['diseased{}'.format(k)]
                                                            + cohort_disease_df['alive{}'.format(k)]
                                                            )
            )
            cohort_disease_df['i{}'.format(k)] = sub_pop_i[:, k]
            cohort_disease_df['f{}'.format(k)] = sub_pop_f[:, k]
            cohort_disease_df['r{}'.format(k)] = sub_pop_r[:, k]
            cohort_disease_df['DR{}'.format(k)] = sub_pop_DR[:, k]
        
        cohort_disease_df['subpop_alive_sum'] = sub_pop_alive_sum[1:]
        cohort_disease_df['alive_error'] = agg_pop[1:,0] - sub_pop_alive_sum[1:]
        cohort_disease_df['subpop_diseased_sum'] = sub_pop_diseased_sum[1:]
        cohort_disease_df['diseased_error'] = agg_pop[1:,1] - sub_pop_diseased_sum[1:]
        cohort_disease_df['subpop_dead_sum'] = sub_pop_dead_sum[1:]
        cohort_disease_df['dead_error'] = agg_pop[1:,2] - sub_pop_dead_sum[1:]

        output_df = output_df.append(cohort_disease_df, sort=False)
        #make_disease_plots(agg_pop, sub_pop, years)
        #make_disease_plots_with_background_mort(agg_pop, dead_other_causes_agg, sub_pop, dead_other_causes, years)
        #make_agg_total_pop_plot(agg_pop, dead_other_causes_agg, years)
        
    
    #plt.show()
    return output_df


def population_state_timestep_2state(pop, rate):
    '''Computes new population(s) for end of the timestep assuming population(s) changes 
    according to two-state Markov process.
    '''
    new_pop =  pop*np.exp(-rate)
    return new_pop

def get_subpop_rates_2state(agg_pop, agg_rate, sub_pops, rate_ratios):
    '''Returns transition rates for sub-populations for current timestep t assuming 
    sub-populations change according to two-state Markov process.
    '''
    if agg_rate == 0:
        return 0 * rate_ratios

    f = lambda x: np.sum(sub_pops*np.power(x,rate_ratios)) - population_state_timestep_2state(agg_pop, agg_rate)
    if f(0)*f(1) >= 0:
        pdb.set_trace()
    root = optimize.brentq(f,0,1)
    return -np.log(root)*rate_ratios

def disaggregate_2state(agg_pop, agg_rates, init_sub_pop_props, rate_ratios, timesteps, inflow_offset):
    sub_pops = np.zeros((timesteps+1, init_sub_pop_props.size))
    sub_pop_rates = np.zeros((timesteps, init_sub_pop_props.size))

    sub_pops[0] = agg_pop[0]*init_sub_pop_props
    
    for t in range (0, timesteps):
        if agg_pop[t] != 0:
            sub_pop_rates[t] = get_subpop_rates_2state(agg_pop[t], agg_rates[t], sub_pops[t], rate_ratios[t])
        sub_pops[t+1] = population_state_timestep_2state(sub_pops[t], sub_pop_rates[t]) + inflow_offset[t]

    return sub_pops, sub_pop_rates


def disaggregate_2state_single_timestep(agg_pop, agg_rates, init_sub_pops, rate_ratios, inflow_offset):
    sub_pop_rates = np.zeros(init_sub_pops.size)

    if agg_pop != 0:
        sub_pop_rates = get_subpop_rates_2state(agg_pop, agg_rates, init_sub_pops, rate_ratios)

    sub_pops = population_state_timestep_2state(init_sub_pops, sub_pop_rates) + inflow_offset

    return sub_pops, sub_pop_rates


def disaggregate_path(agg_pop, agg_rates, init_sub_pop_props, rate_ratios, timesteps):
    m = agg_pop[0].size
    n = init_sub_pop_props[0].size
    sub_pops = np.zeros((timesteps+1,m,n))
    sub_pop_rates = np.zeros((timesteps,m-1,n))
    sub_pop_N = np.matmul(agg_pop[0], init_sub_pop_props)

    #Compute for first state
    sub_pops[:,0,:], sub_pop_rates[:,0,:] = disaggregate_2state(agg_pop[:,0],
                                                                agg_rates[:,0],
                                                                init_sub_pop_props[0], 
                                                                rate_ratios[:,0,:], 
                                                                timesteps,
                                                                np.zeros(timesteps))

    #Compute for 2nd to penultimate states
    for j in range(1,m-1):
        sub_pops[:,j,:], sub_pop_rates[:,j,:] = disaggregate_2state(agg_pop[:,j], 
                                                                    agg_rates[:,j],
                                                                    init_sub_pop_props[j], 
                                                                    rate_ratios[:,j,:], 
                                                                    timesteps, 
                                                                    (sub_pops[0:timesteps,j-1,:] - 
                                                                    sub_pops[1:timesteps+1,j-1,:])
                                                                    )
    
    #Compute for final state
    sub_pops[:,m-1,:] = sub_pop_N - np.sum(sub_pops[:,0:m-1,:], axis=1)

    return sub_pops, sub_pop_rates


def disaggregate_mort_adjusted_disease_no_remission(agg_pop, agg_rates,
                                                    agg_pop_intermediate_alive, agg_pop_intermediate_diseased,
                                                    healthy_deaths_other_agg, diseased_deaths_other_agg,
                                                    sub_pop_mort, init_sub_pop_props, 
                                                    rate_ratios, timesteps):
    n = init_sub_pop_props[0].size
    sub_pops = np.zeros((timesteps+1,3,n))
    sub_pops[0,:,:] = np.transpose(agg_pop[0, None]) * init_sub_pop_props
    sub_pop_rates = np.zeros((timesteps,2,n))
    sub_pop_N = np.zeros((timesteps+1, n))
    sub_pop_N[0] = np.matmul(agg_pop[0], init_sub_pop_props)

    acm_risk = 1 - np.exp(-sub_pop_mort)

    csm_risk = np.zeros((timesteps, n))
    mort_risk_other = np.zeros((timesteps, n))
    dead_other_causes = np.zeros((timesteps+1, n))

    for t in range(timesteps):
        #Compute for alive population
        sub_pops[t+1,0,:], sub_pop_rates[t,0,:] = disaggregate_2state_single_timestep(agg_pop[t,0],
                                                                                      agg_rates[t,0],
                                                                                      sub_pops[t,0,:], 
                                                                                      rate_ratios[t,0,:], 
                                                                                      0)
        #Compute for diseased population
        sub_pops[t+1,1,:], sub_pop_rates[t,1,:] = disaggregate_2state_single_timestep(agg_pop[t,1],
                                                                                      agg_rates[t,1],
                                                                                      sub_pops[t,1,:], 
                                                                                      rate_ratios[t,1,:], 
                                                                                      sub_pops[t,0,:] - sub_pops[t+1,0,:])
        
        #Compute for cause specific dead population
        sub_pops[t+1,2,:] = sub_pop_N[t] - sub_pops[t+1,0,:] - sub_pops[t+1,1,:]

        #Get risks
        #(D - D_prev) / (S_prev + C_prev)
        if ((sub_pops[t, 0, :] + sub_pops[t, 1, :]) == 0).any():
            pdb.set_trace()
        csm_risk[t, :] = ((sub_pops[t+1, 2, :] - sub_pops[t, 2, :]) / 
                          (sub_pops[t, 0, :] + sub_pops[t, 1, :]))

        #csm_risk[t, :] = (1 - np.exp(-sub_pop_rates[t,1,:])) * (sub_pops[t, 1, :] / (sub_pops[t, 0, :] + sub_pops[t, 1, :])) 

        mort_risk_other[t, :] = acm_risk[t, :] - csm_risk[t, :]
        #mort_risk_other[t, :] = np.zeros(n)

        #Apply background mortality
        healthy_deaths_other = mort_risk_other[t, :] * sub_pops[t,0,:]
        diseased_deaths_other = mort_risk_other[t, :] * sub_pops[t,1,:]
        dead_other_causes[t+1, :] = dead_other_causes[t, :] + healthy_deaths_other + diseased_deaths_other
        sub_pops[t+1,0,:] -= healthy_deaths_other
        sub_pops[t+1,1,:] -= diseased_deaths_other
        sub_pop_N[t+1] = sub_pop_N[t] - (healthy_deaths_other + diseased_deaths_other)

    return sub_pops, sub_pop_rates, sub_pop_N, acm_risk, csm_risk, mort_risk_other, dead_other_causes


def disaggregate_YLD(agg_HALY, sub_pop_PY, sub_pop_yld_ratios):
    ref_YLD = ((np.sum(sub_pop_PY, axis=1) - agg_HALY)
                / np.sum(sub_pop_yld_ratios * sub_pop_PY, axis=1)
    )

    return sub_pop_yld_ratios * ref_YLD[:, None]


def compute_PYs(alive_pop, dead_pop):
    return alive_pop[1:] + 0.5 * (dead_pop[1:] - dead_pop[0:-1])


def compute_HALYs(PY, YLD):
    return PY * (1 - YLD)


def load_cohort_file(filename):
    return pd.read_csv(filename)


def make_mortality_pop_plots(agg_pop, sub_pop_alive, sub_pop_dead, years):
    fig, ax = plt.subplots(2, 2, sharey=True)
    for state in range(0, 2):
        ax[state,0].stackplot(range(0,years+1),agg_pop[:,state])
    ax[0, 1].stackplot(range(0,years+1),np.transpose(sub_pop_alive))
    ax[1, 1].stackplot(range(0,years+1),np.transpose(sub_pop_dead))


def make_PY_plots(agg_PY, sub_pop_PY, years):
    fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].stackplot(range(1,years+1), agg_PY)
    ax[1].stackplot(range(1,years+1), np.transpose(sub_pop_PY))


def make_HALY_plots(agg_HALY, sub_pop_HALY, years):
    fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].stackplot(range(1,years+1), agg_HALY)
    ax[1].stackplot(range(1,years+1), np.transpose(sub_pop_HALY))


def make_disease_plots(agg_pop, sub_pop, years):
    fig, ax = plt.subplots(3, 2, sharey=True)
    ax[0, 0].stackplot(range(0,years+1), np.transpose(agg_pop[:, 0]))
    ax[1, 0].stackplot(range(0,years+1), np.transpose(agg_pop[:, 1]))
    ax[2, 0].stackplot(range(0,years+1), np.transpose(agg_pop[:, 2]))

    ax[0, 1].stackplot(range(0,years+1), np.transpose(sub_pop[:, 0, :]))
    ax[1, 1].stackplot(range(0,years+1), np.transpose(sub_pop[:, 1, :]))
    ax[2, 1].stackplot(range(0,years+1), np.transpose(sub_pop[:, 2, :]))


def make_disease_plots_with_background_mort(agg_pop, dead_other_causes_agg, sub_pop, dead_other_causes, years):
    fig, ax = plt.subplots(4, 2, sharey=True)
    ax[0, 0].stackplot(range(0,years+1), np.transpose(agg_pop[:, 0]))
    ax[1, 0].stackplot(range(0,years+1), np.transpose(agg_pop[:, 1]))
    ax[2, 0].stackplot(range(0,years+1), np.transpose(agg_pop[:, 2]))
    ax[3, 0].stackplot(range(0,years+1), dead_other_causes_agg)

    ax[0, 1].stackplot(range(0,years+1), np.transpose(sub_pop[:, 0, :]))
    ax[1, 1].stackplot(range(0,years+1), np.transpose(sub_pop[:, 1, :]))
    ax[2, 1].stackplot(range(0,years+1), np.transpose(sub_pop[:, 2, :]))
    ax[3, 1].stackplot(range(0,years+1), np.transpose(dead_other_causes))


def make_agg_total_pop_plot(agg_pop, dead_other_causes_agg, years):
    plt.figure()
    plt.stackplot(range(0,years+1), agg_pop[:, 0], agg_pop[:, 1], agg_pop[:, 2],
                                    dead_other_causes_agg,
                                    labels=('alive','diseased','dead_disease','dead_other'))


def output_disagg_table(output_df, filename):
    output_df.to_csv(filename, index=False)

#populations = ['maori', 'non-maori']
populations = ['maori']

for folder_name in populations:
    #############################################################
    '''Cohort demography data disaggregation'''
    cohort_filename = './{}/{}_cohort_population_data.csv'.format(folder_name, folder_name)
    cohort_outfile = './output/base_population_disaggregation.csv'.format(folder_name)
    #cohort_filename = './{}/{}_cohort_population_data_naive.csv'.format(folder_name, folder_name)
    #cohort_outfile = './{}/base_population_disaggregation_naive.csv'.format(folder_name)

    cohort_data = load_cohort_file(cohort_filename)
    #print(cohort_data)

    demography_df = disaggregate_cohort_demography(cohort_data)
    output_disagg_table(demography_df, cohort_outfile)
    ##############################################################
    '''Cohort disease data disaggregation'''
    disease_list = ['CHD', 'Stroke']

    for disease in disease_list:
        disease_filename = './{}/diseases/{}_disease_cohort_data.csv'.format(folder_name, disease)
        disease_outfile =  './output/diseases/{}_disease_disaggregation.csv'.format(folder_name, disease)
        disease_data = load_cohort_file(disease_filename)

        disease_df = disaggregate_disease(disease_data)
        output_disagg_table(disease_df, disease_outfile)
    


