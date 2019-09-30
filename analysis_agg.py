# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:53:53 2019

@author: sjpoc
"""
# =============================================================================
# Data plots and radar graphs to anlayze FF vote sharing 82-2016
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Create dataframes from pickled files

transfers = pd.read_pickle('./transfers_82_16.pkl')
first_prefs = pd.read_pickle('./first_prefs.pkl')

# In[32]:

##Fianna Fail's inward transfers
ff_ff = transfers[(transfers['party']=='Fianna Fáil') & (transfers['transfer_party']=='Fianna Fáil')]
ff_fg = transfers[(transfers['party']=='Fianna Fáil') & (transfers['transfer_party']=='Fine Gael')]
ff_lp = transfers[(transfers['party']=='Fianna Fáil') & (transfers['transfer_party']=='Labour Party')]
ff_ind = transfers[(transfers['party']=='Fianna Fáil') & (transfers['transfer_party']=='Independent')]
ff_sf = transfers[(transfers['party']=='Fianna Fáil') & (transfers['transfer_party']=='Sinn Féin')]

##By top three parties plus independents
ff_ff_x, ff_ff_y = ff_ff['elec_num'], ff_ff['cum_pct_trans_18']
ff_fg_x, ff_fg_y = ff_fg['elec_num'], ff_fg['cum_pct_trans_18']
ff_lp_x, ff_lp_y = ff_lp['elec_num'], ff_lp['cum_pct_trans_18']
ff_ind_x, ff_ind_y = ff_ind['elec_num'], ff_ind['cum_pct_trans_18']
ff_sf_x, ff_sf_y = ff_sf['elec_num'], ff_sf['cum_pct_trans_18']

##plotting
plt.plot(ff_ff_x, ff_ff_y, color='g')
plt.plot(ff_fg_x, ff_fg_y, color='blue')
plt.plot(ff_lp_x, ff_lp_y, color='red')
plt.plot(ff_ind_x, ff_ind_y, color='black')
plt.plot(ff_sf_x, ff_sf_y, color='yellow')
plt.xlabel('Elections')
plt.ylabel('Transfers')
plt.title('1st Pref Transfers to FF')
plt.show

# In[33]:

##Labour's inward transfers
lp_ff = transfers[(transfers['party']=='Labour Party') & (transfers['transfer_party']=='Fianna Fáil')]
lp_fg = transfers[(transfers['party']=='Labour Party') & (transfers['transfer_party']=='Fine Gael')]
lp_lp = transfers[(transfers['party']=='Labour Party') & (transfers['transfer_party']=='Labour Party')]
lp_ind = transfers[(transfers['party']=='Labour Party') & (transfers['transfer_party']=='Independent')]

##By top three parties plus independents
lp_ff_x, lp_ff_y = lp_ff['elec_num'], lp_ff['cum_pct_trans_18']
lp_fg_x, lp_fg_y = lp_fg['elec_num'], lp_fg['cum_pct_trans_18']
lp_lp_x, lp_lp_y = lp_lp['elec_num'], lp_lp['cum_pct_trans_18']
lp_ind_x, lp_ind_y = lp_ind['elec_num'], lp_ind['cum_pct_trans_18']

##plotting
plt.plot(lp_ff_x, lp_ff_y, color='g')
plt.plot(lp_fg_x, lp_fg_y, color='blue')
plt.plot(lp_lp_x, lp_lp_y, color='red')
plt.plot(lp_ind_x, lp_ind_y, color='black')
plt.xlabel('Elections')
plt.ylabel('Transfers')
plt.title('1st Pref Transfers to LP')
plt.show


# In[34]:


##Independents' inward transfers
ind_ff = transfers[(transfers['party']=='Independent') & (transfers['transfer_party']=='Fianna Fáil')]
ind_fg = transfers[(transfers['party']=='Independent') & (transfers['transfer_party']=='Fine Gael')]
ind_lp = transfers[(transfers['party']=='Independent') & (transfers['transfer_party']=='Labour Party')]
ind_ind = transfers[(transfers['party']=='Independent') & (transfers['transfer_party']=='Independent')]

##By top three parties plus independents
ind_ff_x, ind_ff_y = ind_ff['elec_num'], ind_ff['cum_pct_trans_18']
ind_fg_x, ind_fg_y = ind_fg['elec_num'], ind_fg['cum_pct_trans_18']
ind_lp_x, ind_lp_y = ind_lp['elec_num'], ind_lp['cum_pct_trans_18']
ind_ind_x, ind_ind_y = ind_ind['elec_num'], ind_ind['cum_pct_trans_18']

##plotting
plt.plot(ind_ff_x, ind_ff_y, color='g')
plt.plot(ind_fg_x, ind_fg_y, color='blue')
plt.plot(ind_lp_x, ind_lp_y, color='red')
plt.plot(ind_ind_x, ind_ind_y, color='black')
plt.xlabel('Elections')
plt.ylabel('Transfers')
plt.title('1st Pref Transfers to Independents')
plt.show


# In[38]:


##Fine Gael's inward transfers
fg_ff = transfers[(transfers['party']=='Fine Gael') & (transfers['transfer_party']=='Fianna Fáil')]
fg_fg = transfers[(transfers['party']=='Fine Gael') & (transfers['transfer_party']=='Fine Gael')]
fg_lp = transfers[(transfers['party']=='Fine Gael') & (transfers['transfer_party']=='Labour Party')]
fg_ind = transfers[(transfers['party']=='Fine Gael') & (transfers['transfer_party']=='Independent')]

##By top three parties plus independents
fg_ff_x, fg_ff_y = fg_ff['elec_num'], fg_ff['cum_pct_trans_18']
fg_fg_x, fg_fg_y = fg_fg['elec_num'], fg_fg['cum_pct_trans_18']
fg_lp_x, fg_lp_y = fg_lp['elec_num'], fg_lp['cum_pct_trans_18']
fg_ind_x, fg_ind_y = fg_ind['elec_num'], fg_ind['cum_pct_trans_18']


##plotting
plt.plot(fg_ff_x, fg_ff_y, color='g')
plt.plot(fg_fg_x, fg_fg_y, color='blue')
plt.plot(fg_lp_x, fg_lp_y, color='red')
plt.plot(fg_ind_x, fg_ind_y, color='black')

plt.xlabel('Elections')
plt.ylabel('Transfers')
plt.title('1st Pref Transfers to FG')
plt.show


# In[39]:

##SF's inward vote
sf_ff = transfers[(transfers['party']=='Sinn Féin') & (transfers['transfer_party']=='Fianna Fáil')]
sf_fg = transfers[(transfers['party']=='Sinn Féin') & (transfers['transfer_party']=='Fine Gael')]
sf_lp = transfers[(transfers['party']=='Sinn Féin') & (transfers['transfer_party']=='Labour Party')]
sf_ind = transfers[(transfers['party']=='Sinn Féin') &(transfers['transfer_party']=='Independent')]

##By top three parties plus independents
sf_ff_x, sf_ff_y = sf_ff['elec_num'], sf_ff['cum_pct_trans_18']
sf_fg_x, sf_fg_y = sf_fg['elec_num'], sf_fg['cum_pct_trans_18']
sf_lp_x, sf_lp_y = sf_lp['elec_num'], sf_lp['cum_pct_trans_18']
sf_ind_x, sf_ind_y = sf_ind['elec_num'], sf_ind['cum_pct_trans_18']


#%%

ff_first_prefs_y = first_prefs[(first_prefs['party']=='Fianna Fáil') & (first_prefs['elec_num']<8)]['first_pref_pct']
ff_first_prefs_x = first_prefs[(first_prefs['party']=='Fianna Fáil') & (first_prefs['elec_num']<8)]['elec_num']

fg_first_prefs_y = first_prefs[(first_prefs['party']=='Fine Gael') & (first_prefs['elec_num']<8)]['first_pref_pct']
fg_first_prefs_x = first_prefs[(first_prefs['party']=='Fine Gael') & (first_prefs['elec_num']<8)]['elec_num']


# In[51]:

ff_ff_x = ff_ff_x[:-2]
ff_ff_y = ff_ff_y[:-2]
fg_fg_x = fg_fg_x[:-2]
fg_fg_y = fg_fg_y[:-2]

fit = np.polyfit(ff_ff_x, ff_ff_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(ff_ff_x, ff_ff_y, 'yo', ff_ff_x, fit_fn(ff_ff_x), '--k', color='g')

fit = np.polyfit(fg_fg_x, fg_fg_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(fg_fg_x, fg_fg_y, 'yo', fg_fg_x, fit_fn(fg_fg_x), '--k', color='b')

fit = np.polyfit(ff_first_prefs_x, ff_first_prefs_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(ff_first_prefs_x, ff_first_prefs_y, 'yo', ff_first_prefs_x, fit_fn(ff_first_prefs_x), '--k', color='r')

fit = np.polyfit(fg_first_prefs_x, fg_first_prefs_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(fg_first_prefs_x, fg_first_prefs_y, 'yo', fg_first_prefs_x, fit_fn(fg_first_prefs_x), '--k', color='b')

plt.show
# In[51]:

fit = np.polyfit(ind_ff_x, ind_ff_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(ind_ff_x, ind_ff_y, 'yo', ind_ff_x, fit_fn(ind_ff_x), '--k', color='black')

fit = np.polyfit(ff_ind_x, ff_ind_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(ff_ind_x, ff_ind_y, 'yo', ff_ind_x, fit_fn(ff_ind_x), '--k', color='green')

fit = np.polyfit(fg_ind_x, fg_ind_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(fg_ind_x, fg_ind_y, 'yo', fg_ind_x, fit_fn(fg_ind_x), '--k', color='blue')

plt.show

# In[51]:

#create table for radar graphs (fianna fail only)
radar_table = pd.DataFrame()
radar_table = transfers[(transfers['party']=='Fianna Fáil') & (transfers['transfer_party']=='Fianna Fáil')][['year', 'cum_pct_trans_18']]
radar_table = radar_table.rename(columns = {'cum_pct_trans_18': 'Fianna Fáil'})

radar_table = pd.merge(radar_table, transfers[(transfers['party']=='Fianna Fáil') & (transfers['transfer_party']=='Labour Party')][['year', 'cum_pct_trans_18']], on='year', how='left')
radar_table = radar_table.rename(columns = {'cum_pct_trans_18': 'Labour Party'})

radar_table = pd.merge(radar_table, transfers[(transfers['party']=='Fianna Fáil') & (transfers['transfer_party']=='Fine Gael')][['year', 'cum_pct_trans_18']], on='year', how='left')
radar_table = radar_table.rename(columns = {'cum_pct_trans_18': 'Fine Gael'})

radar_table = pd.merge(radar_table, transfers[(transfers['party']=='Fianna Fáil') & (transfers['transfer_party']=='Independent')][['year', 'cum_pct_trans_18']], on='year', how='left')
radar_table = radar_table.rename(columns = {'cum_pct_trans_18': 'Independent'})

radar_table = pd.merge(radar_table, transfers[(transfers['party']=='Fianna Fáil') & (transfers['transfer_party']=='Green Party')][['year', 'cum_pct_trans_18']], on='year', how='left')
radar_table = radar_table.rename(columns = {'cum_pct_trans_18': 'Green Party'})

radar_table = pd.merge(radar_table, transfers[(transfers['party']=='Fianna Fáil') & (transfers['transfer_party']=='Sinn Féin')][['year', 'cum_pct_trans_18']], on='year', how='left')
radar_table = radar_table.rename(columns = {'cum_pct_trans_18': 'Sinn Féin'})

radar_table = pd.merge(radar_table, transfers[(transfers['party']=='Fianna Fáil') & (transfers['transfer_party']=='Progressive Democrats')][['year', 'cum_pct_trans_18']], on='year', how='left')
radar_table = radar_table.rename(columns = {'cum_pct_trans_18': 'Progressive Democrats'})

radar_table = radar_table.fillna(0)

# In[51]:

labels=np.array(['Fianna Fáil', 'Labour Party', 'Fine Gael', 'Independent'])
stats=radar_table[radar_table['year']==1982][labels].values[0]*100

angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
# close the plot
stats=np.concatenate((stats,[stats[0]]))
angles=np.concatenate((angles,[angles[0]]))

fig=plt.figure()

#create plot for 1982
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, stats, 'b', label="1982", linewidth=2)
ax.fill(angles, stats, alpha=0.25)

#for 2016
stats=radar_table[radar_table['year']==2016][labels].values[0]*100
angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

# close the plot
stats=np.concatenate((stats,[stats[0]]))
angles=np.concatenate((angles,[angles[0]]))

#create plot for 2016
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, stats, 'r', label="2016", linewidth=2)
ax.fill(angles, stats, alpha=0.25)

ax.tick_params(pad = 5)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

ax.set_thetagrids(angles * 180/np.pi, labels)
ax.set_title("Fianna Fáil Inward Transfers")
ax.title.set_position([.5, 1.1])
ax.grid(True)