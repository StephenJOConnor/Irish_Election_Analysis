import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

# In[4]:

dail = pd.read_pickle('./dail_election_data.pkl')

dail['rounds'] = dail['transfers'].apply(lambda x: len(x))

# In[]

#create function to find inward vote within list x for index i
def inward_votes(x, i):
    if len(x) > i and len(x)>0:
        if x[i] > 0:
            return x[i]
        else:
            return 0
    else:
        return 0
# In[]

# create columns for each inward vote-transfer
for i in range(0, max(dail['rounds'])):   
    dail['inward_vote_' + str(i)] = dail.apply(lambda x: inward_votes(x['transfers'], i), axis = 1)

# In[16]:

# 
for i in range(1, max(dail['rounds'])):
    
    # find the parties eliminated at each round
    rnd_transfers = dail[dail['elimination_rd']==i][['race', 'party']]    
    rnd_transfers = rnd_transfers.drop_duplicates()
    rnd_transfers['party_transfer_' + str(i)] = rnd_transfers.\
        apply(lambda x: sorted(list(rnd_transfers[rnd_transfers['race']== x['race']]['party'])), axis=1)
    rnd_transfers = rnd_transfers.drop(['party'], axis=1)
    
    # find how many votes were transfered to each candidate by eliminated party
    rnd_transfers['party_transfer_' + str(i)] = rnd_transfers['party_transfer_' + str(i)].\
        apply(lambda x: str(x) if x is not None else "[]")
        
    rnd_transfers = rnd_transfers.drop_duplicates()
        
    rnd_transfers['party_transfer_' + str(i)] = rnd_transfers['party_transfer_' + str(i)].astype('str')
    
    dail = pd.merge(dail, rnd_transfers, how = 'left', on = 'race')

# In[]

for i in range(1, max(dail['rounds'])):
    dail['party_transfer_' + str(i)] = dail['party_transfer_' + str(i)].apply(lambda x: '[]' if str(x)=='nan' else x)

# In[]

# isolate single party transfers
def individual_party(x):
    x = ast.literal_eval(x)
    
    if isinstance(x, list):
        if len(x)==1:
            return x[0]
    else:
        return None
    
# In[]

#isolate transfers between individual parties in each round/race
## find the aggregate transferred for each round/race
for i in range(1, max(dail['rounds'])):
    
    # find the single party who transfered votes
    dail['single_party_transfer_' + str(i)] = dail['party_transfer_' + str(i)].apply(lambda x: individual_party(x))
    
    # total vote from that or all transferring parties
    vote_out_ag = dail[dail['elimination_rd']==i][['race', 'vote_out']]
    vote_out_ag = pd.DataFrame(vote_out_ag.groupby(['race'], as_index = False).sum())
    vote_out_ag['agg_vote_out_' + str(i)] = vote_out_ag['vote_out']
    vote_out_ag = vote_out_ag.drop('vote_out', axis = 1)
    dail = pd.merge(dail, vote_out_ag, how = 'left', on = 'race')
    dail['agg_vote_out_' + str(i)] = dail['agg_vote_out_' + str(i)].fillna(0).astype(int)


# In[25]:

#create dataframe to record all transfers between individual parties and their pct share of all transferred votes
for i in range(1, max(dail['rounds'])):
    
    if i==1:
        #find aggregate inward vote by party/election/race for round 1
        transfers = pd.DataFrame(dail.groupby(['elec_num', 'year', 'party', 'single_party_transfer_' + str(i),  'race', 'agg_vote_out_' + str(i)], \
                                            as_index=False)['inward_vote_' + str(i)].sum())
        
        #find the aggregate outword vote by party/election
        transfers = pd.DataFrame(transfers.groupby(['elec_num', 'year', 'party', 'single_party_transfer_' + str(i)], \
                                            as_index=False)['inward_vote_' + str(i), 'agg_vote_out_' + str(i)].sum())
        
    else:
        
        #find aggregate inward vote by party/election/race for round i
        transferred = pd.DataFrame(dail.groupby(['elec_num', 'party', 'year', 'single_party_transfer_' + str(i),  'race', 'agg_vote_out_' + str(i)], \
                                            as_index=False)['inward_vote_' + str(i)].sum())
        
        #find aggregate outward vote by party/election round i
        transferred = pd.DataFrame(transferred.groupby(['elec_num', 'party', 'year','single_party_transfer_' + str(i)], \
                                            as_index=False)['inward_vote_' + str(i), 'agg_vote_out_' + str(i)].sum())
        
        transfers = pd.merge(transfers, transferred, how = 'outer', left_on = ['elec_num', 'year', 'party', 'single_party_transfer_1'], \
                                                                         right_on = ['elec_num', 'year', 'party', 'single_party_transfer_' + str(i)])
        
        transfers['single_party_transfer_1'] = np.where(transfers['single_party_transfer_1'].isnull(), \
           transfers['single_party_transfer_' + str(i)], transfers['single_party_transfer_1'])
        
        
        #drop transfer party name
        transfers = transfers.drop('single_party_transfer_' + str(i), axis=1)
    
transfers = transfers.rename(columns = {"single_party_transfer_1": "transfer_party"})

#sort transfers by elec_num
transfers = transfers.sort_values(by=['elec_num'])

#drop feb 1982 election (it's missing transfers)
transfers = transfers[transfers['elec_num']>0]

# In[25]:

#get rid of null values and calculate pct_trans for each round
for i in range(1, max(dail['rounds'])):
    #fill nans w/ 0
    transfers['agg_vote_out_' + str(i)] = transfers['agg_vote_out_' + str(i)].fillna(0).astype(int)
    transfers['inward_vote_' + str(i)] = transfers['inward_vote_' + str(i)].fillna(0).astype(int)
    
    if i>1:
        #calculate cumulative transfers between individual parties
        transfers['cum_trans_out_' + str(i)] = transfers['agg_vote_out_' + str(i)] + transfers['cum_trans_out_' + str(i-1)]
        transfers['cum_trans_in_' + str(i)] = transfers['inward_vote_' + str(i)] + transfers['cum_trans_in_' + str(i-1)]     
        
    else:
        transfers['cum_trans_out_' + str(i)] = transfers['agg_vote_out_'  + str(i)]
        transfers['cum_trans_in_' + str(i)] = transfers['inward_vote_'  + str(i)]
    
    #calculate pct transfers per round
    transfers['rnd_pct_trans_' + str(i)] = transfers['inward_vote_'  + str(i)]/transfers['agg_vote_out_'  + str(i)]
    transfers['cum_pct_trans_' + str(i)] = transfers['cum_trans_in_' + str(i)]/transfers['cum_trans_out_' + str(i)]
    
    #fill nas when denominator is zero
    transfers['rnd_pct_trans_' + str(i)] = transfers['rnd_pct_trans_' + str(i)].fillna(0)
    transfers['cum_pct_trans_' + str(i)] = transfers['cum_pct_trans_' + str(i)].fillna(0)


# In[27]:

# find the total first prefs per election

total = dail[['elec_num', 'race', 'valid']].drop_duplicates()

total = pd.DataFrame(total.groupby(['elec_num'], \
    as_index=False)['valid'].sum())

# find party share of first preferences
first_pref_count = pd.DataFrame(dail.groupby(['elec_num', 'party'], \
    as_index=False)['inward_vote_0'].sum())
first_pref_count = pd.merge(first_pref_count, total, on = 'elec_num', how = 'left')
first_pref_count['first_pref_pct'] = first_pref_count['inward_vote_0']/first_pref_count['valid']

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
fg_ff = transfers[(transfers['party']=='Fine Gael') &                        (transfers['transfer_party']=='Fianna Fáil')]
fg_fg = transfers[(transfers['party']=='Fine Gael') &                        (transfers['transfer_party']=='Fine Gael')]
fg_lp = transfers[(transfers['party']=='Fine Gael') &                        (transfers['transfer_party']=='Labour Party')]
fg_ind = transfers[(transfers['party']=='Fine Gael') &                        (transfers['transfer_party']=='Independent')]

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


# In[51]:

fit = np.polyfit(fg_ff_x, fg_ff_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(fg_ff_x, fg_ff_y, 'yo', fg_ff_x, fit_fn(fg_ff_x), '--k', color = 'indigo')

fit = np.polyfit(ff_ff_x, ff_ff_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(ff_ff_x, ff_ff_y, 'yo', ff_ff_x, fit_fn(ff_ff_x), '--k', color='g')

plt.show

fit = np.polyfit(ff_fg_x, ff_fg_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(ff_fg_x, ff_fg_y, 'yo', ff_fg_x, fit_fn(ff_fg_x), '--k', color = 'r')

fit = np.polyfit(fg_fg_x, fg_fg_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(fg_fg_x, fg_fg_y, 'yo', fg_fg_x, fit_fn(fg_fg_x), '--k', color='b')

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

labels=np.array(['Fianna Fáil', 'Labour Party', 'Fine Gael', 'Independent', 'Sinn Féin','Green Party', 'Progressive Democrats'])
stats=radar_table[radar_table['year']==1989][labels].values[0]*100

angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
# close the plot
stats=np.concatenate((stats,[stats[0]]))
angles=np.concatenate((angles,[angles[0]]))

fig=plt.figure()

#create plot for 1982
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, stats, 'b', label="1989", linewidth=2)
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

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

ax.set_thetagrids(angles * 180/np.pi, labels)
ax.set_title("Fine Gael Inward Transfers")
ax.grid(True)
