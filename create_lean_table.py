# =============================================================================
# Creates and pickles a lean dataframe that aggregates vote transfers between
# individual parties for Dail elections between 1982 (Nov) and 2016. Also 
# creates first prefs table.
# =============================================================================

import pandas as pd
import numpy as np
import ast
from sqlite3 import connect

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

#%%

#if elemination_rd is 0, make elemination round the last round
dail['elimination_rd'] = dail.apply(lambda x: len(x['round_totals']) if x['elimination_rd']==0 else x['elimination_rd'], axis=1)

#%%

dail_choice = pd.read_sql_query("select race, count(party) from (select distinct race, party " + \
                                "from dail group by race, party having elimination_rd > 1) group by race", con)


# In[25]:

con = connect(":memory:")

dail.drop(['transfers','round_totals'], axis=1).to_sql(name='dail', con=con)

#create dataframe to record all transfers between individual parties and their pct share of all transferred votes
for i in range(1, max(dail['rounds'])):
    
    #find number of parties left in the race per round
    rnd_choice = pd.read_sql_query("select race, count(party) as rnd_" + str(i) + " from (select distinct race, party " + \
                                "from dail group by race, party having elimination_rd > " + str(i) + ") group by race", con)
    
    if i==1:
        
        #create the rnd_choices df
        rnd_choices = rnd_choice
        
        #find aggregate inward vote by party/election/race for round 1
        const_data = pd.DataFrame(dail.groupby(['elec_num', 'year', 'party', 'single_party_transfer_' + str(i),  'race', 'constit', 'agg_vote_out_' + str(i)], \
                                            as_index=False)['inward_vote_' + str(i)].sum())        
        
        #find the aggregate outword vote by party/election
        transfers = pd.DataFrame(const_data.groupby(['elec_num', 'year', 'party', 'single_party_transfer_' + str(i)], \
                                            as_index=False)['inward_vote_' + str(i), 'agg_vote_out_' + str(i)].sum())
        
    else:
        
        #update the rnd_choices df
        rnd_choices = pd.merge(rnd_choices, rnd_choice, how = "left", left_on = ['race'], right_on = ['race'])
        
        #find aggregate inward vote by party/election/race for round i
        const_data_2 = pd.DataFrame(dail.groupby(['elec_num', 'party', 'year', 'single_party_transfer_' + str(i),  'race', 'constit', 'agg_vote_out_' + str(i)], \
                                            as_index=False)['inward_vote_' + str(i)].sum())
        
        #merge constituent-level data
        const_data = pd.merge(const_data, const_data_2, how = 'outer', left_on = ['elec_num', 'year', 'party', 'single_party_transfer_1', 'race', 'constit'], \
                                                                         right_on = ['elec_num', 'year', 'party', 'single_party_transfer_' + str(i), 'race', 'constit'])
        
        #find aggregate outward vote by party/election round i
        transferred = pd.DataFrame(const_data_2.groupby(['elec_num', 'party', 'year','single_party_transfer_' + str(i)], \
                                            as_index=False)['inward_vote_' + str(i), 'agg_vote_out_' + str(i)].sum())
        
        transfers = pd.merge(transfers, transferred, how = 'outer', left_on = ['elec_num', 'year', 'party', 'single_party_transfer_1'], \
                                                                         right_on = ['elec_num', 'year', 'party', 'single_party_transfer_' + str(i)])
        
        transfers['single_party_transfer_1'] = np.where(transfers['single_party_transfer_1'].isnull(), \
           transfers['single_party_transfer_' + str(i)], transfers['single_party_transfer_1'])
        
        const_data['single_party_transfer_1'] = np.where(const_data['single_party_transfer_1'].isnull(), \
           const_data['single_party_transfer_' + str(i)], const_data['single_party_transfer_1'])
        
        
        #drop transfer party name
        transfers = transfers.drop('single_party_transfer_' + str(i), axis=1)
        const_data = const_data.drop('single_party_transfer_' + str(i), axis=1)
        
    
const_data = const_data.rename(columns = {"single_party_transfer_1": "transfer_party"})
transfers = transfers.rename(columns = {"single_party_transfer_1": "transfer_party"})

#sort transfers by elec_num
const_data = const_data.sort_values(by=['elec_num'])
transfers = transfers.sort_values(by=['elec_num'])

#drop feb 1982 election (it's missing transfers)
transfers = transfers[transfers['elec_num']>0]
const_data = const_data[const_data['elec_num']>0]

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

# In[]

transfers.to_pickle("./transfers_82_16.pkl")
first_pref_count.to_pickle("./first_prefs.pkl")
const_data.to_pickle("./const_data.pkl")
rnd_choices.to_pickle("./rnd_choices.pkl")

# In[25]:

def analysis_tables(transfers, const_data, rnd_choices, min_choice=0):

	
	transfers = pd.merge(transfers, rnd_choices, how = "left", on = ['race'])
	const_data = pd.merge(const_data, rnd_choices, how = "left", on = ['race'])
	
	
	
	#get rid of null values and calculate pct_trans for each round
	for i in range(1, 18):
	    #fill nans w/ 0
	    transfers['agg_vote_out_' + str(i)] = transfers['agg_vote_out_' + str(i)].fillna(0).astype(int)
	    transfers['inward_vote_' + str(i)] = transfers['inward_vote_' + str(i)].fillna(0).astype(int)
		
		#void aggregate transfers where min number of party choices don't exist 
		transfers['agg_vote_out_' + str(i)] = transfers.apply(lambda x: x['agg_vote_out_' + str(i)] \
			if x['rnd_' + str(i)] > min_choice else 0)
		transfers['inward_vote_' + str(i)] = transfers.apply(lambda x: x['inward_vote_' + str(i)] \
			if x['rnd_' + str(i)] > min_choice else 0)
	    
		#fill nans w/ 0
	    const_data['agg_vote_out_' + str(i)] = const_data['agg_vote_out_' + str(i)].fillna(0).astype(int)
	    const_data['inward_vote_' + str(i)] = const_data['inward_vote_' + str(i)].fillna(0).astype(int)
		
		#void constituency transfers where min number of party choices don't exist 
		
		
		
	    
	    if i>1:
	        #calculate cumulative transfers between individual parties
	        transfers['cum_trans_out_' + str(i)] = transfers['agg_vote_out_' + str(i)] + transfers['cum_trans_out_' + str(i-1)]
	        transfers['cum_trans_in_' + str(i)] = transfers['inward_vote_' + str(i)] + transfers['cum_trans_in_' + str(i-1)]
	
	        const_data['cum_trans_out_' + str(i)] = const_data['agg_vote_out_' + str(i)] + const_data['cum_trans_out_' + str(i-1)]
	        const_data['cum_trans_in_' + str(i)] = const_data['inward_vote_' + str(i)] + const_data['cum_trans_in_' + str(i-1)]                   
	    else:
	        transfers['cum_trans_out_' + str(i)] = transfers['agg_vote_out_'  + str(i)]
	        transfers['cum_trans_in_' + str(i)] = transfers['inward_vote_'  + str(i)]
	        
	        const_data['cum_trans_out_' + str(i)] = const_data['agg_vote_out_'  + str(i)]
	        const_data['cum_trans_in_' + str(i)] = const_data['inward_vote_'  + str(i)]
	    
	    #calculate pct transfers per round
	    transfers['rnd_pct_trans_' + str(i)] = transfers['inward_vote_'  + str(i)]/transfers['agg_vote_out_'  + str(i)]
	    transfers['cum_pct_trans_' + str(i)] = transfers['cum_trans_in_' + str(i)]/transfers['cum_trans_out_' + str(i)]
	    
	    const_data['rnd_pct_trans_' + str(i)] = const_data['inward_vote_'  + str(i)]/const_data['agg_vote_out_'  + str(i)]
	    const_data['cum_pct_trans_' + str(i)] = const_data['cum_trans_in_' + str(i)]/const_data['cum_trans_out_' + str(i)]
	    
	    #fill nas when denominator is zero
	    transfers['rnd_pct_trans_' + str(i)] = transfers['rnd_pct_trans_' + str(i)].fillna(0)
	    transfers['cum_pct_trans_' + str(i)] = transfers['cum_pct_trans_' + str(i)].fillna(0)
	    
	    const_data['rnd_pct_trans_' + str(i)] = const_data['rnd_pct_trans_' + str(i)].fillna(0)
	    const_data['cum_pct_trans_' + str(i)] = const_data['cum_pct_trans_' + str(i)].fillna(0)
		
		return(transfers, const_data)