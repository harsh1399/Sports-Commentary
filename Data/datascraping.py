import pandas as pd
import json
import urllib3
from time import sleep
import numpy as np
import os

http = urllib3.PoolManager()

col_data = pd.DataFrame()
mat_data = pd.DataFrame()
periods = ["1","2"]
# pages = ["1","2","3","4","5"]

# files = os.listdir("remaining_bbl")
events = None
# with open("BBL/logs.txt",'r') as f:
#     events = f.readlines()
leagueId="381449&"
eventId="433606"

# dat_url = pd.read_csv("./BBL Matches 2011-2019.csv")
# balls = pd.read_csv("./BBL Ball-by-Ball 2011-2019.csv")
# eventId_gp= dat_url['ID']
# len_url= len(eventId_gp)

updated_data = pd.DataFrame()
# for event in events:
# # for count in range(len_url):
# #     eventId = str(eventId_gp[count])
#     eventId = event[:-1]
#     if f"{eventId}.csv" in files:
#         continue
#     if eventId == "654035" or eventId == "1114887":
#         continue
#     print(eventId)
is_fetched = True
for period in periods:
    print("period ",period)
    # match_frame = balls[(balls['ID'] == int(eventId)) & (balls['innings'] == int(period))].copy()
    # match_frame.sort_values(by=["overs","ballnumber"],inplace=True)
    dfs = []
    url = 'https://hsapi.espncricinfo.com/v1/pages/match/comments?lang=en&leagueId=' + leagueId + '&eventId=' + eventId + '&period=' + period + '&page=1&filter=full&liveTest=false'
    match_dat = http.request('GET',url)
    data = json.loads(match_dat.data)
    pagecount = 5
    try:
        pagecount = data['pagination']['pageCount']
    except:
        randomsleep = np.random.randint(low=90, high=120)
        sleep(randomsleep)
        match_dat = http.request('GET', url)
        data = json.loads(match_dat.data)
        try:
            pagecount = data['pagination']['pageCount']
        except:
            with open("remaining_bbl/logs.txt", 'a') as f:
                f.write(f"\n pagination error {eventId}")
    pages = list(range(1,pagecount+1))
    for page in pages:
        print("page ",page)
        col_data = pd.DataFrame()
        url = 'https://hsapi.espncricinfo.com/v1/pages/match/comments?lang=en&leagueId='+leagueId+'&eventId='+eventId+'&period=' +period+ '&page='+str(page)+'&filter=full&liveTest=false'
        match_dat= http.request('GET', url)
        data = json.loads(match_dat.data)
        try:
            df = pd.json_normalize(data['comments'])
        except:
            randomsleep1 = np.random.randint(low=120, high=180)
            sleep(randomsleep1)
            match_dat= http.request('GET', 'https://hsapi.espncricinfo.com/v1/pages/match/comments?lang=en&leagueId='+leagueId+'&eventId='+eventId+'&period=' +period+ '&page='+str(page)+'&filter=full&liveTest=false')
            data = json.loads(match_dat.data)
            try:
                df = pd.json_normalize(data['comments'])
            except:
                is_fetched = False
                break
        if len(data['comments']) !=0:
          df_new = df[::-1]
          dfs.append(df_new)
    if is_fetched==False:
        print(f"error in fetching for: {eventId}")
        # with open("remaining_bbl/logs.txt",'a') as f:
        #     f.write(f"\n{eventId} {period}")
        break
    dataframe = pd.concat(dfs,axis=0)
    # dataframe = dataframe.reset_index(drop=True)
    # match_frame = match_frame.reset_index(drop=True)
    # concated_matchframe = pd.concat([match_frame,dataframe],axis=1)
    updated_data = pd.concat([updated_data,dataframe],axis=0)
updated_data.to_csv(f"commentary.csv")


#balling_lengths = ["Full Toss","Yorker","Full Length","full","fuller","Good Length","length","short","good length"]
# lengths = []
# for i in range(len(balls)):
#     #print(ball['text'].iloc[i]))
#     success = 0
#     print(i)
#     for j in balling_lengths:
#         print(j)
#         text = balls['text'].iloc[i]
#         if isinstance(text,float) is False:
#             if j in balls['text'].iloc[i]:
#                 print("Success")
#                 lengths.append(j)
#                 success = 1
#                 break
#     if success != 1:
#         lengths.append(np.nan)
# length = pd.Series(lengths)
# balls['bowling_length'] = length.fillna(method='bfill')
#
# batting_shots = ["cover","point","punch","punches","third","fine leg","leg","square","square leg","mid wicket","midwicket","hook","pull","mid on","mid-on","mid-off","mid off","long-on","long-off","fine","edge","straight","sweep","down the ground","over the bowler's head","upper-cut"]
# shots = []
# for i in range(len(balls)):
#     success = 0
#     print(i)
#     for j in batting_shots:
#         print(j)
#         text = balls['text'].iloc[i]
#         if isinstance(text,float) is False:
#             if j in balls['text'].iloc[i]:
#                 print("Success")
#                 shots.append(j)
#                 success = 1
#                 break
#     if success != 1:
#         shots.append(np.nan)
# shot = pd.Series(shots)
# ball['batting_shot'] = shot.fillna(method='bfill')
# updated_data.to_csv("IPL_Ball_by_Ball_2022.csv")