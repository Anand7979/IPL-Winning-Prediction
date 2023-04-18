import streamlit as st
import pickle
import pandas as pd

teams=['Rajasthan Royals',
 'Royal Challengers Bangalore',
 'Sunrisers Hyderabad',
 'Delhi Capitals',
 'Chennai Super Kings',
 'Gujarat Titans',
 'Lucknow Super Giants',
 'Kolkata Knight Riders',
 'Punjab Kings',
 'Mumbai Indians']

cities=['Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai',
       'Sharjah', 'Abu Dhabi', 'Delhi', 'Chennai','Hyderabad',
       'Visakhapatnam', 'Chandigarh', 'Bengaluru', 'Jaipur', 'Indore',
       'Bangalore', 'Kanpur', 'Rajkot', 'Raipur', 'Ranchi', 'Cuttack',
       'Dharamsala', 'Kochi', 'Nagpur', 'Johannesburg', 'Centurion',
       'Durban', 'Bloemfontein', 'Port Elizabeth', 'Kimberley',
       'East London', 'Cape Town']

pipe=pickle.load(open('pipe.pkl','rb'))
pipe1=pickle.load(open('pipe1.pkl','rb'))

st.title('IPL WINNING PREDICTOR')

c1,c2=st.columns(2)

with c1:
    batting_team=st.selectbox('Batting Team',sorted(teams))
with c2:
    bowling_team=st.selectbox('Bowling Team',sorted(teams))

cities=st.selectbox('Cities',sorted(cities))

target=int(st.number_input('Target'))

c3,c4,c5=st.columns(3)
with c3:
    runs=int(st.number_input('Runs'))
with c4:
    overs=int(st.number_input('Overs Completed'))
with c5:
    wickets=int(st.number_input('Wickets'))

if st.button('Predict Probability'):
    runs_l=int(target-runs)
    balls_l=int(120-(overs*6))
    wickets=int(10-wickets)
    crr=runs/overs
    rr=(runs_l/balls_l)*6

    df=pd.DataFrame({'BattingTeam':[batting_team],'BowlingTeam':[bowling_team],'City':[cities],'r_l':[runs_l],
                     'b_l':[balls_l],'wicket_l':[wickets],'total_run_x':[target],
                     'c_r':[crr],'r_r':[rr]})
    st.table(df)
    result=pipe.predict_proba(df)
    st.text(result)
    loss=result[0][0]
    win=result[0][1]
    st.header(batting_team + ' ' + str(round(win*100)) + '%')
    st.header(bowling_team + ' ' + str(round(loss * 100)) + '%')



