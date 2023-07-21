<img src="https://github.com/Team-predicting-football-salary/Pigskin_Paydays/blob/main/images/title.png?raw=true">

# Pigskin Paydays

# Project Description 

In the NFL, every football team in the league is alloted a certain amount of money that it must then allocate amongst its players. This is known as the team's "cap." When players are up for contract renewal, the most valued players in every team, quarterbacks, ask for salaries far beyond what a team currently evaluates them at, and the two sides go back and forth before they reach a consensus based on the player's actual worth. Our project aims to predict what the percentage of the cap will be alloted for a player given his performance in a cycle.

# Project Goals

To predict what percentage of a team's cap will be paid out to the quarterback for his contract using objective stats.

# Project Planning

Acquisition
    - For player stats, acquire data from <a href='https://nextgenstats.nfl.com/stats/passing/2022/REG/all#yards'>Next Gen Stats</a>
    - Original individual player stats dataset consisted of 458 rows and 19 columns. Each row contains a single player for a given year between 2010 and 2022. 
    - Original playoff data consisted of 162 rows and 7 columns. Each row contains a team and the round that team made it to in a given year between 2010 and 2022.
    - Original player salary data consisted of 1,590 and 11 columns. Each row contains a player and their salary for a given year. 


Preparation
    - Once we merged our 3 data sets, cleaned the data, and conducted feature engineering, our final data frame consisted of 410 rows and 42 columns for exploration. Each row contains a single player with their stats for a respective year from 2010 to 2022. 
    - Dropped all nulls after the merge.

Exploration

Modeling


# Initial Hypothesis

Quarterbacks with the most passing yards and who extend their team's season will get paid the most money.

# Data Dictionary

Column Name | Description | Key
--- | --- | ---
player_name | The name of the quarterback | string
team | The team that the quarterback plays for | string
year | Year that specific row of stats were collected | integer  
games_played | The amount of games played in a specific season | float  
comp | Completion: Completed pass attempt | float  
att | Passing attempts: The number of times that the quarterback threw the ball that season | float  
comp_pct | Completion Percentage: Percentage of completed passes vs. pass attempts | percentage, float  
yds | Yards: Total number of yards that the quarterback threw that season | float  
avg_yds_per_att | Average Yards Per Attempt: Average yards gained per pass attempt | percentage, float  
td | Touchdowns: Number of touchdowns that quarterback threw - highest scoring play | float  
int | Intercepts: Number of times that the quarterback's throw was intercepted by the opposing team - negative stat for QB's | float  
pass_rating | Passer Rating: one way to evaluate a QB's play performance. Scale from 0 - 158.3 higher the score the better | float  
rush_att | Rushing Attempts: How many times a QB ran the ball instead of passing | float  
rush_yds | Rushing Yards: Amount of yards gain from running the ball | float  
rush_avg | Rushing Average: Average yards gain from each run attempt | percentage, float  
rush_td | Rushing Touchdowns: The amount of touchdowns gained from running the ball | float  
year_signed | The year a QB signed a new contract | integer  
percent_of_cap (target variable) | The amount of cap space allocated to the player | percentage, float  
age | How old the QB is that specific season | integer  
td_perc | Touchdown Percentage: Percentage of touchdowns gained from passing attempts | percentage, float  
int_perc | Interception Percentage: Percentage of interceptions gained from passing attempts | percentage, float  
fir_dn_throws | First Down Throws: Amount of first downs a QB passes for. First down gain is a good stat for QB | integer  
lng_comp | Longest completion: longest completed pass play | integer  
yds_per_comp | Yards per completed pass: Average number of yards gained per completed pass | percentage, float  
yd_per_gm | Yards per Game: Average yards from pass completions per game | percentage, float  
QBR | Quarterback Rating: One way to evaluate a QB's play performance. Scale from 0 - 100 higher the score the better | float  
Sk | Sack: Number of times a QB was tackled for negative yards | integer  
4QC | Fourth Quarter Comeback: The QB's team was losing but the QB took the lead or tied the game in the fourth quarter | float  
GWD | Game winning drive: The QB's team was tied or losing but the QB took the lead to end the game | float  
win | Amount of wins the QB and their team gained | integer  
loss | Amount of losses the QB and their team gained | integer  
wild_card | First round of the playoffs: 1 = QB played that round 0 = QB did not play that round | float  
div_rnd | Second round of the playoffs: 1 = QB played that round 0 = QB did not play that round | float  
conf_rnd | Third round of the playoffs: 1 = QB played that round 0 = QB did not play that round | float  
superbowl | Final round of the playoffs: 1 = QB played that round 0 = QB did not play that round | float  
superbowl_win | 1 = QB won the superbowl 0 = QB did not win the superbowl | float  
win_perc | Percentage of games that the QB won in a specific season | percentage, float  
loss_perc | Percentage of games that the QB lost in a specific season |percentage, float  
td_per_game | The average number of touchdowns a QB passed for in games played | percentage, float  
sk_per_game | The average number of sacks on a QB per in games played | percentage, float



# How to Reproduce

Download the repository and run through the notebook.

# Key Findings

QB Evolution
    - Game stats data matches the televised narative that NFL Quarterbacks are running the ball more often.
        - There has been an varied increase in run attempts from 2010 to 2020. The data shows a major spike, more than doubled from 2020 to 2022.
QB Contracts
    - QB contract size in relation to salary cap percentage stayed relatively flat from 2010 to 2017. They began to rise in 2018 and has increased dramatically from 2021 to 2022; albeit we have a small sample size for 2022.
Higher Pay for Winning QB's
    - We found that on average, Quarterbacks who reached the playoffs demand a higher percentage of a teams salary cap. 
    - There is no statistical signifance between quarterback contract size in relation to a teams salary cap who simply make the playoffs vs. a quarterback who's team advances to the Divisional Round, Conference Championship, Super Bowl, or wins the Super Bowl.

# Conclusion

Additional Factors
    - There are more factors at play which contribute to a player's contract size. Data that is not quantifiable at the moment like a players intagible skills (leadership on and off the field, work ethic, relationships with other players or coaches, etc.) 


# Recommendations

# Next Steps