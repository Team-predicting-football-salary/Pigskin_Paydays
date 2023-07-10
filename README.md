# Pigskin Paydays

# Project Description 

In the NFL, every football team in the league is alloted a certain amount of money that it must then allocate amongst its players. This is known as the team's "cap." When players are up for contract renewal, the most valued players in every team, quarterbacks, ask for salaries far beyond what a team currently evaluates them at, and the two sides go back and forth before they reach a consensus based on the player's actual worth. Our project aims to predict what the percentage of the cap will be alloted for a player given his performance in a cycle.

# Project Goals

To predict what percentage of a team's cap will be paid out to the quarterback for his contract using objective stats.

# Project Planning

Acquisition
    - For player stats, acquire data from <a href='https://nextgenstats.nfl.com/stats/passing/2022/REG/all#yards'>Next Gen Stats</a>

Preparation

Exploration

Modeling


# Initial Hypothesis

Quarterbacks with the most passing yards and who extend their team's season will get paid the most money.

# Data Dictionary

Column Name | Description | Key
--- | --- | ---
player_name | The name of the quarterback | string
team | The team that the quarterback plays for | string
tt | Time to throw: Average amount of time elapsed from the time of snap to throw on every pass attempt for a passer (sacks excluded) | float
cay | Completed air yards: Average total distance in yards of all passes that were completed in the season | float
iay | Intended air yards: Average total distance in yards of all passes | float
ayd | Air yards differential: Difference between completed and intended air yards | float
agg | Aggressiveness (in percent): Percentage of attempts where the ball was thrown to a receiver where there was a defender within one yard distance | percentage, float
lcad | Longest completed air distance: Distance in yards of longest completed throw that season | float
ayts | Air yards to the sticks: Average distance in yards ahead or behind the first down marker for all attempted passes | float
att | Passing attempts: The number of times that the quarterback threw the ball that season | integer
yds | Yards: Total number of yards that the quarterback threw that season | integer
td | Touchdowns: Number of touchdowns that quarterback threw | integer
int | Intercepts: Number of times that the quarterback's throw was intercepted by the opposing team | integer
rate | The given rating that a player is given by the NFL | float
comp | Completion probability: Probability of a pass completion (actual) | percentage, float
xcomp | Expected completion percentage: Probability of a pass completion (expected) | percentage, float
comp_diff | A quarterback's actual completion percentage compared to their expected | percentage, float

# How to Reproduce

Download the repository and run through the notebook.

# Key Findings

# Conclusion

# Recommendations

# Next Steps