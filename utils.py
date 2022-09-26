import pandas as pd
from sklearn.preprocessing import StandardScaler

def config_cli(data:pd.DataFrame):
        # """
        # * :kwarg games_played_column_name str:  WILL ONLY BE PASSED IF filter_games_played : True ON CONFIG - column name for 90m's/games played -> default '90m'.
        # * :kwarg min_games_played int: WILL ONLY BE PASSED IF filter_games_played : True ON CONFIG - minimum of games played by a player to be taken into account -> default 10.
        # * :kwarg position_column_name str: WILL ONLY BE PASSED IF filter_by_position : True ON CONFIG - positions column name -> default 'position'.
        # * :kwarg position str: WILL ONLY BE PASSED IF filter_by_position : True ON CONFIG - selected position to filter -> default 'MF'.
        # * :kwarg statGroup_columns list: WILL ONLY BE PASSED IF filter_by_statGroup : True ON CONFIG - desired column names to evaluate, rest will be dropped.
        # * :kwarg age_column str: WILL ONLY BE PASSED IF filter_by_age : True ON CONFIG -  age column name.
        # * :kwarg age_range list: [min,max] -> default [0,99].
        # * :kwarg nationality_column str: nationality column name.
        # * :kwarg nationality str: desired nationality -> default 'ES'.
        # #### :returns: pd.DataFrame with preprocessed data.
        # """

    base_config = {
            "selected_player":"",
            "filter_by_position":False,
            "filter_by_statGroup":False,
            "filter_games_played":True,
            "filter_by_age":False,
            "filter_by_nationality":False,
            "players_to_recommend":5,
            "fill_na":"mean",
            "scaling":StandardScaler()}
    
    #defaults
    kwargs = {
        "min_games_played":10,
        "position":"MF",
        "age_range":[0,99],
        "nationality":"ES"
    }


    while True:
        games_played = input("\nMinimum of games played - default 10:")
        if not games_played.isdigit():
            print("Please input a number.")
            continue

        kwargs["min_games_played"] = int(games_played)

        break

    #Position
    while True:    
        position = input(f"\nPosition filtering: Input 0 if you'd like to skip this step.\nPositions available:{data['position'].unique().tolist()}\n").strip()

        if position == "0":
            break

        if position not in data['position'].unique().tolist():
            print("Invalid position, please input a correct playing position.")
            continue

        base_config["filter_by_position"] = True
        kwargs["position"] = position

        break
    
    #Stats
    while True:

        statGroup = input("\nStatGroup filtering: Input 0 if you'd like to skip this step.\nPlease input column names you'd like to focus on separated by commas: ").strip()

        if statGroup == "0":
            break

        stats = [x.split(",") for x in statGroup]

        statsOk=True
        for x in stats:
            if x not in data.columns.tolist():
                print(f"{x} not available in selected dataframes columns.")
                statsOk=False
                break

        if not statsOk:
            continue

        base_config["filter_by_statGroup"] =  True
        kwargs["statGroup_columns"] = stats
        break

    #Age
    while True:

        age_input = input("\nAge filtering: Input 0 if you'd like to skip this step.\nPlease input desired age range in the following format -> min_age,max_age : ").strip()

        if age_input == "0":
            break

        split_input = age_input.split(",")
        if len(split_input) == 1:
            print("Invalid selection, please follow the input format: min_age,max_age")
            continue
        ageOk = True
        for age in split_input:
            if not age.isdigit():
                print(f"{age} is not a valid input. Please try again.")
                ageOk = False
                break

        if not ageOk:
            continue

        age_range = [int(split_input[0].strip()),int(split_input[1].strip())]

        base_config["filter_by_age"] =  True
        kwargs["age_range"] = age_range
        break
    
    #Nationality
    while True:

        nationality_input = input(f"\nNationality filtering: Input 0 if you'd like to skip this step. \nPlease select your desired nationality from the following list:\n{data['nationality'].unique().tolist()} : ").strip()

        if nationality_input == "0":
            break

        if nationality_input not in data['nationality'].unique().tolist():
            print("Invalid nationality, please provide a correct input.")
            continue

        base_config["filter_by_nationality"] =  True
        kwargs["nationality"] = nationality_input
        break
    
    return base_config,kwargs