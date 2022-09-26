import recomender_system
import pandas as pd
import os
import utils
import unicodedata


def main():

    dataframe_dir = ""

    if not os.path.exists("outputs/"):
        os.mkdir("outputs")
    
    if not os.path.exists("images/"):
        os.mkdir("images")

    while True:
        dataframe_dir = input("Please select your .csv directory:").strip()
        if not os.path.exists(dataframe_dir):
            print("Path doesn't exist, please re-try")
            continue

        break


    config,params = None,None
    dataframe = pd.read_csv(dataframe_dir)
    rec_instance = recomender_system.PlayerRecomender(dataframe=dataframe)
    data = rec_instance.dataframe.copy()

    while True:
        optionals = input("Input 1 if you'd like to access optional preprocessing parameters\nInput 0 if you'd like to continue with no filter applications:")
        if not optionals.isdigit():
            print("Invalid input")
            continue
        optionals = int(optionals)
        break
    
    if optionals:
        config,params = utils.config_cli(data = data)
    
    if config is not None:
        rec_instance.config = config

    while True:
        tries = 0
        player_input = input("\nPlease select your player of choice: ")
        if unicodedata.normalize("NFD",player_input).encode("ascii","ignore").decode("utf-8").strip().lower() not in data["player"].to_list():
            print(f"{player_input} not in dataframe. Please input a valid player.")
            if tries >=3:
                print(data["player"].to_list())
            continue
        break

    player_input = unicodedata.normalize("NFD",player_input).encode("ascii","ignore").decode("utf-8").strip().lower()

    while True:
        n_recommendations = input("\nPlease select how many recommendations you'd like: ")
        if not n_recommendations.isdigit():
            print("Please provide an integer.")
            continue
        break
    
    if config is not None:
        rec_instance.config = config

    rec_instance.config["selected_player"] = player_input
    rec_instance.preprocess_frame(**params if params is not None else {})
    rec_instance.config["players_to_recommend"] = int(n_recommendations)
    rec_instance.recommend()

if __name__ == "__main__":
    main()