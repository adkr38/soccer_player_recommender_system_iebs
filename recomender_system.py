from multiprocessing.sharedctypes import Value
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import iebs_formulas
import unicodedata
np.seterr(divide='ignore', invalid='ignore')

class PlayerRecomender:
    def __init__(self,dataframe:pd.DataFrame):
        self.dataframe = dataframe
        self._config = {
            "selected_player":"",
            "filter_by_position":False,
            "filter_by_statGroup":False,
            "filter_games_played":True,
            "filter_by_age":False,
            "filter_by_nationality":False,
            "players_to_recommend":5,
            "fill_na":"mean",
            "scaling":StandardScaler()}
        
        self._pca = None
        self._kmeans = None
        self._filtered_frame = None

        self.dataframe["player"] = self.dataframe["player"].apply(lambda x: unicodedata.normalize("NFD",x).encode("ascii","ignore").decode("utf-8").strip().lower())
        for i,index in enumerate(self.dataframe[self.dataframe.duplicated("player") == True].T):
            self.dataframe.loc[index,"player"] = self.dataframe.loc[i,"player"]+f"_{i}"

    def preprocess_frame(self,**kwargs):
        """
        * :kwarg games_played_column_name str:  WILL ONLY BE PASSED IF filter_games_played : True ON CONFIG - column name for 90m's/games played -> default '90m'.
        * :kwarg min_games_played int: WILL ONLY BE PASSED IF filter_games_played : True ON CONFIG - minimum of games played by a player to be taken into account -> default 10.
        * :kwarg position_column_name str: WILL ONLY BE PASSED IF filter_by_position : True ON CONFIG - positions column name -> default 'position'.
        * :kwarg position str: WILL ONLY BE PASSED IF filter_by_position : True ON CONFIG - selected position to filter -> default 'MF'.
        * :kwarg statGroup_columns list: WILL ONLY BE PASSED IF filter_by_statGroup : True ON CONFIG - desired column names to evaluate, rest will be dropped.
        * :kwarg age_column str: WILL ONLY BE PASSED IF filter_by_age : True ON CONFIG -  age column name.
        * :kwarg age_range list: [min,max] -> default [0,99].
        * :kwarg nationality_column str: nationality column name.
        * :kwarg nationality str: desired nationality -> default 'ES'.
        #### :returns: pd.DataFrame with preprocessed data.
        """

        frame = self.dataframe.copy()
        player_row = frame.loc[frame["player"] == self.config["selected_player"]]
        player_row.set_index("player",inplace=True)

        if self.config.get("filter_games_played",False):
            frame = frame.loc[frame[kwargs.get("games_played_column_name","90m")] > kwargs.get("min_games_played",10)]
            frame.drop(kwargs.get("games_played_column_name","90m"),axis=1,inplace=True)
            player_row.drop(kwargs.get("games_played_column_name","90m"),axis=1,inplace=True)


        if self.config.get("filter_by_age",False):
            frame = frame.loc[(frame[kwargs.get("age_column","age")] > kwargs.get("age_range")[0]) & (frame[kwargs.get("age_column","age")] < kwargs.get("age_range")[1])]
            frame.drop(kwargs.get("age_column","age"),axis=1,inplace=True)
            player_row.drop(kwargs.get("age_column","age"),axis=1,inplace=True)


        if self.config.get("filter_by_position",False):
            frame = frame.loc[lambda x: x[kwargs.get("games_played_column_name","position")].str.contains(r"{}|{}|{}".format(kwargs.get("position","MF").upper(),kwargs.get("position","MF").lower(),kwargs.get("position","MF").capitalize()))]
            frame.drop(kwargs.get("games_played_column_name","position"),axis=1,inplace=True)
            player_row.drop(kwargs.get("games_played_column_name","position"),axis=1,inplace=True)


        if self.config.get("filter_by_nationality",False):
            frame = frame.loc[lambda x: x[kwargs.get("nationality_column","nationality")].str.contains(r"{}|{}|{}".format(kwargs.get("nationality","ES").upper(),kwargs.get("position","ES").lower(),kwargs.get("position","ES").capitalize()))]
            frame.drop(kwargs.get("nationality_column","nationality"),axis=1,inplace=True)
            player_row.drop(kwargs.get("nationality_column","nationality"),axis=1,inplace=True)


        if self.config.get("filter_by_statGroup",False):
            frame = frame[kwargs.get("statGroup_columns",frame.columns)]
            player_row = player_row[kwargs.get("statGroup_columns",frame.columns)]

        frame.set_index("player",inplace=True)

        if self._config["selected_player"] not in frame.index.to_list():
            frame = frame.append(player_row)

        frame.drop([x for x in frame if frame[x].dtype in ["string","object"]],axis=1,inplace=True)

        self._filtered_frame = frame.fillna(frame.mean(axis=1))

        inputer = SimpleImputer(strategy=self.config["fill_na"])
        scaler = self.config["scaling"]
        pca_components = PlayerRecomender._calc_pca_components(frame.apply(lambda x:x.fillna(x.mean())))
        pca = PCA(n_components = pca_components)
        pipe = Pipeline(steps=[("inputer",inputer),("scaler",scaler),("pca",pca)])

        preprocessed_dataframe = pd.DataFrame(pipe.fit_transform(frame),index=frame.index,columns=[f"component_{x+1}" for x in range(pca_components)])
        self._pca = pca
        self.preprocessed_dataframe = preprocessed_dataframe


    def apply_k_means(self):

        def get_shilouette_coefficient(dataframe:pd.DataFrame):
            cluster_shilouette_scores = {}
            shilouette_coefficient = 0

            for i in range(5,13):
                km = KMeans(n_clusters=i)
                km_fitted = km.fit_predict(dataframe)
                score = silhouette_score(dataframe, km.labels_, metric='euclidean')
                if score > shilouette_coefficient:
                    shilouette_coefficient = i
                cluster_shilouette_scores[i] = score

            return shilouette_coefficient
        
        dataframe = self.preprocessed_dataframe.copy()
        
        kmeans = KMeans(n_clusters = get_shilouette_coefficient(dataframe=dataframe))
        
        dataframe["cluster"] = kmeans.fit_predict(dataframe)
        self._kmeans = kmeans

        return dataframe
    @classmethod
    def _get_cluster_component_importance_image(cls,kmeans_frame):
        plt.figure(figsize=(13,4))
        cmap = sns.cubehelix_palette(as_cmap=True, reverse=True)
        ax = sns.heatmap(kmeans_frame.groupby("cluster").mean(),annot=True,cmap = cmap,lw=0.5)
        fig = ax.get_figure()
        return fig

    @classmethod
    def _get_clusters_image(cls,selected_player:str,recomended_players:list,kmeans_frame:pd.DataFrame,zoomed=False):
        player_input_vector = kmeans_frame.loc[kmeans_frame.index == selected_player][["component_1","component_2"]].T.to_dict()[selected_player]
        recomended_vectors = {}
        for player in recomended_players:
            recomended_vectors.update(kmeans_frame.loc[kmeans_frame.index == player][["component_1","component_2"]].T.to_dict())
        max_x,min_x, max_y,min_y = kmeans_frame["component_1"].max(),kmeans_frame["component_1"].min() ,kmeans_frame["component_2"].max(),kmeans_frame["component_2"].min()

        range_for_x, range_for_y = [player_input_vector["component_1"] -8 if player_input_vector["component_1"] - 8 > min_x else min_x, player_input_vector["component_1"] +8 if player_input_vector["component_1"] +8 < max_x else max_x] , [player_input_vector["component_2"] -8 if player_input_vector["component_2"] -8 > min_y else min_y,player_input_vector["component_2"] +8 if player_input_vector["component_2"] +8 < max_y else max_y]
        plt.figure(figsize=(50,23))

        fig = px.scatter(title = "Overview" if not zoomed else "Zoomed in",data_frame=kmeans_frame,x="component_1",y="component_2",color="cluster",hover_name=kmeans_frame.index,color_continuous_scale=px.colors.qualitative.Dark24)
        fig.update_traces(marker={'size': 4,'opacity':0.5})
        if zoomed:
            fig.update_xaxes(range=range_for_x, row=1, col=1)
            fig.update_yaxes(range=range_for_y, row=1, col=1)

        fig.update(layout_coloraxis_showscale=False)
        fig.update(layout_showlegend=False)
        fig.add_trace(go.Scatter(name=selected_player,x=[player_input_vector["component_1"]], y=[player_input_vector["component_2"]], mode = 'markers+text',
                                marker_symbol = 'circle-open',
                                marker_color = "black",
                                marker_size = 15,
                                text = [selected_player] if zoomed else [],
                                textposition="bottom center",
                                opacity=1,
                                ))

                                
        for recomended_player in recomended_vectors:
            fig.add_trace(go.Scatter(name=recomended_player,x=[recomended_vectors[recomended_player]["component_1"]], y=[recomended_vectors[recomended_player]["component_2"]], mode = 'markers+text',
                                    marker_symbol = 'square-open',
                                    marker_color = "orange",
                                    text=[recomended_player] if zoomed else [],
                                    textposition="bottom center",
                                    marker_size = 10,
                                    opacity=1,
                                    ))

        return fig
    @classmethod
    def _get_cluster_most_important_feats_image(cls,pca_components,preprocessed_columns,filtered_columns):
        feature_importance = pd.DataFrame(pca_components,index =preprocessed_columns , columns= filtered_columns).T
        component_1_top_5, component_2_top_5,component_3_top_5 = pd.DataFrame(feature_importance["component_1"].abs().sort_values(ascending=False).head()), pd.DataFrame(feature_importance["component_2"].abs().sort_values(ascending=False).head()),pd.DataFrame(feature_importance["component_3"].abs().sort_values(ascending=False).head())
        component_1_top_5.columns,component_2_top_5.columns,component_3_top_5.columns = ["efect_on_component_1"],["efect_on_component_2"],["efect_on_component_3"]
        fig = plt.figure(figsize=(30,4))
        cmap = sns.cubehelix_palette(start=2.9,light=0.9,as_cmap=True,reverse=True)
        a,b,c = 1,3,1
        plt.subplot(a,b,c)
        ax_1 = sns.heatmap(component_1_top_5,annot=True,lw=1,cmap=cmap)
        plt.subplot(a,b,c+1)
        ax_2 = sns.heatmap(component_2_top_5,annot=True,lw=1,cmap=cmap)
        plt.subplot(a,b,c+2)
        ax_3 = sns.heatmap(component_3_top_5,annot=True,lw=1,cmap=cmap)
        fig = ax_3.get_figure()

        return fig

    @classmethod
    def _calc_pca_components(cls,data:pd.DataFrame,thresh=0.94):

        data = StandardScaler().fit_transform(data)

        while True:
            for i in range(2,30):
                    
                pca_instance = PCA(n_components=i)
                pca_instance.fit(data)
                if i == 29:
                    return len([x for x in pca_instance.explained_variance_ratio_ if x >= 0.02])

                if pca_instance.explained_variance_.sum() <= thresh:
                    return i
        
    @classmethod
    def _calc_similarity_vector(cls,frame,selected_player):
        similarity_dict = {}
        frame.index =  [x.strip().lower() for x in frame.index]  

        selected_player_vector = frame.loc[frame.index == selected_player].to_numpy()

        for player in frame.index:
            try:
                similarity_dict[player] = iebs_formulas.average_distance(selected_player_vector,frame.loc[frame.index == player].to_numpy())

            except ValueError as e:
                raise ValueError(e,player,frame.loc[frame.index == player].to_numpy())

        return pd.DataFrame([similarity_dict],index=["average_distance"]).T
    
    def view_current_config_settings(self):
        print(self._config)
    
    def get_visuals(self,recomended_players:list,kmeans_frame:pd.DataFrame):
        similarity_zoomed_visual,similarity_visual,cluster_pca_importance_visual,cluster_attrs_visual = PlayerRecomender._get_clusters_image(selected_player = self.config["selected_player"],recomended_players=recomended_players,kmeans_frame=kmeans_frame,zoomed=True), PlayerRecomender._get_clusters_image(selected_player = self.config["selected_player"],recomended_players=recomended_players,kmeans_frame=kmeans_frame), PlayerRecomender._get_cluster_component_importance_image(kmeans_frame),PlayerRecomender._get_cluster_most_important_feats_image(pca_components=self._pca.components_,preprocessed_columns=self.preprocessed_dataframe.columns,filtered_columns=self._filtered_frame.columns)
        similarity_zoomed_visual.write_image("images/z.png"),similarity_visual.write_image("images/a.png"), cluster_pca_importance_visual.savefig("images/b.png"), cluster_attrs_visual.savefig("images/c.png")
    
    def recommend(self):
        player_name = self.config["selected_player"]

        if not hasattr(self,"preprocessed_dataframe"):
            raise NotImplementedError("Dataframe not preprocessed.")

        distance_vector = PlayerRecomender._calc_similarity_vector(frame = self.preprocessed_dataframe,selected_player = player_name)       
        if not len(distance_vector):
            print(f"No matches available for {player_name}")
        distance_vector.drop(player_name,inplace=True)
        distance_vector.sort_values(by="average_distance",ascending=True,inplace=True) 

        recommendations_raw =  distance_vector.head(self.config["players_to_recommend"])

        attribute_similarity_dict = PlayerRecomender._find_most_similar_attributes_for_matches(player_name,recommendations_raw.index.to_list(),self.dataframe)
        kmeans_frame = self.apply_k_means()
        self.get_visuals(recomended_players = recommendations_raw.index.to_list(), kmeans_frame = kmeans_frame)
        player_cluster = pd.DataFrame([{"player":player_name,"cluster_number":kmeans_frame.loc[kmeans_frame.index == player_name]["cluster"][0]} for player_name in [player_name] + recommendations_raw.index.to_list()])

        with pd.ExcelWriter(f"outputs/{player_name.replace(' ','_')}.xlsx","xlsxwriter") as writer:

            recommendations_raw.to_excel(writer,"recommendations")
            main_worksheet = writer.sheets['recommendations']
            for col in range(len(recommendations_raw.columns)):
                main_worksheet.set_column(1,col,15)
            main_worksheet.set_column(1,4,7)

            main_worksheet.insert_image('O1', 'images/z.png')
            main_worksheet.insert_image('D1', 'images/a.png')

            player_cluster.to_excel(writer,"cluster_analysis",index=False)
            cluster_worksheet = writer.sheets['cluster_analysis']
            for i in range(len(player_cluster.columns)):
                cluster_worksheet.set_column(1,i,15)
            cluster_worksheet.insert_image("B11","images/b.png")
            cluster_worksheet.insert_image("B33","images/c.png")

            for i,player in enumerate(attribute_similarity_dict):
                frame = pd.DataFrame(attribute_similarity_dict[player])
                frame.to_excel(writer,sheet_name=f"player_{i+1}")
                suggested_sheet = writer.sheets[f"player_{i+1}"]
                for col in range(len(frame.columns)):
                    suggested_sheet.set_column(1,col,15)
                    suggested_sheet.set_column(2,col,15)

            print(f"{player_name}.xlsx built correctly.")




    @classmethod
    def _find_most_similar_attributes_for_matches(cls,selected_player:str,matches:list,raw_input_dataframe:pd.DataFrame):
        output_dict = {}

        players = [selected_player] + matches
        frame_for_similar_attributes = raw_input_dataframe.copy()
        frame_for_similar_attributes = frame_for_similar_attributes.loc[frame_for_similar_attributes["player"].isin(players)]
        frame_for_similar_attributes.set_index("player",inplace=True)
        frame_for_similar_attributes.drop([x for x in frame_for_similar_attributes if frame_for_similar_attributes[x].dtype in ["string","object"]],axis=1,inplace=True)
        frame_for_similar_attributes = frame_for_similar_attributes.T[(frame_for_similar_attributes!= 0).all()].T
        frame_for_similar_attributes = pd.DataFrame(StandardScaler().fit_transform(frame_for_similar_attributes),index=frame_for_similar_attributes.index,columns = frame_for_similar_attributes.columns)
        
        def find_most_similar_attributes(frame,player_a,player_b):
            most_similar_feats = (frame.T[player_a] - frame.T[player_b]).abs().sort_values()[:5].index
            return most_similar_feats

        for player in matches:
            attributes = find_most_similar_attributes(frame= frame_for_similar_attributes,player_a = selected_player,player_b = player).to_list() + ["player"]
            frame_out = raw_input_dataframe.loc[raw_input_dataframe["player"].isin([selected_player,player])][attributes]
            frame_out.set_index("player",inplace=True)
            output_dict.update({player:frame_out.T.to_dict()})
        
        return output_dict


    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self,config):
        if not len(config):
            raise AttributeError("Empty dictionary passed.")
        self._config = config