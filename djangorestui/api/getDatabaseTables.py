import os
import pandas as pd
import sqlite3
import json

# to handle double quotes in json
class mydict(dict):
        def __str__(self):
            return json.dumps(self)
class DataBaseTable:

    def __init__(self):
        self.url = 'http://10.0.0.165:8000/simulation/'
        self.db_path = 'db.sqlite3'

    def get_table(self,_id = None,song_name = None):
        if os.path.isfile(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                if _id :
                    cursor.execute("SELECT * from api_loadsongs where id={};".format(_id))
                    results= pd.DataFrame.from_records(data =cursor.fetchall(), columns = self.get_col_names())
                if sim_name:
                    cursor.execute("SELECT * from api_loadsongs where name_of_the_song = ?", [song_name])
                    results= pd.DataFrame.from_records(data =cursor.fetchall(), columns = self.get_col_names())
                # cursor.fetchall()
                conn.commit()
                conn.close()    
                return results
            except:
                print ("Nothing")

    def get_sims_to_run_by_id(self, _id):
        """This function retrieves the simulations yet to be run from db """
        results = self.get_table(_id = _id)
        # print("results",results)
        if results.shape[0] != 0 :
            for i in results[results.id == _id].values:
                sim = mydict()
                for k,v in zip(self.get_col_names(),i):
                    sim[k] = v
            return sim
    
    def get_sims_to_run_by_name(self, sim_name):
        """This function retrieves the simulations yet to be run from db """
        sims_to_run = []
        results = self.get_table(sim_name = sim_name)
        if results.shape[0] != 0:
            for i in results[results.simulation_name == sim_name].values:
                sim = mydict()
                for k,v in zip(self.get_col_names(),i):
                    sim[k] = v
                # sims_to_run.append(sim)
            return sim
    
    def get_col_names(self):
        if os.path.isfile(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
    #             print(True)
            except:
                print(False)
        conn.row_factory = sqlite3.Row # to get the schema/col names
        cursor = conn.cursor()
        cursor.execute("SELECT * from api_simulationinputs limit 1;")
        r = cursor.fetchone()
        conn.commit()
        conn.close()
        return r.keys()
  
class getdataBaseTable:

    # @staticmethod
    def return_table_by_sim_id(_id):
        temp = DataBaseTable()
        table = temp.get_sims_to_run_by_id(_id)
        return table

    def return_table_by_sim_name(sim_name):
        temp = DataBaseTable()
        table = temp.get_sims_to_run_by_name(sim_name)
        return table

# print(getdataBaseTable.return_table())