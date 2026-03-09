import os
import time

import numpy
import numpy as np
import json
import pandas as pd
# import rospy

ROOT_FOLDER = "/home/kinovaresearch/Desktop/TRUST_AND_TRANSPARENCY_USER_STUDY"
TASKS = [
    "shelving",
    "sorting"
]
TREATMENTS = {
    "A" : "Direct Teleoperation",
    "B" : "VOSA Baseline",
    "C" : "Sparse Audio",
    "D" : "Rich Audio",
    "E" : "Sparse Visual",
    "F" : "Rich Visual"
}

class UserStudyExperiment:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(UserStudyExperiment, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.timestamp = int(time.time())

    @staticmethod
    def new_user():
        exp = UserStudyExperiment()
        user_dirs = exp.get_user_dirs()
        exp.user_id = max(user_dirs) + 1 if len(user_dirs) > 0 else 0
        exp.dir = os.path.join(ROOT_FOLDER, str(exp.user_id))
        os.mkdir(exp.dir)
        exp.treatment_order = exp.assign_treatment_order()
        exp.print_treatments()
        exp.create_folder_tree()
        exp.set_active_user()

    def assign_treatment_order(self):
        experiments = {}

        # Add Familiarity Task
        for task in TASKS:
            experiments[task] = []
            treatments = list(TREATMENTS.keys())
            if task != "Reaching":
                np.random.shuffle(treatments)
            for i, treatment in enumerate(treatments):
                experiments[task].append(f"{i + 1}. {TREATMENTS[treatment]} ({treatment})")

        with open(os.path.join(self.dir, "treatments.json"), "w") as f:
            json.dump(experiments, f)

        return experiments

    def print_treatments(self):
        print(f"USER {self.user_id} ASSIGNED TO RANDOMIZED TREATMENT ORDER")
        for k in self.treatment_order:
            print(f"\n{k}")
            for t in self.treatment_order[k]:
                print(f"\t{t}")

    def create_folder_tree(self):
        for task in TASKS:
            task_dir = os.path.join(self.dir, task)
            os.mkdir(task_dir)
            for treatment in TREATMENTS:
                treatment_dir = os.path.join(task_dir, treatment)
                os.mkdir(treatment_dir)

    def set_active_user(self):
        np.savetxt(os.path.join(ROOT_FOLDER, "curr_user.txt"), np.array([self.user_id], dtype=np.int8), fmt="%5u")

    @staticmethod
    def get_active_user():
        user = np.loadtxt(os.path.join(ROOT_FOLDER, "curr_user.txt"), dtype=np.int8)
        return user

    def get_user_dir(self, task, treatment):
        user = UserStudyExperiment.get_active_user()
        path = os.path.join(ROOT_FOLDER, str(user), task, treatment, str(self.timestamp))
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    # @staticmethod
    # def record_result(task, mode, task_duration, input_magnitude, t_f=time.time()):
    #     mode_translate = {0: "A", 1: "B", 2: "C"}
    #
    #     assert task in TASKS
    #     assert mode in mode_translate
    #
    #     user = UserStudyExperiment.get_active_user()
    #     _dir = os.path.join(ROOT_FOLDER, str(user), task, mode_translate[mode], str(int(t_f)))
    #     os.mkdir(_dir)
    #
    #     columns = ["task_duration", "cumulative_input"]
    #     data = np.array([[task_duration, input_magnitude]])
    #
    #     df = pd.DataFrame(data, columns=columns)
    #     df.to_csv(os.path.join(_dir, f"results.csv"), index=False)

    @staticmethod
    def get_user_dirs():
        """
        Retrieve All Folder IDS from the ROOT_FOLDER containing a user experiment
        """
        dirs = []
        for d in os.listdir(ROOT_FOLDER):
            try:
                name = int(d)
            except Exception as e:
                continue
            dirs.append(name)
        return dirs


if __name__ == "__main__":
    study = UserStudyExperiment().new_user()
    active_user = UserStudyExperiment.get_active_user()
    print(f"PERSISTENT EXPERIMENT USER SET TO: {active_user}")