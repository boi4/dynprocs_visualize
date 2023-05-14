import json
from copy import deepcopy
from enum import Enum

import pandas as pd


class ProcessStateType(Enum):
    UNINITIALIZED = 0
    RUNNING = 1
    FINISHED = 2
    START_ANNOUNCED = 3
    SHUTDOWN_ANNOUNCED = 4
    ABOUT_TO_SHUTDOWN = 5
    INACTIVE = 6
    OTHER = 8

class ProcessState:
    def __init__(self, proc_id, statetype, last_pset_id=None):
        self.proc_id = proc_id
        self.statetype = statetype
        self.last_pset_id = last_pset_id

    def is_active(self):
        return self.statetype in [ProcessStateType.RUNNING, ProcessStateType.SHUTDOWN_ANNOUNCED, ProcessStateType.ABOUT_TO_SHUTDOWN]

    def is_visualized(self):
        return self.is_active() or self.statetype == ProcessStateType.START_ANNOUNCED


class JobState:
    def __init__(self, job_id, time, process_states, psets, events):
        self.job_id = job_id
        self.time = time
        self.process_states = process_states
        self.psets = psets
        self.events = events

class ProcessSetState:
    def __init__(self, proc_ids, created_at):
        self.proc_ids = proc_ids
        self.created_at = created_at


class Timeline:
    def __init__(self, log_file, rounding=None):
        self.log_file = log_file
        self.rounding = rounding
        self.ts = None
        self.job_states = None
        self.events = None

        self.read_files()
        self.create_timeline()

    def read_files(self):
        # read log files
        #log_files = glob.glob(f"{self.log_dir}/*.csv")
        #dfs = [pd.read_csv(log_file, names=columns) for log_file in log_files]
        #self.df = pd.concat(dfs, ignore_index=True)
        # parse json
        columns = ['unixtimestamp', 'job_id', 'event', 'event_data']
        self.df = pd.read_csv(self.log_file, names=columns, comment='#')
        self.df.dropna(how="all", inplace=True)
        self.df['event_data'] = self.df['event_data'].apply(json.loads)


        # order df by time
        # need stable sorting algorithjm
        self.df = self.df.sort_values(by=['unixtimestamp'], kind='mergesort')

        # round and compute relative time
        if self.rounding is not None:
            # round down to nearest 10^rounding
            self.df['unixtimestamp'] = self.df['unixtimestamp'].apply(lambda x: x
                            if (x % pow(10, self.rounding) == 0) else x + pow(10, self.rounding) - x % pow(10, self.rounding))
        start_time = self.df['unixtimestamp'].min()
        self.df['time'] = (self.df['unixtimestamp'] - start_time)/1000


    def process_psetop(self, event_data, proc_states, psets):
        op = event_data["op"]
        set_id = event_data["set_id"]
        input_sets = event_data["input_sets"]
        output_sets = event_data["output_sets"]
        initialized_by = event_data["initialized_by"]

        match op:
            # we do not need to handle these
            case "null":
                pass
            case "split":
                pass
            case "union":
                pass
            case "difference":
                pass
            case "intersection":
                pass

            # here we need to update process states
            case "grow":
                union_set = output_sets[0]
                delta_set = output_sets[1]
                for proc in psets[delta_set].proc_ids:
                    proc_states[proc].statetype = ProcessStateType.START_ANNOUNCED
            case "shrink":
                diff_set = output_sets[0]
                delta_set = output_sets[1]
                for proc in psets[delta_set].proc_ids:
                    proc_states[proc].statetype = ProcessStateType.SHUTDOWN_ANNOUNCED
            case "add":
                delta_set = output_sets[0]
                for proc in psets[delta_set].proc_ids:
                    proc_states[proc].statetype = ProcessStateType.START_ANNOUNCED
            case "sub":
                delta_set = output_sets[0]
                for proc in psets[delta_set].proc_ids:
                    proc_states[proc].statetype = ProcessStateType.SHUTDOWN_ANNOUNCED
            case "replace":
                s_new = output_sets[0]
                s_del = output_sets[1]
                s_repl = output_sets[2]
                for proc in psets[s_new].proc_ids:
                    proc_states[proc].statetype = ProcessStateType.START_ANNOUNCED
                for proc in psets[s_del].proc_ids:
                    proc_states[proc].statetype = ProcessStateType.SHUTDOWN_ANNOUNCED

        if op != "null":
            # update all procs in the output
            for pset in output_sets:
                for proc in psets[pset].proc_ids:
                    proc_states[proc].last_pset_id = pset



    def create_timeline(self):
        """
        construct a timeline by evaluating events in their order
        """
        new_pset_only_df = self.df[self.df["event"] == "new_pset"]
        proc_ids = new_pset_only_df.apply(lambda x: x["event_data"]["proc_ids"], axis=1)
        self.procs = sorted(list(set(x for l in proc_ids for x in l)))
        self.num_procs = len(self.procs)

        # determine state at each time moment, by evaluating events in their order
        self.ts = []
        self.job_states = []
        self.events = []

        proc_states = {proc: ProcessState(proc, ProcessStateType.UNINITIALIZED, None) for proc in self.procs}
        psets = {}

        # iterate over each event and update state
        prev_events = []
        prev_event_rows = []
        for i, row in self.df.iterrows():
            event = row["event"]
            event_data = row["event_data"]
            time = row["time"]

            # print(row["time"])
            # print(event)
            # print(psets)
            # print([(k,v.is_active())for (k,v) in proc_states.items()])
            # print("=================================")

            match event:
                case "set_start":
                    # make that mpi://world the last pset id for all procs that are started
                    pset = psets[event_data["set_id"]]
                    for proc in pset.proc_ids:
                        proc_states[proc].last_pset_id = event_data["set_id"]
                case "job_start":
                    # ignore for now
                    pass
                case "job_end":
                    # ignore for now
                    pass
                case "new_pset":
                    psets[event_data["id"]] = ProcessSetState(event_data["proc_ids"], time)
                    # print("new psets")
                    # print(psets)
                case "process_start":
                    proc_id = event_data["proc_id"]
                    if proc_id in proc_states:
                        proc_states[proc_id].statetype = ProcessStateType.RUNNING
                    else:
                        print(f"Ignoring invalid process_start event for proc {proc_id}")
                case "process_shutdown":
                    proc_id = event_data["proc_id"]
                    if proc_id in proc_states:
                        proc_states[proc_id].statetype = ProcessStateType.INACTIVE
                    else:
                        print(f"Ignoring invalid process_shutdown event for proc {proc_id}")
                case "psetop":
                    # updates proc_states
                    self.process_psetop(event_data, proc_states, psets)
                case "finalize_psetop":
                    # update pset states
                    for proc in proc_states:
                        if proc_states[proc].statetype == ProcessStateType.SHUTDOWN_ANNOUNCED:
                            proc_states[proc].statetype = ProcessStateType.ABOUT_TO_SHUTDOWN
                        elif proc_states[proc].statetype == ProcessStateType.START_ANNOUNCED:
                            proc_states[proc].statetype = ProcessStateType.RUNNING
                case "application_message":
                    # ignore for now
                    pass
                case "application_custom":
                    # ignore for now
                    pass
                case _:
                    print("Warning: unknown event", event, "ignored")

            prev_events.append(event)
            prev_event_rows.append(row)

            # only update if there is something visual to show and update after each event in the same time point has been processed
            if i == len(self.df)-1 or \
                (not all(ev in ["set_start", "job_start", "job_end"] for ev in prev_events) and \
                 self.df.iloc[i+1]["time"] != time):
                self.ts.append(time)
                self.job_states.append(JobState(row["job_id"],
                                                time,
                                                deepcopy(proc_states),
                                                deepcopy(psets),
                                                events=deepcopy(prev_events)))
                self.events.append(deepcopy(prev_event_rows))
                prev_events = []
                prev_event_rows = []

        print(f"Reduced {len(self.df)} events to {len(self.ts)} time points")
