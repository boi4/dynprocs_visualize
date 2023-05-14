#!/usr/bin/env python3
import argparse
import json
from copy import deepcopy

import pandas as pd
from manim import *


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
        self.df = self.df.sort_values(by=['unixtimestamp'])

        # round and compute relative time
        if self.rounding is not None:
            # round down to nearest 10^rounding
            self.df['unixtimestamp'] = self.df['unixtimestamp'].apply(lambda x: x
                            if (x % pow(10, self.rounding) == 0) else x + pow(10, self.rounding) - x % pow(10, self.rounding))
        start_time = self.df['unixtimestamp'].min()
        self.df['time'] = (self.df['unixtimestamp'] - start_time)/1000
        print(self.df)


    def process_psetop(self, event_data, proc_states, psets):
        match event_data["op"]:
            case "null":
                pass
            case "add":
                pass
            case "sub":
                pass
            case "split":
                pass
            case "union":
                pass
            case "difference":
                pass
            case "intersection":
                pass
            case "grow":
                pass
            case "shrink":
                pass
            case "replace":
                pass


    def create_timeline(self):
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
        for i, row in self.df.iterrows():
            event = row["event"]
            event_data = row["event_data"]

            # print(row["time"])
            # print(psets)
            # print([(k,v.is_active())for (k,v) in proc_states.items()])
            # print("=================================")
            # print(event)

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
                    psets[event_data["id"]] = ProcessSetState(event_data["proc_ids"], row["time"])
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
                    self.process_psetop(event_data, proc_states, psets)
                case "finalize_psetop":
                    # ignore for now
                    pass
                case "application_message":
                    # ignore for now
                    pass
                case "application_custom":
                    # ignore for now
                    pass
                case _:
                    print("Warning: unknown event", event, "ignored")

            prev_events.append(event)

            print(i)
            print(len(self.df)-1)
            # only update if there is something visual to show and update after each event in the same time point has been processed
            if i == len(self.df)-1 or \
                (not all(ev in ["set_start", "job_start", "job_end"] for ev in prev_events) and \
                self.df.iloc[i+1]["time"] != row["time"]):
                print("Updating")
                self.ts.append(row["time"])
                self.job_states.append(JobState(row["job_id"],
                                                row["time"],
                                                deepcopy(proc_states),
                                                deepcopy(psets),
                                                events=deepcopy(prev_events)))
                self.events.append(row)
                prev_events = []

        print(self.job_states[0].process_states[11].is_active())
        # tun into numpy arrays
        self.ts = np.array(self.ts)
        self.job_states = np.array(self.job_states)
        self.events = np.array(self.events)


class VisualizeDynProcs(Scene):

    def __init__(self, timeline):
        super().__init__()
        self.timeline = timeline

    def construct(self):
        # Circle parameters
        axes_color = GREEN
        process_color = RED
        inactive_line_color = GREEN
        speed_factor = 3
        radius = 0.05
        bg_color = GREY

        self.camera.background_color = bg_color

        maxproc = np.max(self.timeline.procs)
        axes = Axes(
                    x_range=[self.timeline.ts[0], self.timeline.ts[-1], 1],
                    x_length=20,
                    y_range=[0, maxproc+1, 1],
                    axis_config={"color": axes_color},
                    x_axis_config={
                        # "numbers_to_include": np.arange(self.timeline.ts[0], self.timeline.ts[-1], 1),
                        # "numbers_with_elongated_ticks": np.arange(self.timeline.ts[0], self.timeline.ts[-1], 1),
                    },
                    y_axis_config={
                        "numbers_to_include": np.arange(1, len(self.timeline.procs)+1, 1),
                    },
                    tips=False,
                )


        axes.scale(0.6)
        axes.shift(2*UP)

        #labels = axes.get_axis_labels(x_label="time (s)", y_label="Proc")
        #self.add(axes, labels)
        self.add(axes)

        line_segments = []
        line_segment_lengths = []
        for t_0,t_1 in zip(self.timeline.ts[:-1], self.timeline.ts[1:]):
            lines = []
            line_lengths = []
            for proc in range(maxproc+1):
                l = Line(axes.c2p(t_0, proc+1), axes.c2p(t_1, proc+1), color=inactive_line_color, stroke_width=2)
                lines.append(l)
                line_lengths.append(t_1 - t_0)
            line_segments.append(lines)
            line_segment_lengths.append(line_lengths)

        # add line segments to scene
        #for lines in line_segments:
        #    self.add(*lines)

        # iterate over each time slice

        active_procs = []
        proc_to_dot = {}
        proc_to_trace = {}
        for i,elem  in enumerate(zip(self.timeline.ts[:-1], self.timeline.job_states[:-1])):
            t,job_state = elem

            # add and remove dots based on process state
            for proc, proc_state in job_state.process_states.items():
                if proc_state.is_active() and proc not in active_procs:
                    # setup dot
                    dot = Dot(radius=radius, color=process_color)

                    # move dot to correct position
                    proc_pos = axes.c2p(t, proc+1)
                    dot.move_to(proc_pos)

                    self.add(dot)

                    # create trace object
                    # https://docs.manim.community/en/stable/reference/manim.animation.movement.MoveAlongPath.html#manim.animation.movement.MoveAlongPath
                    trace = VMobject()
                    trace.move_to(proc_pos)
                    # need to capture dot and proc_pos in the lambda function
                    trace.add_updater(lambda x,dot=dot, proc_pos=proc_pos.copy():
                                      x.become(Line(proc_pos, dot.get_center()).set_color(process_color)))

                    # add trace
                    self.add(trace)

                    active_procs.append(proc)
                    proc_to_dot[proc] = dot
                    proc_to_trace[proc] = trace
                elif not proc_state.is_active() and proc in active_procs:
                    # remove dot
                    self.remove(proc_to_dot[proc])
                    self.remove(proc_to_trace[proc])
                    proc_to_dot.pop(proc)
                    proc_to_trace.pop(proc)
                    active_procs.remove(proc)



            # move each dot along the line segment
            animations = []
            for proc in active_procs:
                line_segment = line_segments[i][proc]
                line_segment_length = line_segment_lengths[i][proc]
                animation = MoveAlongPath(proc_to_dot[proc], line_segment, run_time=line_segment_length/speed_factor, rate_func=linear)
                animations.append(animation)
            self.play(*animations)

        self.wait(1)


def main(manim_config, log_file, rounding):
    tl = Timeline(log_file, rounding)
    with tempconfig(manim_config):
        VisualizeDynProcs(tl).render()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-file", "-l", type=str, default="log.csv")
    parser.add_argument("--quality", "-q", type=str, default="low_quality", choices=["low_quality", "medium_quality", "high_quality"])
    parser.add_argument("--preview", "-p", action="store_true", default=False)
    parser.add_argument("--round-to", "-r", type=int, default=2)
    # TODO: add mode argument and ability to do moving camera

    args = parser.parse_args()

    manim_config = {
        "quality": args.quality,
        "preview": args.preview,
    }

    main(manim_config, log_file=args.log_file, rounding=args.round_to)
