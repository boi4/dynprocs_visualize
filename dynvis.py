#!/usr/bin/env python3
import argparse
from copy import deepcopy

from manim import *

from timeline import *


class VisualizeDynProcs(Scene):

    def __init__(self, timeline):
        super().__init__()
        self.timeline = timeline

    def construct(self):
        # Circle parameters
        self.axes_color = GREEN
        self.text_color = BLACK

        self.process_color = {
            ProcessStateType.INACTIVE: GREY,
            ProcessStateType.UNINITIALIZED: GREY,
            ProcessStateType.START_ANNOUNCED: GREY,
            ProcessStateType.RUNNING: RED,
            ProcessStateType.SHUTDOWN_ANNOUNCED: RED,
            ProcessStateType.ABOUT_TO_SHUTDOWN: RED,
        }
        self.trace_color = {
            ProcessStateType.INACTIVE: GREY,
            ProcessStateType.UNINITIALIZED: GREY,
            ProcessStateType.START_ANNOUNCED: GREY,
            ProcessStateType.RUNNING: RED,
            ProcessStateType.SHUTDOWN_ANNOUNCED: BLUE,
            ProcessStateType.ABOUT_TO_SHUTDOWN: PURPLE,
        }
        self.inactive_line_color = GREEN
        self.speed_factor = 3
        self.bg_color = WHITE


        self.camera.background_color = self.bg_color
        self.maxproc = np.max(self.timeline.procs)

        self.pset_text_line = 0
        self.pset_text_max_lines = 10
        self.application_text_line = 0
        self.application_text_max_lines = 5


        # setup axes
        self.setup_axes()

        # iterate over each time slice and add/remove dots based on process state
        self.proc_to_dot = {}
        self.proc_to_trace = {}
        prev_job_state = JobState(None, None, {proc_id: ProcessState(proc_id, ProcessStateType.UNINITIALIZED) for proc_id in range(self.maxproc+1)}, None, None)

        for i,elem  in enumerate(zip(self.timeline.ts[:-1], self.timeline.job_states[:-1])):
            t,job_state = elem
            self.time = t

            # process events
            self.process_events(self.timeline.events[i], job_state)

            # move processes on the process line
            self.update_process_lines(i, job_state, prev_job_state)
            prev_job_state = job_state


        # turn remaining dots into crosses
        for dot in self.proc_to_dot.values():
            self.bring_to_front(dot)
            dot.become(Cross(dot, color=PURE_RED))

        self.wait(1)

    def setup_axes(self):
        # setup coordinate system
        self.axes = Axes(
                    x_range=[self.timeline.ts[0], self.timeline.ts[-1], 1],
                    x_length=20,
                    y_range=[0, self.maxproc+1, 1],
                    axis_config={"color": self.axes_color},
                    x_axis_config={
                        "numbers_to_include": np.arange(self.timeline.ts[0], self.timeline.ts[-1], 5),
                        # "numbers_with_elongated_ticks": np.arange(self.timeline.ts[0], self.timeline.ts[-1], 1),
                    },
                    y_axis_config={
                        "numbers_to_include": np.arange(1, len(self.timeline.procs)+1, 1),
                    },
                    tips=False,
                )

        # fix color
        self.axes.get_x_axis().numbers.set_color(self.text_color)
        self.axes.get_y_axis().numbers.set_color(self.text_color)
        self.axes.get_x_axis().get_tick_marks().set_color(self.text_color)
        self.axes.get_y_axis().get_tick_marks().set_color(self.text_color)


        self.axes.scale(0.6)
        self.axes.shift(2*UP)

        labels = self.axes.get_axis_labels(x_label="time", y_label="Proc")
        self.add(self.axes, labels)

        # compute line segments that should be followed by the processes
        self.line_segments = []
        self.line_segment_lengths = []
        for t_0,t_1 in zip(self.timeline.ts[:-1], self.timeline.ts[1:]):
            lines = []
            line_lengths = []
            for proc in range(self.maxproc+1):
                l = Line(self.axes.c2p(t_0, proc+1), self.axes.c2p(t_1, proc+1), color=self.inactive_line_color, stroke_width=2)
                lines.append(l)
                line_lengths.append(t_1 - t_0)
            self.line_segments.append(lines)
            self.line_segment_lengths.append(line_lengths)


    def process_events(self, orig_rows, jobstate):
        # process events that happen at that time point

        application_messages = []
        pset_messages = []
        for event_row in orig_rows:
            event_data = event_row["event_data"]
            match event_row["event"]:
                case "application_message":
                    application_messages.append(event_data["message"])
                case "psetop":
                    op = event_data["op"]
                    set_id = event_data["set_id"]
                    input_sets = event_data["input_sets"]
                    output_sets = event_data["output_sets"]
                    pset_messages.append(f"<b>{op.upper()}</b>({', '.join(input_sets)}) <b>→</b> ({', '.join(output_sets)})")
                case "finalize_psetop":
                    pset_messages.append(f"<b>Finalize</b>")

        if pset_messages:
            self.add_event_text("\n".join(pset_messages), base=2)
        if application_messages:
            self.add_event_text("\n& ".join(application_messages), base=1, markup=False)


    def add_event_text(self, text, base, markup=True):
        text = text.strip()
        scale = 0.2
        line_height = scale
        num_lines = text.count("\n") + 1

        if self.pset_text_line + num_lines > self.pset_text_max_lines:
            self.pset_text_line = 0


        for line in text.split("\n"):
            cls = MarkupText if markup else Text
            t = cls(line, color=self.text_color).scale(scale)
            t.move_to(self.axes.c2p(self.time, 0)).shift((base + self.pset_text_line * line_height) * DOWN)
            self.pset_text_line += 1
            self.add(t)

        # add one line as separator
        self.pset_text_line += 1



    def update_process_lines(self, iteration, job_state, prev_job_state):
        # iterate over each process's state
        for proc, proc_state in job_state.process_states.items():
            prev_proc_state = prev_job_state.process_states[proc]

            # don't do anything if the process state hasn't changed
            if prev_proc_state.statetype == proc_state.statetype:
                continue

            if proc_state.is_visualized():
                # print("proc", proc, "is shown and has it's type changed from ", prev_proc_state.statetype, "to", proc_state.statetype)

                # get position of dot
                proc_pos = self.axes.c2p(self.time, proc+1)


                if not proc in self.proc_to_dot:
                    # setup dot
                    dot = Dot(radius=0.05, color=self.process_color[proc_state.statetype])

                    # move dot to correct position
                    dot.move_to(proc_pos)

                    self.proc_to_dot[proc] = dot

                    # add dot to scene
                    self.add(dot)

                dot = self.proc_to_dot[proc]

                if (prev_proc_state.statetype, proc_state.statetype) in [
                        (ProcessStateType.UNINITIALIZED, ProcessStateType.RUNNING),
                        (ProcessStateType.START_ANNOUNCED, ProcessStateType.RUNNING),
                    ]:
                    # add green startup thing
                    t1 = ArrowTriangleFilledTip().match_height(dot).rotate(180*DEGREES).set_color(PURE_GREEN)
                    t1.move_to(proc_pos)
                    self.add_foreground_mobjects(t1)

                # update color
                dot.set_color(self.process_color[proc_state.statetype])

                # define update function
                # need to capture some variables using default args


                # stop old tracer
                if proc in self.proc_to_trace:
                    self.proc_to_trace[proc].clear_updaters()
                    del self.proc_to_trace[proc]

                # add new tracer
                # https://docs.manim.community/en/stable/reference/manim.animation.movement.MoveAlongPath.html#manim.animation.movement.MoveAlongPath
                trace = VMobject()
                trace.move_to(proc_pos)

                if proc_state.statetype==ProcessStateType.START_ANNOUNCED:
                    # use dashed line in that case
                    trace.add_updater(lambda x,dot=dot, proc_pos=proc_pos.copy(), color=self.trace_color[proc_state.statetype]:
                                    x.become(DashedLine(proc_pos, dot.get_center())).set_color(color))
                else:
                    trace.add_updater(lambda x,dot=dot, proc_pos=proc_pos.copy(), color=self.trace_color[proc_state.statetype]:
                                    x.become(Line(proc_pos, dot.get_center())).set_color(color))


                # add trace
                self.add(trace)
                self.proc_to_trace[proc] = trace
            else:
                dot = self.proc_to_dot[proc]
                self.proc_to_dot.pop(proc)
                # stop tracer
                if proc in self.proc_to_trace:
                    self.proc_to_trace[proc].clear_updaters()
                    del self.proc_to_trace[proc]

                self.bring_to_front(dot)
                dot.become(Cross(dot, color=PURE_RED))

        # move each dot along its line segment
        animations = []
        for proc, proc_state in job_state.process_states.items():
            if proc_state.is_visualized():
                line_segment = self.line_segments[iteration][proc]
                line_segment_length = self.line_segment_lengths[iteration][proc]
                animation = MoveAlongPath(self.proc_to_dot[proc], line_segment, run_time=line_segment_length/self.speed_factor, rate_func=linear)
                animations.append(animation)

        # play animations in parallel
        if animations:
            self.play(*animations)



def main(manim_config, log_file, rounding):
    print("Reading timeline...")
    tl = Timeline(log_file, rounding)
    print("Done reading timeline.")
    print("Events:")
    print(tl.df)
    print("Starting visualization...")
    with tempconfig(manim_config):
        VisualizeDynProcs(tl).render()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-file", "-l", type=str, default="log.csv")
    parser.add_argument("--quality", "-q", type=str, default="low_quality", choices=["low_quality", "medium_quality", "high_quality"])
    parser.add_argument("--preview", "-p", action="store_true", default=False)
    parser.add_argument("--round-to", "-r", type=int, default=2)
    # TODO: add mode argument and ability to do moving camera
    # TODO: add option to specify start end end time of visualization

    args = parser.parse_args()

    manim_config = {
        "quality": args.quality,
        "preview": args.preview,
    }

    main(manim_config, log_file=args.log_file, rounding=args.round_to)
