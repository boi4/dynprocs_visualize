#!/usr/bin/env python3
import argparse
import textwrap
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
        self.pset_text_max_lines = 14
        self.application_text_line = 0
        self.application_text_max_lines = 7
        # we take the default matplotlib colors for cycling
        self.text_highlight_colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
        self.text_highlight_index = 0


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


    def process_events(self, orig_rows, jobstate):
        # process events that happen at that time point

        application_messages = []
        pset_messages = []
        for event_row in orig_rows:
            event_data = event_row["event_data"]
            match event_row["event"]:
                case "application_message":
                    line = event_data["message"]
                    application_messages.append(line)
                case "psetop":
                    op = event_data["op"]
                    set_id = event_data["set_id"]
                    input_sets = event_data["input_sets"]
                    output_sets = event_data["output_sets"]
                    line = f"<b>{op.upper()}</b>({', '.join(input_sets)}) <b>→</b> ({', '.join(output_sets)})"
                    pset_messages.append(line)
                case "finalize_psetop":
                    pset_messages.append(f"<b>Finalize</b>")

        # get next highlight color
        color = self.text_highlight_colors[self.text_highlight_index]
        self.text_highlight_index = (self.text_highlight_index + 1) % len(self.text_highlight_colors)

        # vertical line from top to bottom of axes at that time point
        vertline = Line(self.axes.c2p(self.time, 0)+0.2*DOWN, self.axes.c2p(self.time, self.maxproc+1)+0.2*UP,
                        color=color,
                        stroke_width=1)
        self.add(vertline)

        if application_messages:
            self.add_event_text("\n& ".join(application_messages), base=0.7, is_pset_text=False, color=color)
        if pset_messages:
            self.add_event_text("\n".join(pset_messages), base=2.2, color=color)


    def add_event_text(self, text, base, is_pset_text=True, color=RED):
        scale = 0.16
        line_height = scale
        max_width = 90
        max_lines = self.pset_text_max_lines if is_pset_text else self.application_text_max_lines
        cur_line = self.pset_text_line if is_pset_text else self.application_text_line

        # wrap long lines
        text_lines = text.strip().split("\n")
        ll = [textwrap.wrap(text=line, width=max_width) for line in text_lines]
        text = "\n".join(["\n".join(l) for l in ll])

        num_lines = text.count("\n") + 1


        if cur_line + num_lines > max_lines:
            cur_line = 0


        # create text
        cls = MarkupText if is_pset_text else Text
        t = cls(text, color=self.text_color).scale(scale)
        # create indicator that will be next to the text
        text_line = Line(self.axes.c2p(self.time, 0), self.axes.c2p(self.time, 0) + (t.height*DOWN),
                         color=color, stroke_width=2)

        text_line.move_to(self.axes.c2p(self.time, 0)).shift((base + cur_line * line_height) * DOWN)
        text_line.shift(0.5 * text_line.width * RIGHT)
        t.align_to(text_line, UP)
        t.align_to(text_line, LEFT)
        t.shift(0.05 * RIGHT)

        self.add(VGroup(text_line, t))

        cur_line += num_lines
        # add one line as separator
        cur_line += 1
        if is_pset_text:
            self.pset_text_line = cur_line
        else:
            self.application_text_line = cur_line



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



        # put active dots into group
        vg = VGroup(*self.proc_to_dot.values())

        # animate
        dt = self.timeline.ts[iteration+1] - self.timeline.ts[iteration]
        animation = vg.animate.shift((self.axes.c2p(dt,0)-self.axes.c2p(0,0))[0] * RIGHT)
        self.play(animation, run_time=dt/self.speed_factor, rate_func=linear)



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
    parser.add_argument("--round-to", "-r", type=int, default=3, help="On how many 10^r miliseconds to round the time to")
    # TODO: add mode argument and ability to do moving camera
    # TODO: add option to specify start end end time of visualization
    # TODO: add binary flag whether to show all legend items at beginning
    # TODO: remove the line segments variable

    args = parser.parse_args()

    manim_config = {
        "quality": args.quality,
        "preview": args.preview,
    }

    main(manim_config, log_file=args.log_file, rounding=args.round_to)
